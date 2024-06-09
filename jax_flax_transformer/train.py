from jax import random
import argparse
import json
import os
import jax
from flax.training import train_state, checkpoints
from dataset import get_data_loader
import numpy as np
from modules.transformer import Transformer
import optax
from tqdm import tqdm
from utils import AvgMeter

class TrainManager:
    def __init__(self, config, exmp_batch, **model_kwargs):
        """
        Inputs:
            model_name - Name of the model. Used for saving and checkpointing
            exmp_batch - Example batch to the model for initialization
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            seed - Seed to use for model init
        """
        super().__init__()
        self.model_name = config["model_name"]
        self.max_iters = config["num_epochs"]
        self.lr = config["lr"]
        self.start_epoch = 0
        self.len_loader = config["len_loader"]
        self.seed = config["seed"]
        self.save_interval = config["save_interval"]
        self.tgt_vocab_size = config["tgt_vocab_size"]
        self.model = Transformer(config=config)

        self.ckpt_dir = os.path.join('../checkpoints', self.model_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)


        self.create_functions()
        self.init_model(exmp_batch)

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        
        
    def get_loss_function(self):
        # Return a function that calculates the loss for a batch
        # To be implemented in a task specific sub-class
        raise NotImplementedError
        
    def create_functions(self):
        # Create jitted train and eval functions
        calculate_loss = self.get_loss_function()

        # Training function
        def train_step(state, rng, batch):
            loss_fn = lambda params: calculate_loss(params, rng, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, rng = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads)

            return state, rng, loss, acc
        
        self.train_step = jax.jit(train_step)
        
        # Evaluation function
        def eval_step(state, rng, batch):
            loss, (acc, rng) = calculate_loss(state.params, rng, batch, train=False)
            return loss, acc, rng
        self.eval_step = jax.jit(eval_step)
        
    def init_model(self, exmp_batch):
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, dropout_init_rng = jax.random.split(self.rng, 3)
        
        src, tgt = exmp_batch
        params = self.model.init({'params': init_rng, \
                                  'dropout': dropout_init_rng}, \
                                  src=src, tgt=tgt, \
                                  train=True)['params']

        # Initialize learning rate schedule and optimizer
        # lr_schedule = optax.linear_schedule(init_value=1.0, \
        #                                     end_value=0.5, \
        #                                     transition_steps=self.max_iters)

        optimizer = optax.adam(self.lr)

        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)
        
    def train_model(self, train_loader, val_loader, num_epochs=500):
        # Train model for defined number of epochs
        best_acc = 0.0
        for epoch_idx in tqdm(range(self.start_epoch, num_epochs)):
            # Train step
            train_loss, train_acc = self.train_epoch(train_loader, epoch=epoch_idx)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            # Validation step
            eval_loss, eval_acc = self.eval_model(val_loader)
            self.val_loss.append(eval_loss)
            self.val_acc.append(eval_acc)

            if eval_acc >= best_acc:
                print(f"Hooray!!! New best model is find at epoch {epoch_idx} with accuracy = {eval_acc}")
                best_acc = eval_acc
                self.save_model(step=epoch_idx, alias="best")

            if epoch_idx % self.save_interval == 0:
                self.save_model(step=epoch_idx, alias="most")
            
                
    def train_epoch(self, train_loader, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        loss_tracker, acc_tracker = AvgMeter(), AvgMeter()
        pbar = tqdm(train_loader, total=len(train_loader))
        for batch in pbar:
            self.state, self.rng, loss, accuracy = self.train_step(self.state, self.rng, batch)
            
            # Track metrics
            loss_tracker.update(loss, 1)
            acc_tracker.update(accuracy, 1)

            pbar.set_description(f'Epoch: {epoch + 1:03d} - train_loss: {loss_tracker.avg:.3f} - train_acc: {acc_tracker.avg:.3f}%')
        
        return loss_tracker.avg, acc_tracker.avg

    def eval_model(self, data_loader):
        # Test model on all data points of a data loader and return avg accuracy
        loss_tracker, acc_tracker = AvgMeter(), AvgMeter()
        pbar = tqdm(data_loader, total=len(data_loader))
        for batch in pbar:
            loss, acc, self.rng = self.eval_step(self.state, self.rng, batch)

            loss_tracker.update(loss, 1)
            acc_tracker.update(acc, 1)

            pbar.set_description(f'Validation - val_loss: {loss_tracker.avg:.3f} - val_acc: {acc_tracker.avg:.3f}%')

        return loss_tracker.avg, acc_tracker.avg
    
    def save_model(self, step=0, alias="most"):
        checkpoint_data = {
            "state": self.state.params, \
            "train_loss": self.train_loss, \
            "train_acc": self.train_acc, \
            "val_loss": self.val_loss, \
        }

        # Save the checkpoint
        checkpoints.save_checkpoint(
            ckpt_dir=self.ckpt_dir,
            target=checkpoint_data,
            step=step,
            overwrite=True,
            prefix=alias
        )

        # Save current model at certain training iteration
        # checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, step=step)
        
    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for the pretrained model
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_dir, target=self.state.params)
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.ckpt_dir, f'{self.model_name}.ckpt'), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params["state"], tx=self.state.tx)
        
    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this Transformer
        return os.path.isfile(os.path.join(self.ckpt_dir, f'{self.model_name}.ckpt'))

class TranslateTrainer(TrainManager):    
    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(params, rng, batch, train):
            src, tgt = batch
            tgt_input = tgt[:, :-1] # Exclude <sos> token

            rng, dropout_apply_rng = random.split(rng)

            logits = self.model.apply({"params":params}, src=src, tgt=tgt_input, train=train, rngs={'dropout': dropout_apply_rng})
            tgt = tgt[:,1:] # Exclude <eos> token

            loss = optax.softmax_cross_entropy_with_integer_labels(logits, tgt).mean()

            acc = (logits.argmax(axis=-1) == tgt).mean()

            return loss, (acc, rng)
        return calculate_loss

def train_jax(config, max_epochs=10, **model_args):
    num_train_iters = len(train_loader) * max_epochs
    # Create a trainer module with specified hyperparameters
    trainer = TranslateTrainer(model_name='TranslateTask', 
                               config=config,
                                exmp_batch=next(iter(train_loader)),
                                max_iters=num_train_iters, 
                                **model_args)
    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        trainer.train_model(train_loader, val_loader, num_epochs=max_epochs)
        trainer.load_model()
    else:
        trainer.load_model(pretrained=True)
    val_acc = trainer.eval_model(val_loader)
    test_acc = trainer.eval_model(test_loader)
    
    # Bind parameters to model for easier inference
    trainer.model_bd = trainer.model.bind({'params': trainer.state.params})
    return trainer, {'val_acc': val_acc, 'test_acc': test_acc}

from jax import random
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import configure for training process')
    parser.add_argument('--config_path', type=str, help='Input config file path')
    parser.add_argument('--resume_checkpoint', type=str, help='Checkpoint path to continue training stage')
    args = parser.parse_args()
    print(f"Use configure in {args.config_path} for training JAX...")

    with open(args.config_path, 'r') as file:
        config_args = json.load(file)
    
    # Load data
    src_vocab, tgt_vocab, train_loader, val_loader, test_loader = get_data_loader(config_args)
    
    config_args["src_vocab_size"] = src_vocab.size
    config_args["tgt_vocab_size"] = tgt_vocab.size
    config_args["len_loader"] = len(train_loader)

    train_jax(config=config_args)

