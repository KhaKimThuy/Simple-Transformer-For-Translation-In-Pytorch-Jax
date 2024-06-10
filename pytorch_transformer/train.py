import torch
import os
import argparse
import json
from dataset import get_data_loader
from modules.transformer import Transformer
from utils import AvgMeter, create_mask
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn


class TrainManager:
    def __init__(self, model, train_loader, val_loader, resume_checkpoint, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = args["num_epochs"]
        self.save_interval = args["save_interval"]
        self.device = args["device"]

        self.model.to(self.device)

        self.start_epoch = 0

        self.lr = args["lr"]

        # self.optimizer = optim.RMSprop(params=model.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # self.scheduler = lr_scheduler.LinearLR(self.optimizer, \
        #                                         start_factor=1.0, \
        #                                         end_factor=0.5, \
        #                                         total_iters=self.num_epochs)


        self.criterion = nn.CrossEntropyLoss()

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        if resume_checkpoint is not None:
            self.load_checkpoint(resume_checkpoint)

    def load_checkpoint(self, ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

        self.train_loss = state["train_loss"]
        self.val_loss = state["val_loss"]
        self.train_acc = state["train_acc"]
        self.val_acc = state["val_acc"]

        self.start_epoch = len(self.train_loss) + 1

        print(f"Resume checkpoint from epoch {self.start_epoch}...")
        

    def train(self):
        best_acc = 0.0
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train step
            train_loss, train_acc = self.train_step(epoch)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            # Validation step
            val_loss, val_acc = self.validate()
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)

            # self.scheduler.step()

            if val_acc >= best_acc:
                best_acc = val_acc
                print(f"Hooray!!! New best model is find at epoch {epoch}")
                self.save_checkpoint(filename="best_ckpt.pt")

            if epoch % self.save_interval == 0:
                self.save_checkpoint(filename="most_ckpt.pt")



    def train_step(self, epoch):
        loss_tracker, acc_tracker = AvgMeter(), AvgMeter()
        self.model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader))
        for data in pbar:
            src = data["src"].to(self.device)
            trg = data["trg"].to(self.device)

            trg_input = trg[:, :-1] # Exclude last word to predict

            # create mask
            src_mask = create_mask(src, 1, False, self.device)
            trg_mask = create_mask(trg_input, 1, True, self.device)

            # Predict
            logits = self.model(src, trg_input, src_mask, trg_mask)
            logits = logits.view(-1, logits.size(-1))

            trg = trg[:,1:].contiguous().view(-1)

            # Calculate loss
            self.optimizer.zero_grad()
            loss = self.criterion(logits, trg)
            loss.backward()
            self.optimizer.step()

            # Track metrics
            loss_tracker.update(loss.item())

            predictions = torch.argmax(logits, dim=-1)
            correct_predictions = predictions == trg
            acc = correct_predictions.sum().item() / len(trg)
            acc_tracker.update(acc)

            # Update progressbar description
            pbar.set_description(f'Epoch: {epoch + 1:03d} - loss: {loss_tracker.avg:.3f} - acc: {acc_tracker.avg:.3f}%')
        
        return loss_tracker.avg, acc_tracker.avg

    def validate(self, test_loader=None):
        data_loader = self.val_loader if test_loader is None else test_loader
        loss_tracker, acc_tracker = AvgMeter(), AvgMeter()
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(data_loader, total=len(self.val_loader))
            for data in pbar:
                src = data["src"].to(self.device)
                trg = data["trg"].to(self.device)

                trg_input = trg[:, :-1] # Exclude last word to predict

                # create mask
                src_mask = create_mask(src, 1, False, self.device)
                trg_mask = create_mask(trg_input, 1, True, self.device)

                # Predict
                logits = self.model(src, trg_input, src_mask, trg_mask)
                logits = logits.view(-1, logits.size(-1))

                trg = trg[:,1:].contiguous().view(-1)

                # Calculate loss
                loss = self.criterion(logits, trg)

                # Track metrics
                loss_tracker.update(loss.item())
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions = predictions == trg
                acc = correct_predictions.sum().item() / len(trg)
                acc_tracker.update(acc)

                # Update progressbar description
                pbar.set_description(f'Val_loss: {loss_tracker.avg:.3f} - val_acc: {acc_tracker.avg:.3f}%')
        
        return loss_tracker.avg, acc_tracker.avg

    def save_checkpoint(self, filename="checkpoint.pt"):
        os.makedirs(f"checkpoint", exist_ok=True)
        state = {'state_dict': self.model.state_dict(), "train_loss": self.train_loss, \
                    "train_acc": self.train_acc,"val_loss": self.val_loss, "val_acc": self.val_acc, \
                    'optimizer': self.optimizer.state_dict()}

        torch.save(state, f"checkpoint/{filename}")
        print("Saving model...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import configure for training process')
    parser.add_argument('--config_path', type=str, help='Input config file path')
    parser.add_argument('--resume_checkpoint', type=str, help='Checkpoint path to continue training stage')
    args = parser.parse_args()
    print(f"Use configure in {args.config_path} for training...")

    with open(args.config_path, 'r') as file:
        config_args = json.load(file)
    
    # Load data
    src_vocab, tgt_vocab, train_loader, val_loader, test_loader = get_data_loader(config_args)

    model = Transformer(src_vocab_size=src_vocab.size,
                        tgt_vocab_size=tgt_vocab.size,
                        d_model=config_args["d_model"],
                        n_encoder_layers=config_args["n_encoder_layers"],
                        n_decoder_layers=config_args["n_decoder_layers"],
                        heads=config_args["heads"],
                        dropout=config_args["dropout"])

    train_manager = TrainManager(model, train_loader, val_loader, args.resume_checkpoint, config_args)
    train_manager.train()
