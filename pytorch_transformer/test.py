import argparse
import json
from dataset import get_data_loader
from modules.transformer import Transformer
from train import TrainManager
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import configure for training process')
    parser.add_argument('--config_path', type=str, help='Input config file path')
    parser.add_argument('--resume_checkpoint', type=str, help='Checkpoint path to continue training stage')
    args = parser.parse_args()
    print(f"Use configure in {args.config_path} for testing JAX...")

    with open(args.config_path, 'r') as file:
        config_args = json.load(file)
    
    src_vocab, tgt_vocab, train_loader, val_loader, test_loader = get_data_loader(config_args)

    model = Transformer(src_vocab_size=src_vocab.size,
                        tgt_vocab_size=tgt_vocab.size,
                        d_model=config_args["d_model"],
                        n_encoder_layers=config_args["n_encoder_layers"],
                        n_decoder_layers=config_args["n_decoder_layers"],
                        heads=config_args["heads"],
                        dropout=config_args["dropout"])

    train_manager = TrainManager(model, train_loader, val_loader, args.resume_checkpoint, config_args)
    loss, acc = train_manager.validate(test_loader=test_loader)
    print(f"Test loss : {loss} \nTest accuracy : {acc}")