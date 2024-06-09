from flax import linen as nn
from modules.positional_encoding import PositionalEncoder
from modules.embedder import Embedder
from modules.decoder_layer import DecoderLayer

class Decoder(nn.Module):
    config: dict
    def setup(self):
        self.positional_encoding = PositionalEncoder(self.config["d_model"])
        self.layers = [DecoderLayer(self.config) for _ in range(self.config["n_decoder_layers"])]

    def __call__(self, x, e_out, src_mask, trg_mask, train):
        x = self.positional_encoding(x)
        # Decoder layer
        for module in self.layers:
            x = module(x, e_out, src_mask, trg_mask, train)
        return x
    