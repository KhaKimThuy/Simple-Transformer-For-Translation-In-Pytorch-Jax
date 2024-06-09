from flax import linen as nn
from modules.positional_encoding import PositionalEncoder
from modules.embedder import Embedder
from modules.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    config: dict
    def setup(self):
        self.positional_encoding = PositionalEncoder(d_model=self.config["d_model"])
        self.layers = [EncoderLayer(self.config) for _ in range(self.config["n_encoder_layers"])]

    def __call__(self, x, mask, train):
        # x = Embedder(self.config["src_vocab_size"], self.config["emb_dim"])(x)
        x = self.positional_encoding(x)
        # Encoder layer
        for module in self.layers:
            x = module(x, mask, train)
        return x


