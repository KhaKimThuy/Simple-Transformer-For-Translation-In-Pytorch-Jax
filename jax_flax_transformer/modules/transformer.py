from flax import linen as nn
from modules.decoder import Decoder
from modules.encoder import Encoder
from utils import create_mask

class Transformer(nn.Module):
    config: dict

    def setup(self):
        self.src_embedder = nn.Embed(num_embeddings=self.config["src_vocab_size"], \
                                 features=self.config["emb_dim"])
        self.tgt_embedder = nn.Embed(num_embeddings=self.config["tgt_vocab_size"], \
                                 features=self.config["emb_dim"])
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

        self.out = nn.Dense(features=self.config["tgt_vocab_size"], use_bias=self.config["use_bias"])
        self.dropout = nn.Dropout(rate=self.config["dropout"])


    def __call__(self, src, tgt, train):
        # Initialize mask
        padding_mask = create_mask(src, 1, False)
        decoder_mask = create_mask(tgt, 1, True)

        # Embeding: [batch, length] -> [batch, length, feature]
        src_embedding = self.src_embedder(src.astype('int32'))
        tgt_embedding = self.tgt_embedder(tgt.astype('int32'))

        # Encoder stage
        e_out = self.encoder(x=src_embedding, mask=padding_mask, train=train)

        # Encoder stage
        x = self.decoder(
              x=tgt_embedding,
              e_out=e_out,
              src_mask=padding_mask,
              trg_mask=decoder_mask,
              train=train
            )

        # Linear
        x = self.out(x)
        x = self.dropout(x, deterministic=not self.config["training"])

        return x
    
