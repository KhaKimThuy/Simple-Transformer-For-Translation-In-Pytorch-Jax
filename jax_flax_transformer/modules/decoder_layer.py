from flax import linen as nn
from modules.feed_forward import FeedForward
from modules.multihead_attention import MultiheadAttention

class DecoderLayer(nn.Module):
    config: dict
    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()

        self.self_attn = MultiheadAttention(config=self.config)
        self.cross_attn = MultiheadAttention(config=self.config)
        self.ff = FeedForward(self.config)
        self.dropout = nn.Dropout(self.config["dropout"])

    def __call__(self, x, e_out, src_mask, trg_mask, train):

        # Self Multihead-Attention
        x_norm = self.norm1(x)
        x_attn = self.self_attn(
            kv=x_norm,
            q=x_norm,
            mask=trg_mask,
            train=train
        )
        x = x + self.dropout(x_attn, deterministic=not train)

        # Cross Multihead-Attention
        x_norm = self.norm2(x)
        x_attn = self.cross_attn(
            kv=e_out,
            q=x_norm,
            mask=src_mask,
            train=train
        )
        x = x + self.dropout(x_attn, deterministic=not train)

        # Feed forward
        x_norm = self.norm3(x)
        x_ff = self.ff(x_norm)
        x = x + self.dropout(x_ff, deterministic=not train)

        return x

