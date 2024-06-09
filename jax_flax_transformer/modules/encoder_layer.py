from flax import linen as nn
from modules.feed_forward import FeedForward
from modules.multihead_attention import MultiheadAttention

class EncoderLayer(nn.Module):
    config: dict
    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.self_attn = MultiheadAttention(config=self.config)
        self.ff = FeedForward(self.config)
        self.dropout = nn.Dropout(self.config["dropout"])

    def __call__(self, x, mask, train):
        x_norm = self.norm1(x)
        
        # Multihead Attention
        x_attn = self.self_attn(
            kv=x_norm,
            q=x_norm,
            mask=mask,
            train=train
        )

        x = x + self.dropout(x_attn, deterministic=not train)
        x_norm = self.norm2(x)

        x_ff = self.ff(x_norm)
        x = x + self.dropout(x_ff, deterministic=not train)

        return x
    
