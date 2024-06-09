from flax import linen as nn
class Embedder(nn.Module):
    vocab_size: int
    emb_dim : int
    @nn.compact
    def __call__(self, x):
        nn.Embed(num_embeddings=self.vocab_size, features=self.emb_dim)(
            x.astype('int32')
        )
        return x