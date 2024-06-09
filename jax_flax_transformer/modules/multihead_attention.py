from flax import linen as nn
import jax.numpy as jnp

class MultiheadAttention(nn.Module):
    config: dict
    decode: bool = False
    dtype: jnp.dtype = jnp.float32


    def setup(self):
        self.q_dense = nn.Dense(features=self.config["d_model"], use_bias=self.config["use_bias"], name='query')
        self.k_dense = nn.Dense(features=self.config["d_model"], use_bias=self.config["use_bias"], name='key')
        self.v_dense = nn.Dense(features=self.config["d_model"], use_bias=self.config["use_bias"], name='value')

        self.q_norm = nn.LayerNorm()
        self.k_norm = nn.LayerNorm()
        self.v_norm = nn.LayerNorm()

        self.d_k = self.config["d_model"] // self.config["heads"]

        self.out_dense = nn.Dense(features=self.config["emb_dim"],
                          use_bias=self.config["use_bias"],
                          name='attention_weights')

        self.dropout = nn.Dropout(rate=self.config["dropout"])

    def __call__(self, kv, q, mask, train):

        # q: bs x seq_len x d_model
        # kv: bs x seq_len x d_model

        bs = q.shape[0]

        # Initialize key, query, value through Dense layer [batch=1, length=10, features=10]
        query = self.q_dense(q)
        key = self.k_dense(kv)
        value = self.v_dense(kv)

        # Layer norm
        query = self.q_norm(query)
        key = self.k_norm(key)
        value = self.v_norm(value)

        # Split head [batch=1, length=10, features=10] -> [batch=1, length=10, num_heads=2, depth_per_head=5]
        query = query.reshape(bs, -1, self.config["heads"], self.d_k)
        key = key.reshape(bs, -1, self.config["heads"], self.d_k)
        value = value.reshape(bs, -1, self.config["heads"], self.d_k)

        query = jnp.transpose(query, axes=(0,2,1,3))
        key = jnp.transpose(key, axes=(0,2,1,3))
        value = jnp.transpose(value, axes=(0,2,1,3))

        # Attention mechanism
        logits = self.scaled_dot_product_attention(query, key, value, mask)
        logits = jnp.transpose(logits,axes=(0,2,1,3)).reshape(bs, -1, self.config["d_model"])

        # Linear
        logits = self.out_dense(logits)
        logits = self.dropout(logits, deterministic=not train)

        return logits


    def scaled_dot_product_attention(self, query, key, value, mask):
        """ Matmul
            query: [batch, q_length, num_heads, qk_depth_per_head]
            key: [batch, kv_length, num_heads, qk_depth_per_head]
            -> qk: [batch, num_heads, q_length, kv_length]
        """
        d_k = query.shape[-1]
        scores = jnp.matmul(query, jnp.transpose(key, axes=(0,1,3,2))) / jnp.sqrt(d_k)

        # Mask
        if (mask is not None):
            mask = jnp.expand_dims(mask, axis=1)
            scores = jnp.where(mask==0, -1e9, scores)

        # Softmax will convert all -inf to 0
        scores = nn.softmax(scores, axis=-1).astype(self.dtype)
        output = jnp.matmul(scores, value)

        return output
    