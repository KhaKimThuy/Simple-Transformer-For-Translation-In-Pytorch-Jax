from flax import linen as nn

class FeedForward(nn.Module):
    config: dict
    def setup(self):
        self.dense1 = nn.Dense(features=self.config["d_ff"], use_bias=self.config["use_bias"])
        self.relu = nn.relu
        self.dropout = nn.Dropout(rate=self.config["dropout"])
        self.dense2 = nn.Dense(features=self.config["d_model"], use_bias=self.config["use_bias"])

    def __call__(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x, deterministic=not self.config["training"])
        x = self.dense2(x)
        x = self.dropout(x, deterministic=not self.config["training"])
        return x
