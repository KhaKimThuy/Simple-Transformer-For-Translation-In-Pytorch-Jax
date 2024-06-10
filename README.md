# Performance report of Transformer model for translate task based on Pytorch and Jax framwork  

Conclusion : I trained Transformer model for translate task, I built Transformer with Pytorch and Jax. In jax, I used Flax for building neural network layers, and training phase was too slow. After searching for the reason, a provided reason was `
The reason the JAX code compiles slowly is that during JIT compilation JAX unrolls loops. So in terms of XLA compilation, your function is actually very large: you call rnn_jax.apply() 1000 times, and compilation times tend to be roughly quadratic in the number of statements.
By contrast, your pytorch function uses no Python loops, and so under the hood it is relying on vectorized operations that run much faster.
Any time you use a for loop over data in Python, a good bet is that your code will be slow: this is true whether you're using JAX, torch, numpy, pandas, etc. I'd suggest finding an approach to the problem in JAX that relies on vectorized operations rather than relying on slow Python looping.`
Jax trained a batch about 7 minutes, so Jax will complete an epoch (batch_size = 8 -> 1 epoch = 13715 batchs) in 7x13715 = 96005 minutes = 1600 hours 5 minutes. Thus, I can not finish training Jax with the same number of epochs to Pytorch.

| Framework            | Training time (10 epochs) |
| ---------         | ------- |
| Pytorch         |    3 hr 25 '     |
| Jax (1 device)        |     ~ 1600 hr 25 '    |

[Training time comparison]

| Framework            | Train loss | Val loss | Test loss|
| ---------         | ------- |------- |--------|
| Pytorch         |    2.20     |     2.94       |    2.09  |
| Jax (1 device)        |     None    |  None    |   None   |

[Loss comparison]

| Framework            | Train accuracy | Val accuracy | Test accuracy|
| ---------         | ------- |------- |--------|
| Pytorch         |    0.64     |      0.62      |   0.66   |
| Jax (1 device)        |     None    |   None   |   None   |

[Accuracy comparison]
