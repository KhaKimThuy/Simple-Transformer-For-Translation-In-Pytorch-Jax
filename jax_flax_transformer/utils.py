import jax.numpy as jnp

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_np_mask(n):
    mask = jnp.triu(jnp.ones((n, n)), k=1).astype('uint8')
    mask = mask == 0
    return mask

def create_mask(idx_list, idx_pad, is_trg):
    mask = (idx_list != idx_pad)[:, None, :]
    if is_trg:
        size = idx_list.shape[1]  # seq_len
        np_mask = create_np_mask(size)
        mask = mask & np_mask
    return mask


