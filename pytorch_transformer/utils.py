import torch
import numpy as np
from torch.autograd import Variable
import copy
import torch.nn as nn
import math
import torch
import torch.nn.functional as F

def attention(q, k, v, mask=None, dropout=None):
    """
    q: batch x head x seq_len x d_model
    k: batch x head x seq_len x d_model
    v: batch x head x seq_len x d_model

    mask: batch x 1 x seq_len
    output: batch x head x seq_len x d_model
    """
    d_k = q.size(-1) # last dimension
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k) # batch x head x seq_len x seq_len
    
    # Mask the future scores (to train model) and padding
    if (mask is not None):
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)
    
    # Softmax will convert all -inf to 0
    scores = F.softmax(scores, dim = -1) # Softmax over last dimension
    
    if (dropout is not None):
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)
    return output, scores

def get_clones(module, n):
    module_list = []
    for _ in range(n):
        module_list.append(copy.deepcopy(module))
    return nn.ModuleList(module_list)

def create_np_mask(n, device):
    mask = np.triu(
        np.ones(shape=(1, n, n)),
        k=1
    ).astype('uint8')
    mask = Variable(torch.from_numpy(mask)==0)
    mask = mask.to(device)
    return mask

def create_mask(idx_list, idx_pad, is_trg, device):
    mask = (idx_list != idx_pad).unsqueeze(-2)
    mask = mask.to(device)
    if (is_trg):
        size = idx_list.size(1) # seq_len
        np_mask = create_np_mask(size, device)
        if (mask.is_cuda):
            np_mask.cuda()
        mask = mask & np_mask
    return mask

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

def accuracy(outputs, target_sequences, k=5):
    batch_size = outputs.size(1)
    _, indices = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = indices.eq(target_sequences.view(-1, 1).expand_as(indices))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)




