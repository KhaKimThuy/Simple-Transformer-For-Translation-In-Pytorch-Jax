import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, \
                 d_model, n_encoder_layers, n_decoder_layers, \
                heads, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_encoder_layers, heads, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_decoder_layers, heads, dropout)
        self.out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        e_out = self.encoder(src, src_mask)
        d_out = self.decoder(tgt, e_out, src_mask, tgt_mask)
        out = self.out(d_out)
        return out