import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .position_encoding import PositionalEncoder

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.src_embedding = nn.Embedding(self.cfg.sp_vocab_size, self.cfg.d_model)
        self.tgt_embedding = nn.Embedding(self.cfg.sp_vocab_size, self.cfg.d_model)
        self.positional_encoder = PositionalEncoder(
            self.cfg.seq_len, 
            self.cfg.d_model, 
            self.cfg.device
        )
        self.encoder = Encoder(
            self.cfg.num_layers, 
            self.cfg.d_model, 
            self.cfg.num_heads, 
            self.cfg.d_ff, 
            self.cfg.drop_out
        )
        self.decoder = Decoder(
            self.cfg.num_layers, 
            self.cfg.d_model, 
            self.cfg.num_heads, 
            self.cfg.d_ff, 
            self.cfg.drop_out
        )
        self.output_linear = nn.Linear(self.cfg.d_model, self.cfg.sp_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, tgt_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input) # (B, L) => (B, L, d_model)
        tgt_input = self.tgt_embedding(tgt_input) # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
        tgt_input = self.positional_encoder(tgt_input) # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask) # (B, L, d_model)
        d_output = self.decoder(tgt_input, e_output, e_mask, d_mask) # (B, L, d_model)

        output = self.softmax(self.output_linear(d_output)) # (B, L, d_model) => # (B, L, trg_vocab_size)

        return output