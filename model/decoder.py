import torch
import torch.nn as nn
from .attention import MultiheadAttention
from .layer_norm import LayerNormalization
from .feed_forward import FeedFowardLayer

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop_out=0.1):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model)
        self.masked_multihead_attention = MultiheadAttention(d_model, num_heads, drop_out)
        self.drop_out_1 = nn.Dropout(drop_out)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, num_heads, drop_out)
        self.drop_out_2 = nn.Dropout(drop_out)

        self.layer_norm_3 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, drop_out)
        self.drop_out_3 = nn.Dropout(drop_out)

    def forward(self, x, e_output, e_mask,  d_mask):
        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        ) # (B, L, d_model)
        x_3 = self.layer_norm_3(x) # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3)) # (B, L, d_model)

        return x # (B, L, d_model)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, drop_out):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, drop_out) for i in range(num_layers)]
        )
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)