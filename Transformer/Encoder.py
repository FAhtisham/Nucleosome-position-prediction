import torch
import torch.nn as nn
from utils import clones

class Encoder(nn.Module):
    def __init__( self, layer, N):
        super(Encoder, self).__init__()
        
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)


def forward(self, x, mask=None):
    for layer in self.layers:
        x = layer(x, mask)
    return slef.norm(x)
    