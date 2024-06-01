import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoder import Encoder
from .MiddleNetwork import MiddleNetwork
from .Decoder import Decoder

class ImageRestorationModel(nn.Module):
    """ Multiple Adverse Weather Removal Using Masked-Based Pre-Training and Dual-Pooling Adaptive Convolution (MPDAC)
    """
    def __init__(self, model_size='large', **kwargs):
        super().__init__()

        if model_size == 'small':
            embed_dims = [64, 128, 256, 512]
            Encoder_blk_nums = [2, 2, 6, 2]
            Middle_blk_nums = 3
            Decoder_blk_nums = [1, 1, 1, 1]

        elif model_size == 'large':
            embed_dims = [96, 192, 384, 768]
            Encoder_blk_nums = [3, 3, 9, 3]
            Middle_blk_nums = 3
            Decoder_blk_nums = [2, 2, 2, 2]

        self.Encoder = Encoder(dims=embed_dims, blk_nums=Encoder_blk_nums)

        self.MiddleNetwork = MiddleNetwork(embed_dims=embed_dims, blk_nums=Middle_blk_nums)

        self.Decoder = Decoder(channels=embed_dims, blk_nums=Decoder_blk_nums)

    def forward(self, input, return_Encoder_features=False):
        x1 = self.Encoder(input)
        x2 = self.MiddleNetwork(x1)
        x = self.Decoder(x1, x2)

        if return_Encoder_features:
            return x, x1
        else:
            return x
