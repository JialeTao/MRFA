import math
import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ChannelBlock2d


class OcclusionAwareGenerator(nn.Module):
    def __init__(self, num_channels, block_expansion, max_features, num_up_blocks):
        super(OcclusionAwareGenerator, self).__init__()

        self.num_up_blocks = num_up_blocks
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        up_blocks = []
        resblock = []
        channel_block = []
        for i in range(num_up_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            decoder_in_feature = out_features
            if i==num_up_blocks-1:
                decoder_in_feature = out_features
            up_blocks.append(UpBlock2d(decoder_in_feature, in_features, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
            channel_block.append(ChannelBlock2d(decoder_in_feature*2,kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])
        self.resblock = nn.ModuleList(resblock[::-1])
        self.channel_block = nn.ModuleList(channel_block[::-1])
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

    def encode(self, x):
        f = []
        out = self.first(x)
        f.append(out)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            f.append(out)
        # 256 128 64 32,...,16,8
        return f[::-1]
    
    def decode(self, warp_f, warp_img, occlusion, warp_f_c=None, occlusion_c=None):

        # warp_f 32 64 128 256
        out = warp_f[0] * occlusion[0]
        # out = warp_f[0]
        if warp_f_c is not None:
            out_c = warp_f_c[0]
            out = torch.cat([out,out_c], dim=1)
        for i in range(self.num_up_blocks):
            if warp_f_c is not None:
                out = self.channel_block[i](out)
            out = self.resblock[i](out)
            out = self.up_blocks[i](out)
            out = warp_f[i+1] * occlusion[i+1] + out * (1-occlusion[i+1])
            if warp_f_c is not None and i != self.num_up_blocks - 1:
                out_c = warp_f_c[i+1]
                out = torch.cat([out,out_c], dim=1)
        out = self.final(out)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion[-1]) + warp_img*occlusion[-1]
        return out

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

