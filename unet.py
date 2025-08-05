
import torch as th
import torch.nn as nn
from torch.nn import SiLU
#from timm import trunc_normal_
import einops
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import time
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResBlock(nn.Module):
    def __init__(self,channels,emb_channels,out_channels=None,dropout=0.0):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            SiLU(),
            nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=1, padding=1), #Conv2d
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
            )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.out_channels),   ##GroupNorm32(32, channels)
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1,padding=1)),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels,self.out_channels, kernel_size=1)

    def forward(self, x, emb):

        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):        # len(emb_out.shape)=2    len(h.shape)=4
            emb_out = emb_out[..., None]                # 补None

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = th.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
        x = self.skip_connection(x)

        return x + h
def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class UNetModel(nn.Module):

    def __init__(
            self,
            dropout=0.0,
            mlp_ratio=4.
    ):
        super().__init__()
        self.dropout = dropout,
        self.mlp_ratio=mlp_ratio,

        time_embed_dim = 128     # 512

        self.time_embed = nn.Sequential(
            nn.Linear(128, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        #encoder
        self.conv2d_1 = nn.Conv2d(3,128 , kernel_size=3, stride=1, padding=1)

        self.resblock1_1 = ResBlock(128, time_embed_dim, out_channels=128)
        self.resblock1_2 = ResBlock(128, time_embed_dim, out_channels=128)

        self.conv1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True) #下采样

        self.resblock2_1 = ResBlock(128, time_embed_dim, out_channels=256)
        self.resblock2_2 = ResBlock(256, time_embed_dim, out_channels=256)

        self.conv2 = nn.Conv2d(256, out_channels= 256, kernel_size=3, stride=2, padding=1, bias=True)

        self.resblock3_1 = ResBlock(256, time_embed_dim, out_channels=384)
        self.resblock3_2 = ResBlock(384, time_embed_dim, out_channels=384)

        self.conv3 = nn.Conv2d(384, out_channels=384, kernel_size=3, stride=2, padding=1, bias=True)

        self.resblock4_1 = ResBlock(384, time_embed_dim, out_channels=512)
        self.resblock4_2 = ResBlock(512, time_embed_dim, out_channels=512)


        #decoder
        self.resblock5_1 = ResBlock(512, time_embed_dim, out_channels=512)  # ch*4+ch*3
        self.resblock5_2 = ResBlock(512 + 512, time_embed_dim, out_channels=512)
        self.resblock5_3 = ResBlock(512 + 384, time_embed_dim, out_channels=512)

        self.upsample1 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(512, out_channels=512, kernel_size=3, stride=1, padding=1))

        self.resblock6_1 = ResBlock(512 + 384, time_embed_dim, out_channels=384)
        self.resblock6_2 = ResBlock(384 + 384, time_embed_dim, out_channels=384)
        self.resblock6_3 = ResBlock(384 + 256, time_embed_dim, out_channels=384)

        self.upsample2 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(384, out_channels=384, kernel_size=3, stride=1, padding=1))


        self.resblock7_1 = ResBlock(384 + 256, time_embed_dim, out_channels=256)
        self.resblock7_2 = ResBlock(256 + 256, time_embed_dim, out_channels=256)
        self.resblock7_3 = ResBlock(256 + 128, time_embed_dim, out_channels=256)

        self.upsample3 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1,padding=1))

        self.resblock8_1 = ResBlock(256 + 128, time_embed_dim, out_channels=128)
        self.resblock8_2 = ResBlock(128 + 128, time_embed_dim, out_channels=128)
        self.resblock8_3 = ResBlock(128 + 128, time_embed_dim, out_channels=128)


        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=128),
            SiLU(),
            zero_module(nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=True)),
        )

    def forward(self, x, timesteps):

        hs = []
        x = x.type(th.float32)     #(1,3,16,16)
        temb = self.time_embed(timestep_embedding(timesteps, 128))

        #encoder
        x = self.conv2d_1(x)      #(1,128,16,16)
        hs.append(x)

        x = self.resblock1_1(x, temb)    #(1,128,16,16)
        hs.append(x)
        x = self.resblock1_2(x, temb)     #(1,128,16,16)
        hs.append(x)
        x = self.conv1(x)                #(1,128,8,8)
        hs.append(x)

        x = self.resblock2_1(x, temb)    #(1,256,8,8)
        hs.append(x)
        x = self.resblock2_2(x, temb)     #(1,256,8,8)
        hs.append(x)
        x = self.conv2(x)                 #(1,256,4,4)
        hs.append(x)

        x = self.resblock3_1(x, temb)     #(1,384,4,4)
        hs.append(x)
        x = self.resblock3_2(x, temb)     #(1,384,4,4)
        hs.append(x)
        x = self.conv3(x)                 #(1,384,2,2)
        hs.append(x)

        x = self.resblock4_1(x, temb)     #(1,512,2,2)
        hs.append(x)
        x = self.resblock4_2(x, temb)     #(1,512,2,2)
        hs.append(x)


        #decoder
        x = self.resblock5_1(hs.pop(), temb)                               #(1,512,2,2)
        x = self.resblock5_2(torch.cat([x, hs.pop()], dim=1), temb)        #(1,512,2,2)
        x = self.resblock5_3(torch.cat([x, hs.pop()], dim=1), temb)        #(1,512,2,2)
        x = self.upsample1(x)                                               #(1,512,4,4)

        x = self.resblock6_1(torch.cat([x, hs.pop()], dim=1), temb)        #(1,384,4,4)
        x = self.resblock6_2(torch.cat([x, hs.pop()], dim=1), temb)         #(1,384,4,4)
        x = self.resblock6_3(torch.cat([x, hs.pop()], dim=1), temb)         #(1,384,4,4)
        x = self.upsample2(x)                                               #(1,384,8,8)

        x = self.resblock7_1(torch.cat([x,hs.pop()], dim=1), temb)      #(1,256,8,8)
        x = self.resblock7_2(torch.cat([x, hs.pop()], dim=1), temb)     #(1,256,8,8)
        x = self.resblock7_3(torch.cat([x, hs.pop()], dim=1), temb)      #(1,256,8,8)
        x = self.upsample3(x)                                           #(1,256,16,16)

        x = self.resblock8_1(torch.cat([x,hs.pop()], dim=1), temb)      #(1,128,16,16)
        x = self.resblock8_2(torch.cat([x, hs.pop()], dim=1), temb)     #(1,128,16,16)
        x = self.resblock8_3(torch.cat([x, hs.pop()], dim=1), temb)      #(1,128,16,16)
        x = self.out(x)                                                 #（1，3，16，16）

        x = x.type(x.dtype)
        return x