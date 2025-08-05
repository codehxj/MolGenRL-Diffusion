#VAE模型

from inspect import isfunction

import torch
import torch.utils.data
from torch.nn import functional as F
from torch import nn, einsum
from einops import rearrange
def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        #print(inner_dim)
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        w=context   #atte1:None         atte2:context
        h = self.heads      #8

        q = self.to_q(x)    # 1，4096 ，128
        #context = default(context, x) # 1，4096，128
        k = self.to_k(x)  # atte1:1，4096，128   atte2:1,77,128
        v = self.to_v(x)  # atte1:1，4096，128   atte2:1,77,128

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale        # atte1:8，4096，4096，
                                                                        # atte2:8, 4096，77   self.scale=8

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class MolecularVAE(nn.Module):
    def __init__(self):
        super(MolecularVAE, self).__init__()

        self.conv1_1 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                     nn.Conv2d(292, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.conv2_1 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                     nn.Conv2d(128, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.conv3_1 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                     nn.Conv2d(64, out_channels=32, kernel_size=3, stride=1, padding=1))
        self.conv4_1 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                     nn.Conv2d(32, out_channels=3, kernel_size=3, stride=1, padding=1))


        self.conv1_2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=292, kernel_size=3, stride=2, padding=1, bias=True)

        # The input filter dim should be 35
        #  corresponds to the size of CHARSET
        self.conv1d1 = nn.Conv1d(35, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        # self.atte1 = Attention(112)
        # self.atte2 = Attention(104)
        # self.atte3 = Attention(94)


        self.fc0 = nn.Linear(940, 435)
        self.fc11 = nn.Linear(435, 292)
        self.fc12 = nn.Linear(435, 292)

        self.fc2 = nn.Linear(292, 292)
        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.fc3 = nn.Linear(501, 35)

#改变通道
    def channel_encoder(self,x):
        x = x.view(x.size(0), 292, 1, 1)
        x = self.conv1_1(x)  # 1 128 2 2
        x = nn.ReLU()(x)
        x = self.conv2_1(x)  # 1  64 4 4
        x = nn.ReLU()(x)
        x = self.conv3_1(x)  # 1 32 8 8
        x = nn.ReLU()(x)
        x = self.conv4_1(x)  # 1 3 16 16
        return x
#改回通道
    def channel_decoder(self,x):
        x = self.conv1_2(x)  # （1，32，8，8）
        x = nn.ReLU()(x)
        x = self.conv2_2(x)  # （1，64，4，4）
        x = nn.ReLU()(x)
        x = self.conv3_2(x)  # （1，128，2，2）
        x = nn.ReLU()(x)
        x = self.conv4_2(x)  # （1，292，1，1）
        x = x.view(x.size(0), 292)
        return x

#编码
    def encode(self, x):
        #print(x.shape)     # (1,35,120)   35是字典中的原子以及符号的种类数  ，120是最大原子数
        h = F.relu(self.conv1d1(x))   #（1，9，112）    self.conv1d1 = nn.Conv1d(35, 9, kernel_size=9)
        h = F.relu(self.conv1d2(h))   #(1,9,104)       self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        h = F.relu(self.conv1d3(h))   #(1,10,94)       self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        h = h.view(h.size(0), -1)  #展平 (1,940)
        h = F.selu(self.fc0(h))   #linear层  （1，435）   self.fc0 = nn.Linear(940, 435)
        return self.fc11(h), self.fc12(h)  #均值和对数方差  都是（1，292）
                                        #self.fc11 = nn.Linear(435, 292)
                                        #self.fc12 = nn.Linear(435, 292)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)       #计算标准差
            eps = 1e-2 * torch.randn_like(std) #生成与标准差相同形状的随机噪声
            w = eps.mul(std).add_(mu)        #生成一个正太分布的采样，
            return w
        else:
            return mu
#解码
    def decode(self, z):
        z = F.selu(self.fc2(z))                  # （1.292）           self.fc2 = nn.Linear(292, 292)
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)      #（1，120，292）
        out, h = self.gru(z)             #  out (1,120,501)  h(3,1,501)   self.gru = nn.GRU(292, 501, 3, batch_first=True)
        out_reshape = out.contiguous().view(-1, out.size(-1))      #(120,501)
        y0 = F.softmax(self.fc3(out_reshape), dim=1)        #(120,35)      self.fc3 = nn.Linear(501, 35)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))#（1，120，35）
        return y

    def forward(self, x):

        mu, logvar = self.encode(x)           #通过VAE编码器，编码得到均值和对数方差  维度（1，292）
        z = self.reparametrize(mu, logvar)    #重采样得到潜在表示
        x = mu.detach()                       #detach,将编码解码   与通道的编码解码分开  ，取mu而没取Z,可以使结果更可控，不重要          
        x = self.channel_encoder(x)           #adapter 编码       （1，292） ->(1,3,16,16)
        x = self.channel_decoder(x)           #adapter 解码        (1,3,16,16)  -> (1,292)
        return self.decode(z), mu, logvar ,x   