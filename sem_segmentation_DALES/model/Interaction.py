from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
import math
import torch
from einops import rearrange, repeat


class Pre_Linear(nn.Module):
    def __init__(self, d_model, num_heads, d_k):
        super(Pre_Linear, self).__init__()
        self.linear = nn.Linear(d_model, num_heads * d_k)
        self.heads = num_heads
        self.d_k = d_k

    def forward(self, x):
        head_shape = x.shape[: -1]#[N,C]-->N
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class CrossAttention(nn.Module):
    def __init__(self, in_feature, num_heads, d_model):
        super(CrossAttention, self).__init__()
        self.ln_cross_x = nn.LayerNorm( in_feature)  # c
        self.ln_cross_y = nn.LayerNorm( in_feature)  # c
        self.heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.scale = 1 / math.sqrt(self.d_k)
        self.linear_trans_q = Pre_Linear(in_feature, num_heads, self.d_k)
        self.linear_trans_k = Pre_Linear(in_feature, num_heads, self.d_k)
        self.linear_trans_v = Pre_Linear(in_feature, num_heads, self.d_k)

        self.drop_out = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Conv1d(in_feature, in_feature, 1)
        self.layer_norm = nn.LayerNorm(in_feature)
        self.act = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, cross_x,cross_y):  # B, N, C

        cross_x = cross_x.permute(1, 0, 2) # N, B, C
        cross_y = cross_y.permute(1, 0, 2)
        cross_norm_x = self.ln_cross_x(cross_x)
        cross_norm_y = self.ln_cross_y(cross_y)
        query = self.linear_trans_q(cross_norm_x)  # N, B, C -> (N, num_heads, d_k)
        key = self.linear_trans_k(cross_norm_y)  # N, B, H, d_k
        value = self.linear_trans_v(cross_norm_y)# N, B, H, d_k
        N, B, H, d_k = query.shape

        key=key.permute(1, 2,  0, 3)
        value = value.permute(1, 2, 0, 3)
        query = query.permute(1, 2, 0, 3)
        scores =  (query @ key.transpose(-2, -1)) * self.scale
        attention = self.softmax(scores)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        attention = self.drop_out(attention)
        x = attention @ value
        query = query.permute(0, 2, 1, 3).reshape(B, N, -1).permute(0, 2, 1)
        x = x.permute(0, 2, 1, 3).reshape(B, N, -1).permute(0, 2, 1)  # B, C, N
        x = self.conv(query - x)
        x = self.act(self.layer_norm(x.permute(0, 2, 1)))
        x = query.permute(0, 2, 1) + x
        return x

class FFN(nn.Module):
    def __init__(self, spm_dim,hidden_features,act_layer=nn.GELU,drop=0.):
        super().__init__()
        self.dwconv1 = nn.Conv1d(hidden_features, hidden_features, 1)
        self.act = act_layer()
        self.norm1=nn.BatchNorm1d(spm_dim)
        self.fc1 = nn.Linear(spm_dim, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, spm_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, spm_feature):
        spm_feature=self.fc1(spm_feature)
        G = spm_feature.shape[1] // 85
        spm_level_feature1= spm_feature[:, :64 * G, :]
        spm_level_feature2 = spm_feature[:, 64* G:80 * G, :]
        spm_level_feature3= spm_feature[:, 80* G:84*G, :]
        spm_level_feature4 = spm_feature[:, 84 * G:, :]
        spm_level_feature1 = self.dwconv1(spm_level_feature1.permute(0, 2, 1))
        spm_level_feature2 = self.dwconv1(spm_level_feature2.permute(0, 2, 1))
        spm_level_feature3 = self.dwconv1(spm_level_feature3.permute(0, 2, 1))
        spm_level_feature4 = self.dwconv1(spm_level_feature4.permute(0, 2, 1))
        spm_feature=torch.cat([spm_level_feature1.permute(0, 2, 1),spm_level_feature2.permute(0, 2, 1),spm_level_feature3.permute(0, 2, 1),spm_level_feature4.permute(0, 2, 1)],dim=1)
        spm_feature = self.act(spm_feature)
        spm_feature = self.drop(spm_feature)
        spm_feature= self.fc2(spm_feature)
        spm_feature= self.drop(spm_feature)
        return spm_feature

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


#cross_attention+ffn
class extractor(nn.Module):
    def __init__(self, pct_dim=384,spm_dim=384,num_head=6,with_ffn=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),ffn_ratio=0.25,drop=0.,drop_path=0.):
        super().__init__()
        #spm_dim=[(B,G,C),(B,G//4,C),(B,G//8,C)]
        self.pct_norm = norm_layer(pct_dim)
        self.spm_norm = norm_layer(spm_dim)
        self.cross_attention=CrossAttention(pct_dim, num_head,spm_dim)
        self.with_ffn=with_ffn
        if with_ffn:
            self.ffn = FFN(spm_dim, hidden_features=int(spm_dim * ffn_ratio), drop=drop)
            self.ffn_norm = nn.LayerNorm(spm_dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,spm_feature,pct_feature):
        #cross_attention(self, cross_x, cross_y):  # B, N, C
        pct_feature_cross = self.pct_norm(pct_feature)
        spm_feature_cross = self.spm_norm(spm_feature)
        spm_feature_cross= self.cross_attention( spm_feature_cross,pct_feature_cross)
        spm_feature = spm_feature + spm_feature_cross
        if self.with_ffn:
            spm_feature = spm_feature + self.drop_path(self.ffn(self.ffn_norm(spm_feature)))
        return spm_feature



#cross_attention
class injector(nn.Module):
    def __init__(self, spm_level_num=3,pct_dim=384,norm_layer=partial(nn.LayerNorm, eps=1e-6),spm_dim=384, num_head=6, init_values=0.):
        # spm_dim=[(B,G,C),(B,G//4,C),(B,G//8,C)]
        super().__init__()
        # self.pct_norm = norm_layer(pct_dim)
        self.pct_norm_list = nn.ModuleList()
        self.spm_norm_list = nn.ModuleList()
        self.cross_list = nn.ModuleList()
        self.weight_list=nn.ModuleList()
        self.weight_list_bn=nn.ModuleList()
        for i in range(spm_level_num):
             self.weight_list.append(nn.Conv1d(pct_dim, 1, 1))
             self.weight_list_bn.append(nn.BatchNorm1d(1))
             self.pct_norm_list.append(norm_layer(pct_dim))
             self.spm_norm_list.append(norm_layer(spm_dim))
             self.cross_list.append(CrossAttention(pct_dim, num_head, spm_dim))

        self.gamma = nn.Parameter(init_values * torch.ones((pct_dim)), requires_grad=True)


    def forward(self,spm_feature,pct_feature):
        #cross_attention(self, cross_x, cross_y):  # B, N, C
        #spm_feature should be cut
        G = spm_feature.shape[1] // 85

        spm_level_feature=[]
        spm_level_feature.append(spm_feature[:, :64 * G, :])
        spm_level_feature.append(spm_feature[:, 64 * G:80 * G, :])
        spm_level_feature.append(spm_feature[:, 80 * G:84 * G, :])
        spm_level_feature.append(spm_feature[:, 84 * G:, :])
        weight_pct_feature=[]
        pct_feature_cross = pct_feature
        for i, cross in enumerate(self.cross_list):
            spm_norm = self.spm_norm_list[i]
            pct_norm = self.pct_norm_list[i]
            pct_feature_cross=self.cross_list[i](pct_norm(pct_feature_cross),spm_norm(spm_level_feature[i]))
            weight = self.weight_list[i]
            weight_bn = self.weight_list_bn[i]
            weight = F.relu(weight_bn(weight(pct_feature_cross.permute(0,2,1))))
            weight=weight.permute(0,2,1)
            # weight_pct_feature.append(weight)
            pct_feature_cross+=pct_feature_cross
            # weight_pct_feature.append(weight*pct_feature_cross)
        pct_feature=pct_feature+ self.gamma*pct_feature_cross
        return pct_feature


class injector_weight(nn.Module):
    def __init__(self, spm_level_num=3,pct_dim=384,norm_layer=partial(nn.LayerNorm, eps=1e-6),spm_dim=384, num_head=6, init_values=0.):
        # spm_dim=[(B,G,C),(B,G//4,C),(B,G//8,C)]
        super().__init__()
        # self.pct_norm = norm_layer(pct_dim)
        self.pct_norm_list = nn.ModuleList()
        self.spm_norm_list = nn.ModuleList()
        self.cross_list = nn.ModuleList()
        # self.weight_list=[]
        self.weight_pct_feature=[]
        # self.weight_list_bn=nn.ModuleList()
        self.weight1=nn.Parameter(init_values * torch.ones((pct_dim)), requires_grad=True)
        self.weight2 = nn.Parameter(init_values * torch.ones((pct_dim)), requires_grad=True)
        self.weight3 = nn.Parameter(init_values * torch.ones((pct_dim)), requires_grad=True)
        self.weight4 = nn.Parameter(init_values * torch.ones((pct_dim)), requires_grad=True)
        for i in range(spm_level_num):
             self.pct_norm_list.append(norm_layer(pct_dim))
             self.spm_norm_list.append(norm_layer(spm_dim))
             self.cross_list.append(CrossAttention(pct_dim, num_head, spm_dim))


    def forward(self,spm_feature,pct_feature):
        #cross_attention(self, cross_x, cross_y):  # B, N, C
        #spm_feature should be cut
        G = spm_feature.shape[1] // 85

        spm_level_feature = []
        spm_level_feature.append(spm_feature[:, :64 * G, :])
        spm_level_feature.append(spm_feature[:, 64 * G:80 * G, :])
        spm_level_feature.append(spm_feature[:, 80 * G:84 * G, :])
        spm_level_feature.append(spm_feature[:, 84 * G:, :])

        pct_feature_cross = pct_feature
        spm_norm = self.spm_norm_list[0]
        pct_norm = self.pct_norm_list[0]
        pct_feature_cross1=self.cross_list[0](pct_norm(pct_feature_cross),spm_norm(spm_level_feature[0]))
        spm_norm1 = self.spm_norm_list[1]
        pct_norm1 = self.pct_norm_list[1]
        pct_feature_cross2 = self.cross_list[1](pct_norm1(pct_feature_cross), spm_norm1(spm_level_feature[1]))
        spm_norm2 = self.spm_norm_list[2]
        pct_norm2 = self.pct_norm_list[2]
        pct_feature_cross3 = self.cross_list[2](pct_norm2(pct_feature_cross), spm_norm2(spm_level_feature[2]))
        spm_norm3 = self.spm_norm_list[3]
        pct_norm3 = self.pct_norm_list[3]
        pct_feature_cross4 = self.cross_list[3](pct_norm3(pct_feature_cross), spm_norm3(spm_level_feature[3]))
        pct_feature = pct_feature + self.weight1*pct_feature_cross1+self.weight2*pct_feature_cross2+self.weight2*pct_feature_cross3+self.weight3*pct_feature_cross4
        return pct_feature