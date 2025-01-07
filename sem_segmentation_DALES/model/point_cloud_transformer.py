from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_

from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''

        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv1 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv1(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv2(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x



class PointTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        # config = config.transformer_config
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.cls_dim = config.transformer_config.cls_dim
        self.num_heads = config.transformer_config.num_heads
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.group_size = config.transformer_config.group_size
        self.num_group = config.transformer_config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )


        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )
        # self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.
        # self.blocks = nn.ModuleList([
        #     Block(
        #         dim=self.trans_dim, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        #         drop=0., attn_drop=0.,
        #         drop_path=dpr[i] if isinstance(dpr, list) else dpr
        #     )
        #     for i in range(self.depth)])
        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    def load_model_from_ckpt(self,bert_ckpt_path):
        bert_ckpt_path=self.config.pretrain
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

    # def forward(self, pts):
    #     # divide the point clo  ud in the same form. This is important
    #     neighborhood, center = self.group_divider(pts)
    #     # encoder the input cloud blocks
    #     group_input_tokens = self.encoder(neighborhood)  # B G N
    #     group_input_tokens = self.reduce_dim(group_input_tokens)
    #     # prepare cls
    #     cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
    #     cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
    #     # add pos embedding
    #     pos = self.pos_embed(center)
    #     # final input
    #     x = torch.cat((cls_tokens, group_input_tokens), dim=1)
    #     pos = torch.cat((cls_pos, pos), dim=1)
    #     # transformer
    #     x = self.blocks(x, pos)
    #     x = self.norm(x)
    #     concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
    #     ret = self.cls_head_finetune(concat_f)
    #     return ret


# class Point_transformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         print_log(f'[Point_BERT] build dVAE_BERT ...', logger='Point_BERT')
#         self.config = config
#         self.m = config.m
#         self.T = config.T
#         self.K = config.K
#
#         self.moco_loss = config.transformer_config.moco_loss
#         self.dvae_loss = config.transformer_config.dvae_loss
#         self.cutmix_loss = config.transformer_config.cutmix_loss
#
#         self.return_all_tokens = config.transformer_config.return_all_tokens
#         if self.return_all_tokens:
#             print_log(f'[Point_BERT] Point_BERT calc the loss for all token ...', logger='Point_BERT')
#         else:
#             print_log(f'[Point_BERT] Point_BERT [NOT] calc the loss for all token ...', logger='Point_BERT')
#
#         self.transformer_q = MaskTransformer(config)
#         self.transformer_q._prepare_encoder(self.config.dvae_config.ckpt)
#
#         self.transformer_k = MaskTransformer(config)
#         for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
#             param_k.data.copy_(param_q.data)  # initialize
#             param_k.requires_grad = False  # not update by gradient
#
#         self.dvae = DiscreteVAE(config.dvae_config)
#         self._prepare_dvae()
#
#         for param in self.dvae.parameters():
#             param.requires_grad = False
#
#         self.group_size = config.dvae_config.group_size
#         self.num_group = config.dvae_config.num_group
#
#         print_log(
#             f'[Point_BERT Group] cutmix_BERT divide point cloud into G{self.num_group} x S{self.group_size} points ...',
#             logger='Point_BERT')
#         self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
#
#         # create the queue
#         self.register_buffer("queue", torch.randn(self.transformer_q.cls_dim, self.K))
#         self.queue = nn.functional.normalize(self.queue, dim=0)
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
#
#         # loss
#         self.build_loss_func()
#
#     # def _prepare_dvae(self):
#     #     dvae_ckpt = self.config.dvae_config.ckpt
#     #     ckpt = torch.load(dvae_ckpt, map_location='cpu')
#     #     base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
#     #     self.dvae.load_state_dict(base_ckpt, strict=True)
#     #     print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger='Point_BERT')
#
#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         """
#         Momentum update of the key encoder
#         """
#         for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
#             param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
#
#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys):
#         # gather keys before updating queue
#         keys = concat_all_gather(keys)
#
#         batch_size = keys.shape[0]
#
#         ptr = int(self.queue_ptr)
#         assert self.K % batch_size == 0  # for simplicity
#
#         # replace the keys at ptr (dequeue and enqueue)
#         self.queue[:, ptr:ptr + batch_size] = keys.T
#         ptr = (ptr + batch_size) % self.K  # move pointer
#
#         self.queue_ptr[0] = ptr
#
#     def build_loss_func(self):
#         self.loss_ce = nn.CrossEntropyLoss()
#         self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none')
#
#     def forward_eval(self, pts):
#         with torch.no_grad():
#             neighborhood, center = self.group_divider(pts)
#             cls_feature = self.transformer_q(neighborhood, center, only_cls_tokens=True, noaug=True)
#             return cls_feature
#
#     def _mixup_pc(self, neighborhood, center, dvae_label):
#         '''
#             neighborhood : B G M 3
#             center: B G 3
#             dvae_label: B G
#             ----------------------
#             mixup_ratio: /alpha:
#                 mixup_label = alpha * origin + (1 - alpha) * flip
#
#         '''
#         mixup_ratio = torch.rand(neighborhood.size(0))
#         mixup_mask = torch.rand(neighborhood.shape[:2]) < mixup_ratio.unsqueeze(-1)
#         mixup_mask = mixup_mask.type_as(neighborhood)
#         mixup_neighborhood = neighborhood * mixup_mask.unsqueeze(-1).unsqueeze(-1) + neighborhood.flip(0) * (
#                     1 - mixup_mask.unsqueeze(-1).unsqueeze(-1))
#         mixup_center = center * mixup_mask.unsqueeze(-1) + center.flip(0) * (1 - mixup_mask.unsqueeze(-1))
#         mixup_dvae_label = dvae_label * mixup_mask + dvae_label.flip(0) * (1 - mixup_mask)
#
#         return mixup_ratio.to(neighborhood.device), mixup_neighborhood, mixup_center, mixup_dvae_label.long()
#
#     def forward(self, pts, noaug=False, **kwargs):
#         if noaug:
#             return self.forward_eval(pts)
#         else:
#             # divide the point cloud in the same form. This is important
#             neighborhood, center = self.group_divider(pts)
#             # produce the gt point tokens
#             with torch.no_grad():
#                 gt_logits = self.dvae.encoder(neighborhood)
#                 gt_logits = self.dvae.dgcnn_1(gt_logits, center)  # B G N
#                 dvae_label = gt_logits.argmax(-1).long()  # B G
#             # forward the query model in mask style 1.
#             if self.return_all_tokens:
#                 q_cls_feature, logits = self.transformer_q(neighborhood, center,
#                                                            return_all_tokens=self.return_all_tokens)  # logits :  N G C
#             else:
#                 q_cls_feature, real_logits, flake_logits, mask = self.transformer_q(neighborhood, center,
#                                                                                     return_all_tokens=self.return_all_tokens)  # logits :  N' C where N' is the mask.sum()
#             q_cls_feature = nn.functional.normalize(q_cls_feature, dim=1)
#
#             mixup_ratio, mixup_neighborhood, mixup_center, mix_dvae_label = self._mixup_pc(neighborhood, center,
#                                                                                            dvae_label)
#             if self.return_all_tokens:
#                 mixup_cls_feature, mixup_logits = self.transformer_q(mixup_neighborhood, mixup_center,
#                                                                      return_all_tokens=self.return_all_tokens)
#             else:
#                 mixup_cls_feature, mixup_real_logits, mixup_flake_logits, mixup_mask = self.transformer_q(
#                     mixup_neighborhood, mixup_center, return_all_tokens=self.return_all_tokens)
#             mixup_cls_feature = nn.functional.normalize(mixup_cls_feature, dim=1)
#
#             # compute key features
#             with torch.no_grad():  # no gradient to keys
#                 self._momentum_update_key_encoder()  # update the key encoder
#                 k_cls_feature = self.transformer_k(neighborhood, center, only_cls_tokens=True)  # keys: NxC
#                 k_cls_feature = nn.functional.normalize(k_cls_feature, dim=1)
#
#             if self.moco_loss:
#                 # ce loss with moco contrast
#                 l_pos = torch.einsum('nc, nc->n', [q_cls_feature, k_cls_feature]).unsqueeze(-1)  # n 1
#                 l_neg = torch.einsum('nc, ck->nk', [q_cls_feature, self.queue.clone().detach()])  # n k
#                 ce_logits = torch.cat([l_pos, l_neg], dim=1)
#                 ce_logits /= self.T
#                 labels = torch.zeros(l_pos.shape[0], dtype=torch.long).to(pts.device)
#                 moco_loss = self.loss_ce(ce_logits, labels)
#             else:
#                 moco_loss = torch.tensor(0.).to(pts.device)
#
#             if self.dvae_loss:
#                 if self.return_all_tokens:
#                     dvae_loss = self.loss_ce(logits.reshape(-1, logits.size(-1)), dvae_label.reshape(-1, )) + \
#                                 self.loss_ce(mixup_logits.reshape(-1, mixup_logits.size(-1)),
#                                              mix_dvae_label.reshape(-1, ))
#                 else:
#                     dvae_loss = self.loss_ce(flake_logits, dvae_label[mask]) + \
#                                 self.loss_ce(mixup_flake_logits, mix_dvae_label[mixup_mask])
#             else:
#                 dvae_loss = torch.tensor(0.).to(pts.device)
#
#             if self.cutmix_loss:
#                 l_pos = torch.einsum('nc, mc->nm', [mixup_cls_feature, k_cls_feature])  # n n
#                 l_neg = torch.einsum('nc, ck->nk', [mixup_cls_feature, self.queue.clone().detach()])  # n k
#                 ce_logits = torch.cat([l_pos, l_neg], dim=1)
#                 ce_logits /= self.T
#                 labels = torch.arange(l_pos.shape[0], dtype=torch.long).to(pts.device)
#                 cutmix_loss = (mixup_ratio * self.loss_ce_batch(ce_logits, labels) + (
#                             1 - mixup_ratio) * self.loss_ce_batch(ce_logits, labels.flip(0))).mean()
#             else:
#                 cutmix_loss = torch.tensor(0.).to(pts.device)
#             self._dequeue_and_enqueue(k_cls_feature)
#             return moco_loss + dvae_loss, cutmix_loss