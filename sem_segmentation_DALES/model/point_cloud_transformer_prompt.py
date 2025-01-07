#adapter moudle
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, trunc_normal_

from .pointnet import PointNetFeaturePropagation
from .spm_pnp2 import get_SPM
from .prompt import InteractionBlock
from .point_cloud_transformer import PointTransformer


class PCTAdapter(PointTransformer):
    def __init__(self,  config):
        super().__init__(config)
        # pretrain=config.pretrain
        self.add_pct_feature=config.add_pct_feature
        self.version=config.version
        config=config.adapter_config
        self.init_values=config.init_values
        self.ffn_ratio=config.ffn_ratio
        self.drop=config.drop
        self.drop_path=config.drop_path
        self.with_ffn=config.with_ffn
        self.alpha=config.alpha
        self.beta=config.beta
        # self.extra_extractor=config.extra_extractor
        self.interaction_indexes=config.interaction_indexes

        # self.deform_num_heads=config.deform_num_heads
        self.num_block = len(self.blocks.blocks)
        # self.pretrain_size = (pretrain_size, pretrain_size)
        #插入adapter的阶段
        self.interaction_indexes = config.interaction_indexes
        embed_dim = self.trans_dim
        self.level_embed = nn.Parameter(torch.zeros(4, embed_dim))
        self.spm = get_SPM(init_group_num=self.num_group,init_neighbors_num=self.group_size,normal_channel=True,embed_dim=embed_dim)
        # self.get_patch=get_patch_token(init_group_num=self.num_group,init_neighbors_num=self.group_size,normal_channel=True,embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlock(spm_level_num=4, spm_dim=embed_dim, pct_dim=embed_dim, num_heads=self.num_heads, norm_layer=self.norm_layer, drop=self.drop, drop_path=self.drop_path, with_ffn=self.with_ffn,ffn_ratio=self.ffn_ratio, init_values=self.init_values, extra_extractor=True if i == len(self.interaction_indexes) - 1 else False,)
            for i in range(len(self.interaction_indexes))
        ])
        # self.interpolate1=PointNetFeaturePropagation(embed_dim, [embed_dim,embed_dim])
        # self.interpolate2=PointNetFeaturePropagation(embed_dim, [embed_dim,embed_dim])
        self.propagation_3 = PointNetFeaturePropagation(in_channel=self.trans_dim+3,
                                                        mlp=[self.trans_dim * 4, self.trans_dim])
        self.propagation_2 = PointNetFeaturePropagation(in_channel=self.trans_dim,
                                                        mlp=[self.trans_dim * 4, self.trans_dim])
        self.propagation_1 = PointNetFeaturePropagation(in_channel=self.trans_dim ,
                                                        mlp=[self.trans_dim * 4, self.trans_dim])

        # self.conv_fuse1 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
        #                                nn.BatchNorm1d(embed_dim),
        #                                nn.LeakyReLU(negative_slope=0.2))
        # self.conv_fuse2 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
        #                                nn.BatchNorm1d(embed_dim),
        #                                nn.LeakyReLU(negative_slope=0.2))
        # self.conv_fuse3 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
        #                                 nn.BatchNorm1d(embed_dim),
        #                                 nn.LeakyReLU(negative_slope=0.2))
        # self.conv_fuse4 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
        #                                 nn.BatchNorm1d(embed_dim),
        #                                 nn.LeakyReLU(negative_slope=0.2))
        normal_(self.level_embed)
        # self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_weights)
        # self.apply(self.load_model_from_ckpt)

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)
        # elif isinstance(m, torch.nn.Conv1d):
        #     torch.nn.init.kaiming_normal_(m.weight)
        #     if m.bias is not None:
        #         torch.nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Conv2d):
        #     fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     fan_out //= m.groups
        #     m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        return

    def split_spm(self,spm_feature):
        G = spm_feature.shape[1] // 85
        spm_level_feature = []
        spm_level_feature.append(spm_feature[:, :64 * G, :])
        spm_level_feature.append(spm_feature[:, 64 * G:80 * G, :])
        spm_level_feature.append(spm_feature[:, 80 * G:84 * G, :])
        spm_level_feature.append(spm_feature[:, 84 * G:, :])
        return spm_level_feature

    def _add_level_embed(self, l1_points, l2_points, l3_points,l4_points):
        l1_points = l1_points + self.level_embed[0]
        l2_points = l2_points + self.level_embed[1]
        l3_points = l3_points + self.level_embed[2]
        l4_points = l4_points + self.level_embed[3]
        return l1_points, l2_points, l3_points,l4_points

    def forward(self, pts,norm):
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

        # SPM forward
        pts_norm = torch.cat([pts, norm], dim=1)
        spm_feature_l1,spm_feature_l2,spm_feature_l3,spm_feature_l4,xyz_8G,xyz_4G,xyz_2G,xyz_G = self.spm(pts_norm)
        # spm_feature_l1, spm_feature_l2, spm_feature_l3 = self.spm(pts)
        spm_feature_l1,spm_feature_l2,spm_feature_l3,spm_feature_l4= self._add_level_embed(spm_feature_l1,spm_feature_l2,spm_feature_l3,spm_feature_l4)
        spm_feature = torch.cat([spm_feature_l1,spm_feature_l2,spm_feature_l3,spm_feature_l4], dim=1)

        # Patch Embedding forward
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts.permute(0, 2, 1).contiguous())
        #neighborhood, center = self.group_divider(pts.permute(0,2,1).contiguous(),norm.permute(0,2,1).contiguous())
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N  MLP提取特征
        group_input_tokens = self.reduce_dim(group_input_tokens)  # MLP
        # group_input_tokens,center=self.get_patch(pts_norm)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        # GEO_embedding = self.embedding(center)
        pos = self.pos_embed(center)
        # final input
        # pct_feature= torch.cat((cls_tokens, group_input_tokens), dim=1)
        # pos = torch.cat((cls_pos, pos), dim=1)
        # pct_feature=pct_feature+pos
        pct_feature=group_input_tokens+pos
        cls=cls_tokens+cls_pos
        # Interaction
        outs = list()
        feature_list = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            pct_feature, spm_feature, feature_list = layer(spm_feature, pct_feature,
                                                                self.blocks.blocks[indexes[0]:indexes[-1] + 1],
                                                                feature_list,indexes[0])
            outs.append(pct_feature.transpose(1, 2).contiguous())


        # Split & Reshape

        x1, x2, x3, x4 = feature_list
        center=center.permute(0,2,1)
        x1 = self.propagation_1(xyz1=xyz_8G, xyz2=center, points1=None, points2=x1)
        x2 = self.propagation_2(xyz1=xyz_4G,xyz2=center,points1=None,points2=x2)
        #*1
        # spm_feature_l4 = self.propagation_3(xyz1=center,xyz2=xyz_G,points1=None,points2=spm_feature_l4.permute(0, 2, 1))
        x4 = self.propagation_3(xyz1=xyz_G, xyz2=center, points1=xyz_G, points2=x4)
        spm_feature_l1,spm_feature_l2,spm_feature_l3,spm_feature_l4=self.split_spm(spm_feature)
        c1, c2, c3, c4 = spm_feature_l1.permute(0,2,1)+x1,  spm_feature_l2.permute(0,2,1) + x2,  spm_feature_l3.permute(0,2,1) + x3,  spm_feature_l4.permute(0,2,1) + x4
        # c1, c2, c3, c4 = x1, x2, x3, x4
        # Final Norm
        xyz_list=[xyz_G,xyz_2G,xyz_4G,xyz_8G,pts]
        # xyz_list.append(xyz_G),xyz_list.append(xyz_2G),xyz_list.append(xyz_4G),xyz_list.append(xyz_8G),xyz_list.append(pts)

        f1=c1
        f2 = c2
        f3 =c3
        f4 = c4

        return  f1,f2, f3, f4,xyz_list
        # return f1