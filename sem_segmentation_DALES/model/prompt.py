from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from .Interaction import injector_weight, extractor



class InteractionBlock(nn.Module):
    def __init__(self, spm_level_num =4,spm_dim=384, pct_dim=384,norm_layer=partial(nn.LayerNorm, eps=1e-6),num_heads=6,drop=0., drop_path=0., with_ffn=True, ffn_ratio=0.25, init_values=0., extra_extractor=False):
        super().__init__()
        # extractor __init__(self, pct_dim, spm_dim, num_head, with_ffn=True, ffn_ratio=0.25, drop=0., drop_path=0.):
        # injector __init__(self, , spm_level_num,pct_dim, spm_dim, num_head, init_values):
        # spm_dim=[(B,G,C),(B,G//4,C),(B,G//8,C)]
        # self.injector = injector_weight(spm_level_num=spm_level_num,norm_layer=partial(nn.LayerNorm, eps=1e-6),pct_dim=pct_dim, spm_dim=spm_dim, num_head=num_heads, init_values=init_values)
        self.extractor = extractor(pct_dim=pct_dim, spm_dim=spm_dim,norm_layer=partial(nn.LayerNorm, eps=1e-6), num_head=num_heads, with_ffn=with_ffn, ffn_ratio=ffn_ratio, drop=drop, drop_path=drop_path)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                extractor(pct_dim=pct_dim, spm_dim=spm_dim, num_head=num_heads,norm_layer=partial(nn.LayerNorm, eps=1e-6), with_ffn=with_ffn, ffn_ratio=ffn_ratio, drop=drop, drop_path=drop_path)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

        self.prompt_pos = nn.Parameter(torch.randn(3, 1, pct_dim))# 可学习的Prompt参数

        # 初始化Prompt参数
        # self.dynamic_prompt = nn.Parameter(torch.randn(3, 1, pct_dim))  # 可学习的Prompt参数
        self.prompt_updater = nn.Sequential(
            nn.Conv1d(pct_dim, pct_dim, 1),  # 用卷积动态更新Prompt
            nn.LeakyReLU(negative_slope=0.02),
            # nn.LayerNorm(pct_dim)
        )
        self.layer_norm = nn.LayerNorm(pct_dim)
    def split_spm(self,spm_feature):
        G = spm_feature.shape[1] // 85
        spm_level_feature = []
        # spm_level_feature.append(spm_feature[:, :64 * G, :])
        spm_level_feature.append(spm_feature[:, 64 * G:80 * G, :])
        spm_level_feature.append(spm_feature[:, 80 * G:84 * G, :])
        spm_level_feature.append(spm_feature[:, 84 * G:, :])
        return spm_level_feature

    def forward(self, spm_feature, pct_feature, blocks, feature_list,idx_start):
        ##layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],deform_inputs1, deform_inputs2, H, W)
        # injector self, spm_feature, pct_feature
        # pct_feature = self.injector(spm_feature, pct_feature)
        # 插入多尺度prompt

        batch_size,_,_ = spm_feature.shape

        spm_feature_list = self.split_spm(spm_feature)
        prompt_list=[]
        # for i,item in enumerate(spm_feature_list):
        #     prompt_list.append(F.adaptive_max_pool1d(item.permute(0, 2, 1), 1).view(batch_size, -1).unsqueeze(1)+self.prompt_pos[i])

        # 根据不同的输入动态更新Prompt
        for i, item in enumerate(spm_feature_list):
            dynamic_prompt = F.adaptive_max_pool1d(item.permute(0, 2, 1), 1).view(batch_size, -1).unsqueeze(1) + self.prompt_pos[i]
            dynamic_prompt = self.prompt_updater(dynamic_prompt.permute(0, 2, 1)).permute(0, 2, 1)  # 更新Prompt
            dynamic_prompt = self.layer_norm(dynamic_prompt)
            prompt_list.append(dynamic_prompt)

        pct_feature = torch.cat((prompt_list[0],prompt_list[1],prompt_list[2], pct_feature), dim=1)
        # feature_list = []
        fetch_idx = [2, 5, 8, 11]

        # fetch_idx = [1, 3,5, 7,9, 11]
        for idx, blk in enumerate(blocks):
            pct_feature = blk(pct_feature)
            idx_tmp=idx_start+idx
            if idx_tmp in fetch_idx:
                feature_list.append(pct_feature[:, 3:, ].transpose(1, 2).contiguous())
        # extractor self, spm_feature, pct_feature
        # cls, pct_feature = pct_feature[:, :1, ], pct_feature[:, 1:, ]
        spm_feature = self.extractor(spm_feature, pct_feature)
        pct_feature=pct_feature[:,3:]
        return pct_feature, spm_feature, feature_list
