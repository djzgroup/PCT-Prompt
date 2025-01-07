import torch
from torch import nn
import torch.nn.functional as F
import sys

from model.point_cloud_transformer_prompt import PCTAdapter
from util.tools import square_distance, index_points

sys.path.append('..')


class get_semantic_model(nn.Module):
    def __init__(self, config):
        super(get_semantic_model, self).__init__()
        self.model=PCTAdapter(config)
        self.dim=config.transformer_config.trans_dim
        self.class_num=config.class_num
        self.channel_list=config.part_segmentation.channel_list
        self.de_inchannel_list=config.part_segmentation.de_inchannel_list
        self.de_outchannel_list = config.part_segmentation.de_outchannel_list
        self.global_feature_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()

        self.mlp_head = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(self.dim// 2,self.class_num)
        )

        self.global_fuse = nn.Sequential(
            nn.Conv1d(in_channels=320, out_channels=64, kernel_size=1, bias=True),

            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(self.dim+64, 128, 1, bias=True),

            nn.Dropout(),
            nn.Conv1d(128, self.class_num, 1, bias=True)
        )
        for i in range(len(self.channel_list)):
            channel = self.channel_list[i]
            de_in = self.de_inchannel_list[i]
            de_out = self.de_outchannel_list[i]
            global_down = nn.Sequential(
                nn.Conv1d(in_channels=channel, out_channels=64, kernel_size=1, bias=True),

                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
            decoder = PointNetFeaturePropagation(de_in, de_out, bias=True)
            self.global_feature_list.append(global_down)
            self.decoder_list.append(decoder)

        self.global_feature_list.append(nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=64, kernel_size=1, bias=True),

            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        ))

    def forward(self, x):
        pts= x[:, :3, :]
        norm = x[:, 3:6, :]
        f1, f2, f3, f4,xyz_list=self.model(pts=pts,norm=norm)
        x_list=[ f4 ,f3, f2,f1,pts]
        x = x_list[0]
        batch_size,_,_=f1.shape
        for i in range(len(self.decoder_list)):
            if i==3:
                x = self.decoder_list[i](xyz_list[i + 1].permute(0, 2, 1), xyz_list[i].permute(0, 2, 1), None, x)
            else:
                x = self.decoder_list[i](xyz_list[i + 1].permute(0, 2, 1), xyz_list[i].permute(0, 2, 1), x_list[i + 1], x)

        gf_list = []
        x_list[4]=x
        for i in range(len(x_list)):
            gf_list.append(F.adaptive_max_pool1d(self.global_feature_list[i](x_list[i]), 1))
        global_feature = self.global_fuse(torch.cat(gf_list, dim=1))
        x = torch.cat([x, global_feature.repeat([1, 1, x.shape[-1]])], dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=bias),

            nn.LeakyReLU(negative_slope=0.01)
        )

        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, bias=bias),

            nn.LeakyReLU(negative_slope=0.01)
        )

        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, bias=bias),

        )

        self.net3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, bias=bias),

            nn.LeakyReLU(negative_slope=0.01)
        )

        self.net4 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, bias=bias),

        )

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = F.leaky_relu((self.net2(self.net1(new_points)) + new_points), negative_slope=0.01)
        new_points = F.leaky_relu((self.net4(self.net3(new_points)) + new_points), negative_slope=0.01)
        return new_points

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, weight):
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -((one_hot * log_prb) * weight).sum(dim=1).mean()
        # total_loss = F.nll_loss(pred, target, weight=weight)

        return loss

def compute_TEs(data, net, batch_size):
    repetitions = 100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = net(data)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * batch_size) / total_time
    print('FinalThroughput:', Throughput)

