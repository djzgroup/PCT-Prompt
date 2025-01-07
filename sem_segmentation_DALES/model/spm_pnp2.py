from .pointnet import *

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_neighbors(x, feature, k=20, idx=None):
    '''
        input: x, [B,3,N]
               feature, [B,C,N]
        output: neighbor_x, [B,6,N,K]
                neighbor_feat, [B,2C,N,k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_x = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    neighbor_x = torch.cat((neighbor_x - x, x), dim=3).permute(0, 3, 1, 2)

    _, num_dims, _ = feature.size()

    feature = feature.transpose(2,
                                1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_feat = feature.view(batch_size * num_points, -1)[idx, :]
    neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims)
    feature = feature.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    neighbor_feat = torch.cat((neighbor_feat - feature, feature), dim=3).permute(0, 3, 1, 2)

    return neighbor_x, neighbor_feat


class Mish(nn.Module):
    '''new activation function'''

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx):
        ctx = ctx * (torch.tanh(F.softplus(ctx)))
        return ctx

    @staticmethod
    def backward(ctx, grad_output):
        input_grad = (torch.exp(ctx) * (4 * (ctx + 1) + 4 * torch.exp(2 * ctx) + torch.exp(3 * ctx) +
                                        torch.exp(ctx) * (4 * ctx + 6))) / (2 * torch.exp(ctx) + torch.exp(2 * ctx) + 2)
        return input_grad


class PnP3D(nn.Module):
    def __init__(self, input_features_dim):
        super(PnP3D, self).__init__()

        self.mish = Mish()

        self.conv_mlp1 = nn.Conv2d(6, input_features_dim // 2, 1)
        self.bn_mlp1 = nn.BatchNorm2d(input_features_dim // 2)

        self.conv_mlp2 = nn.Conv2d(input_features_dim * 2, input_features_dim // 2, 1)
        self.bn_mlp2 = nn.BatchNorm2d(input_features_dim // 2)

        self.conv_down1 = nn.Conv1d(input_features_dim, input_features_dim // 8, 1, bias=False)
        self.conv_down2 = nn.Conv1d(input_features_dim, input_features_dim // 8, 1, bias=False)

        self.conv_up = nn.Conv1d(input_features_dim // 8, input_features_dim, 1)
        self.bn_up = nn.BatchNorm1d(input_features_dim)

    def forward(self, xyz, features, k):
        # Local Context fusion
        neighbor_xyz, neighbor_feat = get_neighbors(xyz, features, k=k)

        neighbor_xyz = F.relu(self.bn_mlp1(self.conv_mlp1(neighbor_xyz)))  # B,C/2,N,k
        neighbor_feat = F.relu(self.bn_mlp2(self.conv_mlp2(neighbor_feat)))  # B,C/2,N,k

        f_encoding = torch.cat((neighbor_xyz, neighbor_feat), dim=1)  # B,C,N,k
        f_encoding = f_encoding.max(dim=-1, keepdim=False)[0]  # B,C,N

        # Global Bilinear Regularization
        f_encoding_1 = F.relu(self.conv_down1(f_encoding))  # B,C/8,N
        f_encoding_2 = F.relu(self.conv_down2(f_encoding))  # B,C/8,N

        f_encoding_channel = f_encoding_1.mean(dim=-1, keepdim=True)[0]  # B,C/8,1
        f_encoding_space = f_encoding_2.mean(dim=1, keepdim=True)[0]  # B,1,N
        final_encoding = torch.matmul(f_encoding_channel, f_encoding_space)  # B,C/8,N
        final_encoding = torch.sqrt(final_encoding + 1e-12)  # B,C/8,N
        final_encoding = final_encoding + f_encoding_1 + f_encoding_2  # B,C/8,N
        final_encoding = F.relu(self.bn_up(self.conv_up(final_encoding)))  # B,C,N

        f_out = f_encoding - final_encoding

        # Mish Activation
        f_out = self.mish(f_out)

        return f_out


class get_SPM(nn.Module):
    def __init__(self,init_group_num=512,init_neighbors_num=32,normal_channel=True,embed_dim=64,alpha=1000,beta=100):
        super(get_SPM, self).__init__()
        in_channel = 4 if normal_channel else 3
        self.normal_channel = normal_channel
        self.embed_dim=embed_dim//4
        self.out_dim=embed_dim
        self.alpha=alpha
        self.beta=beta
        self.init_group_num=init_group_num
        self.init_neighbors_num=init_neighbors_num

        self.embeding=nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=1, bias=True),
                nn.ReLU(inplace=True)
        )

        self.sa1 = PointNetSetAbstraction_knn_GF(group_num=self.init_group_num * 16, neighbors_num=self.init_neighbors_num, in_channel=64, mlp=[self.embed_dim // 2, self.embed_dim // 2, self.embed_dim], group_all=False, use_xyz=True, normalize="anchor")
        self.sa2 = PointNetSetAbstraction_knn_GF(group_num=self.init_group_num * 4, neighbors_num=self.init_neighbors_num, in_channel=self.embed_dim, mlp=[self.embed_dim, self.embed_dim, self.embed_dim * 2], group_all=False, use_xyz=True, normalize="anchor")
        self.sa3 = PointNetSetAbstraction_knn_GF(group_num=self.init_group_num, neighbors_num=self.init_neighbors_num, in_channel=self.embed_dim * 2, mlp=[self.embed_dim * 2, self.embed_dim * 2, self.embed_dim * 4], group_all=False, use_xyz=True, normalize="anchor")
        self.sa4 = PointNetSetAbstraction_knn_GF(group_num=self.init_group_num // 4, neighbors_num=self.init_neighbors_num,in_channel=self.embed_dim * 4,mlp=[self.embed_dim * 4, self.embed_dim * 4, self.embed_dim * 8],group_all=False, use_xyz=True, normalize="anchor")

        self.pnps1 = PnP3D(self.embed_dim)
        self.pnps2 = PnP3D(self.embed_dim * 2)
        self.pnps3 = PnP3D(self.embed_dim * 4)
        self.pnps4 = PnP3D(self.embed_dim * 8)

        self.change_dim1 = nn.Linear(self.embed_dim, self.out_dim)
        self.change_dim2 = nn.Linear(2 * self.embed_dim, self.out_dim)
        self.change_dim3 = nn.Linear(4 * self.embed_dim, self.out_dim)
        self.change_dim4 = nn.Linear(8 * self.embed_dim, self.out_dim)

    def forward(self, xyz):
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l0_points = self.embeding(l0_points)

        # 第一层
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.pnps1(l1_xyz, l1_points, k=self.init_neighbors_num)

        # 第二层
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.pnps2(l2_xyz, l2_points, k=self.init_neighbors_num)

        # 第三层
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.pnps3(l3_xyz, l3_points, k=self.init_neighbors_num)

        # 第四层
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.pnps4(l4_xyz, l4_points, k=self.init_neighbors_num)

        l1_points = self.change_dim1(l1_points.permute(0, 2, 1))
        l2_points = self.change_dim2(l2_points.permute(0, 2, 1))
        l3_points = self.change_dim3(l3_points.permute(0, 2, 1))
        l4_points = self.change_dim4(l4_points.permute(0, 2, 1))

        return l1_points, l2_points, l3_points, l4_points, l1_xyz, l2_xyz, l3_xyz, l4_xyz


