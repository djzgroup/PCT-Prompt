import os
import numpy as np
import torch
# import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util_surf import data_prepare

NUM_CLASS = 13
from plyfile import PlyData
ID2TRAINID = np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7])
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}

def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties

def read_dales_tile(
        filepath, xyz=True, intensity=True, semantic=True, instance=False,
        remap=False,):
    key = 'testing'
    with open(filepath, "rb") as f:
        tile = PlyData.read(f)
        if xyz:
            pos = torch.stack([
                torch.FloatTensor(tile[key][axis])
                for axis in ["x", "y", "z"]], dim=-1)

        if intensity:
            # Heuristic to bring the intensity distribution in [0, 1]
            intensity = torch.FloatTensor( tile[key]['intensity']).clip(min=0, max=60000) / 60000
            intensity=intensity.reshape(intensity.shape[0],1)

        if semantic:
            y = torch.LongTensor(tile[key]['sem_class'])
            y = torch.from_numpy(ID2TRAINID)[y] if remap else y


            # Check if the file start with ply
        # if b'ply' not in plyfile.readline():
        #     raise ValueError('The file does not start whith the word ply')
        #
        # # get binary_little/big or ascii
        # fmt = plyfile.readline().split()[1].decode()
        # if fmt == "ascii":
        #     raise ValueError('The file is not binary')
        #
        # # get extension for building the numpy dtypes
        # ext = valid_formats[fmt]
        #
        # # PointCloud reader vs mesh reader
        #
        #         # Parse header
        # num_points, properties = parse_header(plyfile, ext)
        #
        # # Get data
        # data = np.fromfile(plyfile, dtype=properties, count=num_points)

        # if instance:
        #     instance = torch.LongTensor(tile[key]['ins_class'])

    return pos.numpy().astype(np.float32),intensity.numpy().astype(np.float32),y.numpy().astype(np.float32)
    # return data
class DALES(Dataset):
    def __init__(self, args, split, coord_transform=None, rgb_transform=None,
                 rgb_mean=None, rgb_std=None, shuffle_index=False):
        super().__init__()
        self.args, self.split, self.coord_transform, self.rgb_transform, self.rgb_mean, self.rgb_std, self.shuffle_index = \
            args, split, coord_transform, rgb_transform, rgb_mean, rgb_std, shuffle_index
        self.stop_aug = False
        self.data_list = sorted(os.listdir(self.args.data_dir+split))
        self.data_idx = np.arange(len(self.data_list))


    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = np.load(os.path.join(self.args.data_dir+"\\"+self.split, self.data_list[data_idx])).astype(np.float32)
        coord, feat, label= read_dales_tile(os.path.join(self.args.data_dir+self.split, self.data_list[data_idx]), intensity=True, semantic=True, instance=False,
            remap=True)

        coord, feat, label = data_prepare(coord, feat, label, self.args, self.split, self.coord_transform, self.rgb_transform,self.rgb_mean, self.rgb_std, self.shuffle_index, self.stop_aug)
        return torch.cat([coord, feat],1), label

    def __len__(self):
        return len(self.data_idx) * self.args.loop

    @staticmethod
    def print_weight(data_root, data_list):
        print('Computing label weight...')
        num_point_list = []
        label_freq = np.zeros(NUM_CLASS)
        label_total = np.zeros(NUM_CLASS)
        # load data
        for idx, item in enumerate(data_list):
            data_path = os.path.join(data_root, item + '.npy')
            data = np.load(data_path)
            labels = data[:, 6]
            freq = np.histogram(labels, range(NUM_CLASS + 1))[0]
            label_freq += freq
            label_total += (freq > 0).astype(np.float) * labels.size
            num_point_list.append(labels.size)

        # label weight
        label_freq = label_freq / label_total
        label_weight = np.median(label_freq) / label_freq
        print(label_weight)

    @staticmethod
    def print_mean_std(data_root, data_list):
        print('Computing color mean & std...')
        point_list = []
        for idx, item in enumerate(data_list):
            data_path = os.path.join(data_root, item + '.npy')
            data = np.load(data_path)
            point_list.append(data[:, 3:6])

        points = np.vstack(point_list) / 255.
        mean = np.mean(points, 0)
        std = np.std(points, 0)
        print(f'mean: {mean}, std:{std}')