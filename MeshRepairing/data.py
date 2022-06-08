import os

import numpy as np
import torch
from torch.utils.data import Dataset

from models.layers.mesh import Mesh


class MRepDataset(Dataset):

    def __init__(self, opt, train: bool, boundary: bool = False):
        super(MRepDataset, self).__init__()
        self.__opt = opt
        self.__boundary = boundary
        self.__root = opt.dataroot
        self.__paths = self.__create_dataset_random(train) if opt.data_random else self.__create_dataset(train)
        self.__n_edges = opt.n_edges
        self.__mean, self.__std = self.__get_mean_std(train)

    def __getitem__(self, index: int):
        if self.__opt.data_random:
            x_paths, y_path = self.__paths[index]
            x_path = x_paths[torch.randint(len(x_paths), (1,))]
        else:
            x_path, y_path = self.__paths[index]
        x_mesh = Mesh(self.__opt, file=x_path, boundary=self.__boundary)
        fe = (x_mesh.features - self.__mean) / self.__std
        y_mesh = Mesh(self.__opt, file=y_path)
        data = {'x': x_mesh, 'x_features': fe, 'y': y_mesh}
        if self.__opt.use_meta:
            meta = np.load(x_path[:-3] + 'npz', allow_pickle=True)
            data['face_meta'] = meta['faces']
            data['vert_meta'] = meta['verts']
        return data

    def __len__(self):
        return len(self.__paths)

    def __create_dataset(self, train: bool):
        test_or_train = 'train' if train else 'test'
        input_path = os.path.join(self.__root, test_or_train)
        paths = []
        for directory, _, files in os.walk(input_path):
            dir_name = os.path.basename(directory)
            if dir_name in [input_path, 'cache']:
                continue
            for file in files:
                if not file.endswith('.obj'):
                    continue
                x_path = os.path.join(directory, file)
                y_path = os.path.join(self.__root, dir_name + '.obj')
                paths.append((x_path, y_path))
        return paths

    def __create_dataset_random(self, train: bool):
        test_or_train = 'train' if train else 'test'
        paths = []
        for file in os.listdir(self.__root):
            if os.path.isfile(os.path.join(self.__root, file)) and file.endswith('.obj'):
                x_dir = os.path.join(self.__root, test_or_train, file[:-4])
                x_paths = [os.path.join(x_dir, f) for f in os.listdir(x_dir) if f.endswith('.obj')]
                y_path = os.path.join(self.__root, file)
                paths.append((x_paths, y_path))
        return paths

    def __get_mean_std(self, train: bool):
        """ Computes mean and standard deviation from training data
        :returns:
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        """
        cache_file = os.path.join(self.__root, 'cache', 'mean_std_cache.npz')
        if os.path.exists(cache_file):
            mean_std_dict = np.load(cache_file, encoding='latin1', allow_pickle=True)
            return mean_std_dict['mean'], mean_std_dict['std']
        print(100 * '-')
        print(f"Computing mean and standard deviation for {'train' if train else 'test'} data")
        mean, std = np.array(0.), np.array(0.)
        i = 0
        for path, _ in self.__paths:
            if self.__opt.data_random:
                for p in path:
                    features = Mesh(self.__opt, file=p, boundary=self.__boundary).features
                    mean = mean + features.mean(axis=1)
                    std = std + features.std(axis=1)
                    i += 1
            else:
                features = Mesh(self.__opt, file=path, boundary=self.__boundary).features
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
                i += 1
        mean = (mean / i)[:, np.newaxis]
        std = (std / i)[:, np.newaxis]
        os.mkdir(os.path.join(self.__root, 'cache'))
        np.savez_compressed(cache_file, mean=mean, std=std)
        return mean, std

    def mesh_collate(self, batch):
        """
        custom collate function as default_collate does not allow for special data types
        """
        x = np.array([b['x'] for b in batch])  # tensor not possible here
        x_features = torch.stack(self.pad_x_features([b['x_features'] for b in batch]))
        y = np.array([b['y'] for b in batch])  # tensor not possible here
        data = {'x': x, 'x_features': x_features, 'y': y}
        if self.__opt.use_meta:
            data['face_meta'] = [torch.from_numpy(b['face_meta']).float() for b in batch]
            data['vert_meta'] = [torch.from_numpy(b['vert_meta']).float() for b in batch]
        return data

    def pad_x_features(self, x_features):
        return [torch.from_numpy(self.pad(x_fe, self.__opt.n_edges)) for x_fe in x_features]

    @staticmethod
    def pad(input_arr, target_length, val=0, dim=1):
        n_pad = [(0, 0) for _ in range(len(input_arr.shape))]
        n_pad[dim] = (0, target_length - input_arr.shape[dim])
        return np.pad(input_arr, pad_width=n_pad, constant_values=val)
