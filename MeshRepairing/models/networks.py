from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv

from models.layers.mesh_conv import MeshConv, MResConv


class GlobalNet(nn.Module):
    """
    Tries to create an overall good mesh from a faulty mesh
    """

    def __init__(self, opt, init_weights_size=0.01, fc_n=2500):

        super(GlobalNet, self).__init__()
        self.convs = [5] + opt.convs_glob

        for i in range(len(self.convs) - 1):
            setattr(self, f'conv_{i}', MeshConv(self.convs[i], self.convs[i + 1]))
            setattr(self, f'norm_{i}', nn.BatchNorm2d(opt.convs_glob[i]))

        self.global_pool = nn.AvgPool1d(opt.n_edges)
        self.dropout = nn.Dropout(.4)
        self.dense_1 = nn.Linear(self.convs[-1], fc_n)
        self.dense_2 = nn.Linear(fc_n, 642 * 3)  # 642: number of verts in trimesh sphere

        init_weights(self, 'xavier', init_weights_size)

    def forward(self, x, meshes, intermediates):
        """
        forward PureConvNet
        :param x: batch x features x n_edges
        :param meshes: the faulty meshes
        :param intermediates: intermediate meshes
        :return: new vertex positions
        """

        old_verts = torch.stack([inter.vs.clone().detach() for inter in intermediates])
        copied_meshes = np.array([mesh.deep_copy() for mesh in meshes])
        x = x[:, :5]

        for i in range(len(self.convs) - 1):
            x = getattr(self, f'conv_{i}')(x, copied_meshes)
            x = F.relu(getattr(self, f'norm_{i}')(x))

        x = self.global_pool(x.squeeze())
        x = x.view(-1, self.convs[-1])

        x = self.dropout(x)
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)

        est_verts = x.reshape((-1, 642, 3))  # 642: number of verts in trimesh sphere
        return est_verts.float() + old_verts


class NVertsNet(nn.Module):
    """
    For a mesh where some faces are missing, this network predicts how many vertices need to be added
    """
    def __init__(self, opt, n_classes, fc_n, nresblocks=3):
        super(NVertsNet, self).__init__()
        self.convs = [6] + opt.convs_n_vs

        for i, ki in enumerate(self.convs[:-1]):
            setattr(self, f'conv_{i}', MResConv(ki, self.convs[i + 1], nresblocks))
            setattr(self, f'norm_{i}', nn.BatchNorm2d(opt.convs_n_vs[i]))

        self.global_pool = nn.AvgPool1d(opt.n_edges)
        self.dropout = nn.Dropout(.4)
        self.dense_1 = nn.Linear(self.convs[-1], fc_n)
        self.dense_2 = nn.Linear(fc_n, n_classes)

        init_weights(self, 'xavier', .1)

    def __call__(self, x, meshes):
        return self.forward(x, meshes)

    def forward(self, x, meshes):
        for i in range(len(self.convs) - 1):
            x = getattr(self, f'conv_{i}')(x, meshes)
            x = F.relu(getattr(self, f'norm_{i}')(x))

        x = self.global_pool(x.squeeze())
        x = x.view(-1, self.convs[-1])

        x = self.dropout(x)
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)
        return x


class VertPosNet(nn.Module):
    """
    graph convolution version of VertPosNet
    """
    def __init__(self, opt, n_classes, fc_n):
        super(VertPosNet, self).__init__()
        self.convs = [3] + opt.convs_v_pos

        for i, ki in enumerate(self.convs[:-1]):
            setattr(self, f'conv_{i}', GraphConv(ki, self.convs[i + 1]))

        self.dense = nn.Linear(self.convs[-1], fc_n)
        self.out_layers = [nn.Linear(fc_n, 3 * i) for i in range(1, n_classes)]

    def forward(self, meshes, n_verts: List[int]):
        for i, mesh in enumerate(meshes):
            edges = torch.from_numpy(mesh.edges).long()
            x = mesh.vs.float()
            boundary_verts = mesh.vs[mesh.boundary_loop]
            boundary_mean = boundary_verts.mean(dim=0)

            for j in range(len(self.convs) - 1):
                x = getattr(self, f'conv_{j}')(x, edges)
                x = F.relu(x)

            x = x[mesh.boundary_vs]  # keep only boundary verts
            x = torch.mean(x.squeeze(), dim=0).unsqueeze(0)  # global pool
            x = F.relu(self.dense(x))

            x = self.out_layers[n_verts[i] - 1](x) if n_verts[i] > 0 else torch.tensor([])
            x = x.reshape((n_verts[i], 3))
            yield boundary_mean + x


def face_areas_normals(faces, vs):
    face_normals = torch.cross(vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
                               vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :], dim=2)
    face_areas = torch.norm(face_normals, dim=2)
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = .5 * face_areas
    return face_areas, face_normals


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0., init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1., init_gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
