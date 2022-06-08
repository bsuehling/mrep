import os.path
import warnings
from typing import List

import numpy as np
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from data import MRepDataset
from linalg import *
from models.layers.mesh import Mesh
from models.networks import NVertsNet, VertPosNet
from options import Options


def predict_n_verts(batch):
    predicted = n_vs_net(batch['x_features'][:, :6], batch['x'])
    real = torch.tensor([len(vert_meta) for vert_meta in batch['vert_meta']])
    return predicted, real


def predict_vert_positions(batch, n_vs: List[int]):
    for mesh in batch['x']:
        # scale = mesh.scale if type(mesh.scale) is torch.Tensor else torch.from_numpy(mesh.scale)
        if type(mesh.scale) is torch.Tensor:
            scale = mesh.scale
        elif type(mesh.scale) is np.ndarray:
            scale = torch.from_numpy(mesh.scale)
        else:
            scale = torch.tensor([mesh.scale])
        trans = mesh.translation if type(mesh.translation) is torch.Tensor else torch.from_numpy(mesh.translation)
        mesh.vs /= scale
        mesh.vs -= trans
    predicted = [out for out in vert_pos_net(batch['x'], n_vs)]
    for i, mesh in enumerate(batch['x']):
        # scale = mesh.scale if type(mesh.scale) is torch.Tensor else torch.from_numpy(mesh.scale)
        if type(mesh.scale) is torch.Tensor:
            scale = mesh.scale
        elif type(mesh.scale) is np.ndarray:
            scale = torch.from_numpy(mesh.scale)
        else:
            scale = torch.tensor([mesh.scale])
        trans = mesh.translation if type(mesh.translation) is torch.Tensor else torch.from_numpy(mesh.translation)
        predicted[i] = predicted[i] + trans
        predicted[i] = predicted[i] * scale
    real = batch['vert_meta']
    return predicted, real


def predict_faces(batch, n_vs: List[int]):
    faces_predicted, faces_real = [], []
    features, candidates = new_vs_features(batch['x'], n_vs)

    for i, mesh in enumerate(batch['x']):
        loop, fe, cands = mesh.boundary_loop, features[i], candidates[i]
        new_faces = []

        # connect each predicted vertex with a boundary edge:
        for _ in range(n_vs[i]):
            sel_e, new_v = fe.T.argmin() % fe.shape[0], fe.argmin() % fe.shape[1]
            fe[sel_e, :] = float('inf')
            fe[:, new_v] = float('inf')
            new_face = cands[sel_e, new_v].tolist()
            mesh.add_edge([new_face[0], new_face[2]])
            mesh.add_edge([new_face[1], new_face[2]])
            idx = [j for j in range(len(loop)) if loop[j] == new_face[0] and loop[(j + 1) % len(loop)] == new_face[1]]
            loop.insert((idx[0] + 1) % len(loop), new_face[2])
            mesh.add_face(new_face)
            new_faces.append(new_face)

        # add new edges and faces to the remaining hole:
        fe, cands = [], []
        for j, vert in enumerate(loop):
            cand = [vert, loop[(j + 1) % len(loop)], loop[(j + 2) % len(loop)]]
            cands.append(cand)
            fe.append(get_badness(mesh.vs[cand]))
        fe, cands = torch.tensor(fe, device=device), torch.tensor(cands, device=device)
        while len(loop) > 3:
            sel_e = fe.argmin()
            new_face = cands[sel_e].tolist()
            new_faces.append(new_face)
            mesh.add_edge([new_face[0], new_face[2]])
            mesh.add_face(new_face)
            # print(sel_e, len(loop))
            cand1 = [loop[sel_e - 1], loop[sel_e], loop[(sel_e + 2) % len(loop)]]
            cand2 = [loop[sel_e], loop[(sel_e + 2) % len(loop)], loop[(sel_e + 3) % len(loop)]]
            fe1 = get_badness(mesh.vs[cand1]).unsqueeze(0)
            fe2 = get_badness(mesh.vs[cand2]).unsqueeze(0)
            idx = [j for j in range(len(loop)) if loop[j] == new_face[1] and loop[j - 1] == new_face[0]]
            loop.pop(idx[0])
            if sel_e == 0:
                cands = torch.cat([torch.tensor([cand2]), cands[2:-1], torch.tensor([cand1])])
                fe = torch.cat([fe2, fe[2:-1], fe1])
            elif sel_e < len(loop) - 1:
                cands = torch.cat([cands[:sel_e - 1], torch.tensor([cand1, cand2]), cands[sel_e + 2:]])
                fe = torch.cat([fe[:sel_e - 1], fe1, fe2, fe[sel_e + 2:]])
            elif sel_e == len(loop) - 1:
                cands = torch.cat([cands[:-3], torch.tensor([cand1, cand2])])
                fe = torch.cat([fe[:-3], fe1, fe2])
            else:
                cands = torch.cat([cands[1:-2], torch.tensor([cand1, cand2])])
                fe = torch.cat([fe[1:-2], fe1, fe2])

        mesh.add_face(loop)
        new_faces.append(loop)

        faces_predicted.append(Mesh(opt, verts=mesh.vs, faces=new_faces))
        faces_real.append(Mesh(opt, faces=batch['face_meta'][i]))

        name = os.path.split(mesh.file)[1]
        os.makedirs(f'{export_dir}/meshes', exist_ok=True)
        mesh.export(file=f'{export_dir}/meshes/{name}')

    return batch['x'], faces_predicted, faces_real


def new_vs_features(meshes, n_vs: List[int]):
    mesh_features, face_candidates = [], []
    for i, mesh in enumerate(meshes):
        features, candidates = [], []
        loop = mesh.boundary_loop
        for j, vert_0 in enumerate(loop):
            vert_1 = loop[(j + 1) % len(loop)]
            fe, cands = [], []
            for k in range(n_vs[i]):
                new_cand = [vert_0, vert_1, len(mesh.vs) - k - 1]
                cands.append(new_cand)
                fe.append(get_badness(mesh.vs[new_cand]))
            candidates.append(cands)
            features.append(fe)
        face_candidates.append(torch.tensor(candidates, device=device))
        mesh_features.append(torch.tensor(features, device=device))
    return mesh_features, face_candidates


def get_badness(triangle: torch.Tensor):
    """
    Punishes a triangle for being different from an equilateral triangle.\n
    :param triangle: shape: (3,3)
    """
    angles = torch.tensor([inner_angle(triangle[(k + 1) % 3], triangle[k]) for k in range(3)])
    return torch.sum((angles - torch.pi / 3) ** 2)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', '.*__floordiv__ is deprecated.*')

    opt = Options().args

    torch.manual_seed(opt.seed)
    rng = np.random.default_rng(opt.seed)

    if opt.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{opt.gpu}')
    else:
        device = torch.device('cpu')

    train_data = MRepDataset(opt, True, boundary=True)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                              collate_fn=train_data.mesh_collate)
    n_vs_net = NVertsNet(opt, opt.verts_max, 200).to(device)
    vert_pos_net = VertPosNet(opt, opt.verts_max, 250)

    export_dir = os.path.abspath(opt.export_dir)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    run_id = 0
    while os.path.exists(os.path.join(export_dir, f'run_{run_id:03d}')):
        run_id += 1
    export_dir = os.path.join(export_dir, f'run_{run_id:03d}')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    n_vs_net.load_state_dict(torch.load('./saved_data/pillows/eval_hole/models/epoch_0011/n_vs_net.pt'))
    vert_pos_net.load_state_dict(torch.load('./saved_data/pillows/eval_hole/models/epoch_0051/v_pos_net.pt'))

    for i, batch in enumerate(train_loader):

        n_vs_pred, n_vs_real = predict_n_verts(batch)
        n_vs_pred = n_vs_pred.argmax(dim=1)

        vert_pos_pred, vert_pos_real = predict_vert_positions(batch, n_vs_pred)

        [batch['x'][idx].add_verts(pos) for idx, pos in enumerate(vert_pos_pred)]
        meshes_pred, faces_pred, faces_real = predict_faces(batch, n_vs_pred)
