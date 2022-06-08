import os.path
import warnings

import numpy as np
import numpy.random
import torch
from torch.utils.data import DataLoader
from trimesh.primitives import Sphere

from data import MRepDataset
from models.layers.mesh import Mesh
from models.networks import GlobalNet
from options import Options


def repair_meshes(batch):
    hulls = [Sphere() for _ in batch['x']]
    intermediates = np.array([Mesh(opt, verts=h.vertices, faces=h.faces) for h in hulls])
    for idx, inter in enumerate(intermediates):
        inter.vs /= inter.scale
        inter.vs -= inter.translation

    features = batch['x_features'].to(device).float()
    for idx, new_vs in enumerate(net(features, batch['x'], intermediates)):
        inter = intermediates[idx]
        inter.update_verts(new_vs)
        inter.vs = inter.vs + inter.translation
        inter.vs *= inter.scale
        name = os.path.split(batch['x'][idx].file)[1]
        os.makedirs(f'{export_dir}/meshes', exist_ok=True)
        inter.export(file=f'{export_dir}/meshes/{name}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', '.*__floordiv__ is deprecated.*')

    opt = Options().args

    torch.manual_seed(opt.seed)
    rng = np.random.default_rng(opt.seed)

    if opt.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{opt.gpu}')
    else:
        device = torch.device('cpu')

    train_data = MRepDataset(opt, True)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                              collate_fn=train_data.mesh_collate)
    net = GlobalNet(opt, init_weights_size=opt.init_weights).to(device)

    export_dir = os.path.abspath(opt.export_dir)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    run_id = 0
    while os.path.exists(os.path.join(export_dir, f'run_{run_id:03d}')):
        run_id += 1
    export_dir = os.path.join(export_dir, f'run_{run_id:03d}')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    net.load_state_dict(torch.load('./saved_data/pillows/eval_global/models/epoch_0024/net.pt'))

    net.eval()

    for i, batch in enumerate(train_loader):
        repair_meshes(batch)
