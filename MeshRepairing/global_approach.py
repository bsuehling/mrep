import os.path
import time
import warnings

import numpy as np
import numpy.random
import torch
from torch.utils.data import DataLoader
from trimesh.primitives import Sphere

from data import MRepDataset
from graphics import save_loss_plot
from losses import GlobalCriterion
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
        os.makedirs(f'{export_dir}/meshes/epoch_{epoch:04d}', exist_ok=True)
        inter.export(file=f'{export_dir}/meshes/epoch_{epoch:04d}/object_{name[6:10]}.obj')

    return criterion(intermediates, batch['y'])


def save_model(epoch: int):
    os.makedirs(f'{export_dir}/models/epoch_{epoch:04d}', exist_ok=True)
    torch.save(net.state_dict(), f'{export_dir}/models/epoch_{epoch:04d}/net.pt')
    torch.save(optimizer.state_dict(), f'{export_dir}/models/epoch_{epoch:04d}/optimizer.pt')
    torch.save(scheduler.state_dict(), f'{export_dir}/models/epoch_{epoch:04d}/scheduler.pt')
    torch.save(net.state_dict(), f'{latest_dir}/net.pt')
    torch.save(optimizer.state_dict(), f'{latest_dir}/optimizer.pt')
    torch.save(scheduler.state_dict(), f'{latest_dir}/scheduler.pt')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', '.*__floordiv__ is deprecated.*')

    opt = Options().args

    torch.manual_seed(opt.seed)
    rng = np.random.default_rng(opt.seed)

    if opt.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{opt.gpu}')
    else:
        device = torch.device('cpu')

    train_data, test_data = MRepDataset(opt, True), MRepDataset(opt, False)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                              collate_fn=train_data.mesh_collate, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, collate_fn=test_data.mesh_collate)
    criterion = GlobalCriterion(opt)
    net = GlobalNet(opt, init_weights_size=opt.init_weights).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr_glob)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30)

    export_dir = os.path.abspath(opt.export_dir)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    latest_dir = os.path.join(export_dir, 'latest')
    if not os.path.exists(latest_dir):
        os.makedirs(latest_dir, exist_ok=True)
    run_id = 0
    while os.path.exists(os.path.join(export_dir, f'run_{run_id:03d}')):
        run_id += 1
    export_dir = os.path.join(export_dir, f'run_{run_id:03d}')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    with open(os.path.join(export_dir, 'options.txt'), 'w') as f:
        options = vars(opt)
        for option in options:
            f.write(f"{option + ':':20} {options[option]}\n")

    if opt.continue_train:
        net.load_state_dict(torch.load(f'{latest_dir}/net.pt'))
        optimizer.load_state_dict(torch.load(f'{latest_dir}/optimizer.pt'))
        scheduler.load_state_dict(torch.load(f'{latest_dir}/scheduler.pt'))

    start_time = time.time()

    for epoch in range(opt.start_epoch + 1, opt.epochs + 1):
        print(100 * '-')

        epoch_start_time = time.time()
        net.train()
        epoch_loss = 0
        loss_meta = dict()

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss, meta = repair_meshes(batch)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

            for key in meta:
                if key not in loss_meta:
                    loss_meta[key] = 0
                loss_meta[key] += meta[key]

        scheduler.step(epoch_loss)

        if opt.test_each_epoch:
            net.eval()
            test_loss = 0
            for i, data in enumerate(test_loader):
                test_loss += repair_meshes(data)[0]

            with open(f'{export_dir}/test_log.txt', 'a') as f:
                f.write(f'{test_loss * opt.batch_size / len(test_loader)}\n')

        save_model(epoch)

        time_taken = int(time.time() - epoch_start_time)
        print(f'end of epoch {epoch} of {opt.epochs}, loss: {epoch_loss}, time taken: {time_taken} seconds')
        loss_info = ''
        for key in loss_meta:
            if loss_info:
                loss_info += ', '
            loss_info += f'{key}: {loss_meta[key]}'
        print(loss_info)
        print(f"learning rate: {optimizer.param_groups[0]['lr']}")

        with open(f'{export_dir}/loss_log.txt', 'a') as f:
            f.write(f'{epoch_loss * opt.batch_size / len(train_loader)}\n')
            for key in loss_meta:
                f.write(f' {loss_meta[key]}')
            f.write('\n')
    save_loss_plot(opt.epochs - opt.start_epoch, 'loss_plot', os.path.join(export_dir, 'loss_log.txt'),
                   test_acc_file=os.path.join(export_dir, 'test_log.txt'), log_scale=True)
