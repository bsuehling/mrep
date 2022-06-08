import os

import torch
from losses import ChamferDistance, Laplacian
from models.layers.mesh import Mesh


def import_mesh(mesh_file):
    vs, faces = [], []
    with open(mesh_file) as f:
        for line in f:
            line = line.strip()
            split_line = line.split()
            if not split_line:
                continue
            elif split_line[0] == 'v':
                vs.append([float(v) for v in split_line[1:4]])
            elif split_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in split_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [ind - 1 if ind >= 0 else len(vs) + ind for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
    return vs, faces


if __name__ == '__main__':
    base_path = './saved_data/evaluation'
    samples = 10000
    chamfer = ChamferDistance(None, samples=samples)
    laplacian = Laplacian()

    ##### knifes: #####
    y_vs, y_faces = import_mesh(f'{base_path}/knife/knife.obj')
    y_mesh = Mesh(None, verts=y_vs, faces=y_faces)
    # global:
    global_path = f'{base_path}/knife_global'
    for file in os.listdir(global_path):
        filepath = os.path.join(global_path, file)
        x_vs, x_faces = import_mesh(filepath)
        x_mesh = Mesh(None, x_vs, x_faces)
        chamfer_distance = chamfer([x_mesh], [y_mesh])
        laplacian_loss = laplacian([x_mesh])
        with open(filepath[:-4] + '_chamfer.txt', 'w+') as f:
            f.write(str(chamfer_distance.item()))
        with open(filepath[:-4] + '_laplacian.txt', 'w+') as f:
            f.write(str(laplacian_loss.item()))
    # hole_fill:
    hole_path = f'{base_path}/knife_holes'
    for file in os.listdir(hole_path):
        filepath = os.path.join(hole_path, file)
        x_vs, x_faces = import_mesh(filepath)
        x_mesh = Mesh(None, x_vs, x_faces)
        chamfer_distance = chamfer([x_mesh], [y_mesh])
        laplacian_loss = laplacian([x_mesh])
        with open(filepath[:-4] + '_chamfer.txt', 'w+') as f:
            f.write(str(chamfer_distance.item()))
        with open(filepath[:-4] + '_laplacian.txt', 'w+') as f:
            f.write(str(laplacian_loss.item()))
    # manifold:
    manifold_path = f'{base_path}/knife_manifold'
    for file in os.listdir(manifold_path):
        filepath = os.path.join(manifold_path, file)
        x_vs, x_faces = import_mesh(filepath)
        x_mesh = Mesh(None, x_vs, x_faces)
        chamfer_distance = chamfer([x_mesh], [y_mesh])
        laplacian_loss = laplacian([x_mesh])
        with open(filepath[:-4] + '_chamfer.txt', 'w+') as f:
            f.write(str(chamfer_distance.item()))
        with open(filepath[:-4] + '_laplacian.txt', 'w+') as f:
            f.write(str(laplacian_loss.item()))

    ##### pillows: #####
    y_meshes = []
    for i in range(4):
        vs, faces = import_mesh(f'{base_path}/pillows/pillow_{i}.obj')
        y_meshes.append(Mesh(None, verts=vs, faces=faces))
    # global:
    for i in range(4):
        for j in range(3):
            file = f'{base_path}/pillows_global/pillow_{i}_00{j}.obj'
            x_vs, x_faces = import_mesh(file)
            x_mesh = Mesh(None, x_vs, x_faces)
            chamfer_distance = chamfer([x_mesh], [y_meshes[i]])
            laplacian_loss = laplacian([x_mesh])
            with open(file[:-4] + '_chamfer.txt', 'w+') as f:
                f.write(str(chamfer_distance.item()))
            with open(file[:-4] + '_laplacian.txt', 'w+') as f:
                f.write(str(laplacian_loss.item()))
    # hole_fill:
    for i in range(4):
        for j in range(3):
            file = f'{base_path}/pillows_holes/pillow_{i}_00{j}.obj'
            x_vs, x_faces = import_mesh(file)
            x_mesh = Mesh(None, x_vs, x_faces)
            chamfer_distance = chamfer([x_mesh], [y_meshes[i]])
            laplacian_loss = laplacian([x_mesh])
            with open(file[:-4] + '_chamfer.txt', 'w+') as f:
                f.write(str(chamfer_distance.item()))
            with open(file[:-4] + '_laplacian.txt', 'w+') as f:
                f.write(str(laplacian_loss.item()))
    # manifold:
    for i in range(4):
        for j in range(3):
            file = f'{base_path}/pillows_manifold/pillow_{i}_00{j}.obj'
            x_vs, x_faces = import_mesh(file)
            x_mesh = Mesh(None, x_vs, x_faces)
            chamfer_distance = chamfer([x_mesh], [y_meshes[i]])
            laplacian_loss = laplacian([x_mesh])
            with open(file[:-4] + '_chamfer.txt', 'w+') as f:
                f.write(str(chamfer_distance.item()))
            with open(file[:-4] + '_laplacian.txt', 'w+') as f:
                f.write(str(laplacian_loss.item()))
