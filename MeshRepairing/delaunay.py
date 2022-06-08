import numpy as np
from scipy.spatial import Delaunay

from models.layers.mesh import Mesh

if __name__ == '__main__':
    file = './saved_data/pillow.obj'

    vs = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            split_line = line.split()
            if split_line and split_line[0] == 'v':
                vs.append([float(v) for v in split_line[1:4]])

    vs = np.asarray(vs)

    delaunay_tesselation = Delaunay(vs)

    mesh = Mesh(None, verts=vs, faces=delaunay_tesselation.simplices)
    mesh.export(file='./saved_data/delaunay_pillow.obj')
