import argparse
import os
import random
import sys
from typing import Iterable

import bmesh
import bpy
import mathutils
import numpy as np


class Options:
    def __init__(self):
        self.args = None
        self.__parse_args()

    def __parse_args(self):
        p = argparse.ArgumentParser(description='flaw generator options')
        aa = p.add_argument
        aa('--displace', default=1., type=float, help='displace factor for vertices')
        aa('--export-dir', default='./datasets', help='where to save models')
        aa('--face-prop', action='store_true', help='probability of a face to be split is proportional to its size')
        aa('--faces', nargs=2, default=[.0, .1], type=int, help='which fraction of the original faces to delete')
        aa('--faces-abs', nargs=2, type=int, help='how many faces to delete at least and at most. Overrides --faces')
        aa('--fixed-hole-pos', action='store_true',
           help='always create large hole starting with first face in .obj file')
        aa('--gap-length', nargs=2, default=[5, 30], type=int, help='desired minimum and maximum gap length')
        aa('--gaps', nargs=2, default=[0, 2], type=int, help='how many gaps to produce at least and at most')
        aa('--hole', default=5, type=float, help='expected value for the number of faces to delete for the large hole.'
                                                 '0 means no large whole will be created')
        aa('--hole-fixed', type=int, help='exact number of faces to delete for large hole. Overwrites --hole')
        aa('--hole-norm', default=0, type=float,
           help='if larger than 0, the size of large holes will be normally distributed using this value as std.'
                'otherwise, a geometric distribution will be used.')
        aa('--meta', action='store_true', help='whether to save meta information about deleted faces')
        aa('--triangulate', action='store_false', help='triangulates input meshes before creating flaws')
        aa('--root', type=str, required=True, help='path to original meshes')
        aa('--seed', type=int, help='which random seed to use. default is None')
        aa('--splits', default=100, type=int, help='how many face splits to perform')
        aa('--test-samples', default=2, type=int, help='how many faulty meshes for testing to create per input mesh')
        aa('--train-samples', default=10, type=int, help='how many faulty meshes for training to create per input mesh')
        try:
            arg_list = sys.argv[sys.argv.index('--') + 1:]
        except ValueError:
            arg_list = []
        self.args = p.parse_args(args=arg_list)
        print('Arguments passed to program:')
        options = vars(self.args)
        if options['seed'] is None:
            options['seed'] = random.randint(0, 100000)
        for option in options:
            print(f"{option + ':':20} {options[option]}")
        print(100 * '-')


def create_export_dirs(root: str) -> (str, str, str):
    ex_dir = os.path.join(root, 'dataset_{:03d}')
    dir_id = 0
    while os.path.isdir(ex_dir.format(dir_id)):
        dir_id += 1
    ex_dir = ex_dir.format(dir_id)
    tr_dir = os.path.join(ex_dir, 'train')
    te_dir = os.path.join(ex_dir, 'test')
    os.mkdir(ex_dir)
    os.mkdir(tr_dir)
    if opt.test_samples > 0:
        os.mkdir(te_dir)
    return ex_dir, tr_dir, te_dir


def import_bm(filepath: str) -> bmesh.types.BMesh:
    bpy.ops.import_scene.obj(filepath=filepath)
    o = bpy.context.selected_objects[0]
    bm = bmesh.new()
    bm.from_mesh(o.data)
    return bm


def export_bm(filepath: str, bm: bmesh.types.BMesh):
    if not filepath.endswith('.obj'):
        filepath += '.obj'
    o = bpy.context.selected_objects[0]
    bm.to_mesh(o.data)
    bpy.ops.export_scene.obj(filepath=filepath, use_uvs=False, use_materials=False)


def save_meta(filepath: str, before: bmesh.types.BMesh, after: bmesh.types.BMesh, faces_deleted: np.ndarray):
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    verts_before = [v.co[:] for v in before.verts]
    verts_after = [v.co[:] for v in after.verts]
    verts_deleted = np.array([list(x) for x in (set(verts_before) - set(verts_after))])
    np.savez_compressed(filepath, faces=faces_deleted, verts=verts_deleted)


def split_faces(bm: bmesh.types.BMesh):
    def calc_p_draw(faces):
        areas = []
        for _, face in enumerate(faces):
            areas.append(face.calc_area())
        areas = np.array(areas)
        total_area = areas.sum()
        return areas / total_area

    n_splits = opt.splits
    while n_splits > 0:
        p_draw = calc_p_draw(bm.faces) if opt.face_prop else None
        if n_splits >= 10:
            faces_to_split = rng.choice(bm.faces, size=10, p=p_draw, replace=False)
            n_splits -= 10
        else:
            faces_to_split = rng.choice(bm.faces, size=n_splits, p=p_draw, replace=False)
            n_splits = 0
        for cur_face in faces_to_split:
            f_verts = [v for v in cur_face.verts]
            center = cur_face.calc_center_median()
            bm.faces.remove(cur_face)
            new_v = bm.verts.new(center)
            for v in f_verts:
                bm.edges.new([v, new_v])
            for i in range(3):
                bm.faces.new([new_v, f_verts[i], f_verts[(i + 1) % 3]])


def init_gaps(bm: bmesh.types.BMesh, max_lengths: Iterable[int] = None):
    if not max_lengths:
        return
    collected = set()
    for max_l in max_lengths:
        cur_edge = rng.choice(bm.edges)
        while cur_edge.calc_length() == 0:
            cur_edge = rng.choice(bm.edges)
        cur_vert = rng.choice(cur_edge.verts)
        collected.add(cur_edge)
        for i in range(max_l):
            to_check = set(cur_vert.link_edges) - collected
            if to_check:
                to_add = to_check.pop()
            else:
                break
            cur_edge_vec = to_vect(cur_edge)
            while to_check:
                tmp = to_check.pop()
                if tmp.calc_length() == 0:
                    continue
                if cur_edge_vec.angle(to_vect(tmp)) < cur_edge_vec.angle(to_vect(to_add)):
                    to_add = tmp
            collected.add(to_add)
            cur_edge = to_add
            cur_vert = to_add.other_vert(cur_vert)
    bmesh.ops.split_edges(bm, edges=list(collected))


def delete_faces(bm: bmesh.types.BMesh, n_min: int = 0, n_max: int = -1):
    def delete_face():
        face_choices = rng.choice(list(large_whole), size=len(large_whole))
        neighbors = set()
        for face_choice in face_choices:
            for _, e in enumerate(face_choice.edges):
                neighbors.update(e.link_faces)
            candidates = neighbors - large_whole
            if candidates:
                large_whole.add(rng.choice(list(candidates)))
                break

    if n_max < 0 or n_max > len(bm.faces):
        n_max = len(bm.faces)
    n_to_delete = rng.integers(n_min, n_max + 1)
    if opt.fixed_hole_pos:
        faces_to_delete = set(bm.faces[:n_to_delete + 1])
    else:
        faces_to_delete = set(rng.choice(bm.faces, size=n_to_delete, replace=False))
    if not faces_to_delete:
        return
    large_whole = {faces_to_delete.pop()}
    p_continue = 1 - 1 / opt.hole if opt.hole > 0 else 0
    if opt.hole_fixed:  # fixed hole size
        for _ in range(opt.hole_fixed - 1):
            delete_face()
    elif opt.hole_norm:  # normal distribution
        for _ in range(int(np.rint(rng.normal(opt.hole, opt.hole_norm))) - 1):
            delete_face()
    else:
        while rng.random() < p_continue:  # geometric distribution
            delete_face()
    faces_to_delete |= large_whole
    faces_deleted = np.array([[[v.co[i] for i in range(3)] for v in face.verts] for face in faces_to_delete])
    bmesh.ops.delete(bm, geom=list(faces_to_delete), context='FACES')
    return faces_deleted


def displace_vertices(bm: bmesh.types.BMesh, factor: float = 1.):
    new_coordinates = []
    bm.verts.index_update()
    for _, v in enumerate(bm.verts):
        neighbors = []
        for _, e in enumerate(v.link_edges):
            neighbors.append(e.other_vert(v))
        x_difs, y_difs, z_difs = [], [], []
        for nei in neighbors:
            dif = v.co - nei.co
            x_difs.append(abs(dif[0]))
            y_difs.append(abs(dif[1]))
            z_difs.append(abs(dif[2]))
        dx_mean = np.mean(x_difs)
        dy_mean = np.mean(y_difs)
        dz_mean = np.mean(z_difs)
        d_max = max(dx_mean, dy_mean, dz_mean)

        def calculate_displace(d_mean):
            displace = rng.normal(scale=max(.004 * factor * d_mean, .001 * factor * d_max))
            displace = np.sign(displace) * np.sqrt(np.abs(displace))
            return displace

        x_displace = calculate_displace(dx_mean)
        y_displace = calculate_displace(dy_mean)
        z_displace = calculate_displace(dz_mean)
        vec_displace = mathutils.Vector((x_displace, y_displace, z_displace))
        new_coordinates.append(v.co + vec_displace)
    for ind, v in enumerate(bm.verts):
        v.co = new_coordinates[ind]


def to_vect(e: bmesh.types.BMEdge) -> mathutils.Vector:
    return e.verts[1].co - e.verts[0].co


if __name__ == '__main__':

    opt = Options().args

    rng = np.random.default_rng(opt.seed)

    source_dir = os.path.abspath(opt.root)
    export_dir = os.path.abspath(opt.export_dir)
    exp_dir, train_dir, test_dir = create_export_dirs(export_dir)

    # at the beginning, delete all existing objects:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for idx, element in enumerate(os.scandir(source_dir)):

        if element.is_dir() or not element.name.endswith('.obj'):
            continue

        path_to_file = os.path.join(source_dir, element.name)

        mesh = import_bm(path_to_file)
        if opt.triangulate:
            bmesh.ops.triangulate(mesh, faces=mesh.faces)
        export_bm(os.path.join(exp_dir, element.name), mesh)
        elem_name = os.path.splitext(element.name)[0]
        train_path = os.path.join(train_dir, elem_name)
        test_path = os.path.join(test_dir, elem_name)
        os.mkdir(train_path)
        if opt.test_samples > 0:
            os.mkdir(test_path)

        for i in range(opt.train_samples + opt.test_samples):
            me = mesh.copy()

            split_faces(me)

            n_gaps = rng.integers(opt.gaps[0], opt.gaps[1] + 1)
            gap_lengths = rng.integers(opt.gap_length[0], opt.gap_length[1] + 1, size=n_gaps)
            init_gaps(me, max_lengths=gap_lengths)

            displace_factor = opt.displace * (1 - np.square(rng.random()))
            displace_vertices(me, factor=displace_factor)

            before = me.copy()

            if opt.faces_abs is not None:
                n_min, n_max = opt.faces_abs[0], opt.faces_abs[1]
            else:
                n_min, n_max = int(opt.faces[0] * len(me.faces)), int(opt.faces[1] * len(me.faces))
            faces_deleted = delete_faces(me, n_min=n_min, n_max=n_max)

            obj_name = elem_name + f'_{i:03d}'
            if i < opt.train_samples:
                export_location = os.path.join(train_path, obj_name)
            else:
                export_location = os.path.join(test_path, obj_name)
            export_bm(export_location, me)

            if opt.meta:
                save_meta(export_location, before, me, faces_deleted)

            me.free()

        bpy.ops.object.delete(use_global=False, confirm=False)
