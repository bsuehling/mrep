import ntpath
import os
import pickle
from typing import List

import numpy as np
import torch

from linalg import inner_angle


class Mesh:
    def __init__(self, opt, verts=None, faces=None, file=None, device='cpu', export_folder='', boundary=False):
        self.__opt = opt
        self.device = device
        self.history_data = None
        if verts is not None and faces is not None:
            self.vs = np.asarray(verts)
            self.faces = np.asarray(faces, dtype=int)
            self.scale_and_translation()
        elif faces is not None:
            self.vs, self.faces = self.from_face_coords(faces)
            self.scale_and_translation()
        else:
            self.file = file
            self.vs = self.v_mask = self.filename = self.features = None
            self.edges = self.gemm_edges = self.sides = None
            self.pool_count = 0
            self.fill_mesh(boundary)
            self.export_folder = export_folder
            self.export()
        self.n_verts = len(self.vs)
        self.vs = torch.tensor(self.vs, device=self.device)
        self.faces = torch.tensor(self.faces, device=self.device, dtype=torch.long)

    @staticmethod
    def from_face_coords(faces: torch.tensor):
        """
        For faces represented as lists of vertex coordinates, creates a list of vertices and a list of faces represented
        as vertex indices.\n
        :param faces: the face coordinates (shape: (N,3))
        :return: vertex positions and faces with vertex indices
        """
        vs, fs = [], []
        for f in faces.tolist():
            new_f = []
            for v in f:
                try:
                    idx = vs.index(v)
                except ValueError:
                    idx = len(vs)
                    vs.append(v)
                new_f.append(idx)
            fs.append(new_f)
        return np.asarray(vs), np.asarray(fs)

    def fill_mesh(self, boundary: bool):
        load_path = self.get_mesh_path()
        if load_path is not None and os.path.exists(load_path):
            mesh_data = np.load(load_path, encoding='latin1', allow_pickle=True)
            self.vs = mesh_data['vs']
            self.faces = mesh_data['faces']
            self.edges = mesh_data['edges']
            self.edge_faces = mesh_data['edge_faces']
            self.edge2key = mesh_data['edge2key'].item()
            self.ve = mesh_data['ve'].tolist()
            self.gemm_edges = mesh_data['gemm_edges']
            self.edges_count = int(mesh_data['edges_count'])
            self.v_mask = mesh_data['v_mask']
            self.filename = str(mesh_data['filename'])
            self.features = mesh_data['features']
            self.sides = mesh_data['sides']
            self.edge_lengths = mesh_data['edge_lengths']
            self.scale = mesh_data['scale']
            self.translation = mesh_data['translation']
            self.boundary_loop = mesh_data['boundary_loop']
            self.boundary_vs = mesh_data['boundary_vs']
            if self.boundary_loop.shape != ():
                self.boundary_loop = self.boundary_loop.tolist()
                self.boundary_vs = torch.from_numpy(self.boundary_vs).to(self.device)
        else:
            self.from_scratch(boundary)
            ve = np.array(self.ve, dtype=object)
            np.savez_compressed(load_path, gemm_edges=self.gemm_edges, vs=self.vs, edges=self.edges, scale=self.scale,
                                edges_count=self.edges_count, v_mask=self.v_mask, filename=self.filename, ve=ve,
                                sides=self.sides, features=self.features, translation=self.translation,
                                faces=self.faces, edge_lengths=self.edge_lengths, edge_faces=self.edge_faces,
                                edge2key=self.edge2key, boundary_loop=self.boundary_loop, boundary_vs=self.boundary_vs)

    def get_mesh_path(self):
        filename, _ = os.path.splitext(self.file)
        dir_name = os.path.dirname(filename)
        prefix = os.path.basename(filename)
        load_dir = os.path.join(dir_name, 'cache')
        load_file = os.path.join(load_dir, f'{prefix}.npz')
        if not os.path.isdir(load_dir):
            os.makedirs(load_dir, exist_ok=True)
        return load_file

    def from_scratch(self, boundary: bool):
        self.fill_from_file()
        self.v_mask = np.ones(len(self.vs), dtype=bool)
        self.remove_non_manifolds()
        self.build_gemm()
        self.scale_and_translation()
        self.boundary_loop = self.boundary_vs = None
        if boundary:
            self.boundary_loop = self.find_boundary_loop()
            self.boundary_vs = torch.tensor([v in self.boundary_loop for v in range(len(self.vs))], device=self.device)
        features = []
        edge_points = self.get_edge_points()
        with np.errstate(divide='raise'):
            try:
                features.append(self.dihedral_angle(edge_points))
                features.append(self.symmetric_opposite_angles(edge_points))
                features.append(self.symmetric_ratios(edge_points))
                if boundary:
                    boundaries = [self.is_boundary_edge(i) for i in range(len(self.edges))]
                    features.append(np.array(boundaries, dtype=np.float64)[np.newaxis])
                    features.append(self.edge_lengths[np.newaxis])
                    edge_normals = []  # average of two adjacent faces' normals
                    for i in range(len(self.edges)):
                        f1 = torch.tensor(self.vs[self.edge_faces[i][0]])
                        f2 = torch.tensor(self.vs[self.edge_faces[i][1]])
                        n1 = torch.cross(f1[1] - f1[0], f1[2] - f1[0])
                        n2 = torch.cross(f2[1] - f2[0], f2[2] - f2[0])
                        edge_normals.append(((n1 + n2) / torch.norm(n1 + n2)).tolist())
                    features.append(np.array(edge_normals, dtype=np.float64).transpose())
                self.features = np.concatenate(features, axis=0)
            except Exception as e:
                print(e)
                raise ValueError(self.filename, 'bad features')

    def fill_from_file(self):
        self.filename = ntpath.split(self.file)[1]
        vs, faces = [], []
        with open(self.file) as f:
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
        self.vs = np.asarray(vs)
        self.faces = np.asarray(faces, dtype=int)
        assert np.logical_and(self.faces >= 0, self.faces < len(vs)).all()

    def remove_non_manifolds(self):
        """
        removes non_manifolds and at the same time calculates face normals and areas
        """
        edges_set = set()
        mask = np.ones(len(self.faces), dtype=bool)
        self.face_normals = np.cross(self.vs[self.faces[:, 1]] - self.vs[self.faces[:, 0]],
                                     self.vs[self.faces[:, 2]] - self.vs[self.faces[:, 1]])
        self.face_areas = np.sqrt((self.face_normals ** 2).sum(axis=1))
        for face_id, face in enumerate(self.faces):
            if self.face_areas[face_id] == 0:
                mask[face_id] = False
                continue
            face_edges = []
            is_manifold = False
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                if cur_edge in edges_set:
                    is_manifold = True
                    break
                else:
                    face_edges.append(cur_edge)
            if is_manifold:
                mask[face_id] = False
            else:
                for _, edge in enumerate(face_edges):
                    edges_set.add(edge)
        self.faces = self.faces[mask]
        self.face_areas = self.face_areas[mask]
        self.face_normals = self.face_normals[mask]
        self.face_normals /= self.face_areas[:, np.newaxis]
        self.face_areas *= 0.5

    def build_gemm(self):
        self.ve = [[] for _ in self.vs]
        self.vei = [[] for _ in self.vs]
        edge_nb = []
        sides = []
        edge2key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        edge_faces = []
        for face_id, face in enumerate(self.faces):
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edge_faces.append([face, [0, 0, 0]])
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    self.ve[edge[0]].append(edges_count)
                    self.ve[edge[1]].append(edges_count)
                    self.vei[edge[0]].append(0)
                    self.vei[edge[1]].append(1)
                    nb_count.append(0)
                    edges_count += 1
                else:
                    edge_faces[edge2key[edge]][1] = face
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
        self.edges = np.array(edges, dtype=np.int32)
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count
        self.edge2key = edge2key
        self.edge_faces = np.array(edge_faces)
        self.edge_lengths = np.linalg.norm(self.vs[self.edges[:, 0]] - self.vs[self.edges[:, 1]], axis=1)

    def scale_and_translation(self):
        """
        computes scale and translation of the mesh
        """
        self.scale = max([self.vs[:, i].max() - self.vs[:, i].min() for i in range(3)])
        scaled_vs = self.vs / self.scale
        target_mins = [(scaled_vs[:, i].max() - scaled_vs[:, i].min()) / 2. for i in range(3)]
        translation = np.array([(scaled_vs[:, i].min() - target_mins[i]) for i in range(3)])[None, :]
        self.translation = torch.from_numpy(translation).to(self.device)

    def get_edge_points(self):
        """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
            for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id
            each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2]
            or edge_points[edge_id, 3]
        """
        edge_points = np.zeros((self.edges_count, 4), dtype=np.int32)
        for edge_id, edge in enumerate(self.edges):
            edge_points[edge_id] = self.get_side_points(edge_id)
        return edge_points

    def get_side_points(self, edge_id):
        gemm_edges = self.gemm_edges[edge_id]
        edge_a = self.edges[edge_id]
        if gemm_edges[0] == -1:
            edge_b = self.edges[gemm_edges[2]]
            edge_c = self.edges[gemm_edges[3]]
        else:
            edge_b = self.edges[gemm_edges[0]]
            edge_c = self.edges[gemm_edges[1]]
        if gemm_edges[2] == -1:
            edge_d = self.edges[gemm_edges[0]]
            edge_e = self.edges[gemm_edges[1]]
        else:
            edge_d = self.edges[gemm_edges[2]]
            edge_e = self.edges[gemm_edges[3]]
        first_vertex = 0
        second_vertex = 0
        third_vertex = 0
        if edge_a[1] in edge_b:
            first_vertex = 1
        if edge_b[1] in edge_c:
            second_vertex = 1
        if edge_d[1] in edge_e:
            third_vertex = 1
        return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]

    def get_normals(self, edge_points, side):
        edge_a = self.vs[edge_points[:, side // 2 + 2]] - self.vs[edge_points[:, side // 2]]
        edge_b = self.vs[edge_points[:, 1 - side // 2]] - self.vs[edge_points[:, side // 2]]
        normals = np.cross(edge_a, edge_b)
        div = self.fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
        normals /= div[:, np.newaxis]
        return normals

    def get_opposite_angles(self, edge_points, side):
        edges_a = self.vs[edge_points[:, side // 2]] - self.vs[edge_points[:, side // 2 + 2]]
        edges_b = self.vs[edge_points[:, 1 - side // 2]] - self.vs[edge_points[:, side // 2 + 2]]

        edges_a /= self.fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
        edges_b /= self.fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
        dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
        return np.arccos(dot)

    def get_ratios(self, edge_points, side):
        point_o = self.vs[edge_points[:, side // 2 + 2]]
        point_a = self.vs[edge_points[:, side // 2]]
        point_b = self.vs[edge_points[:, 1 - side // 2]]
        line_ab = point_b - point_a
        projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / self.fixed_division(
            np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
        closest_point = point_a + (projection_length / self.edge_lengths)[:, np.newaxis] * line_ab
        d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
        return d / self.edge_lengths

    @staticmethod
    def fixed_division(to_div, epsilon):
        if epsilon == 0:
            to_div[to_div == 0] = 0.1
        else:
            to_div += epsilon
        return to_div

    def dihedral_angle(self, edge_points):
        normals_a = self.get_normals(edge_points, 0)
        normals_b = self.get_normals(edge_points, 3)
        dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
        angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
        return angles

    def symmetric_opposite_angles(self, edge_points):
        """ computes two angles: one for each face shared between the edge
            the angle is in each face opposite the edge
            sort handles order ambiguity
        """
        angles_a = self.get_opposite_angles(edge_points, 0)
        angles_b = self.get_opposite_angles(edge_points, 3)
        angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
        angles = np.sort(angles, axis=0)
        return angles

    def symmetric_ratios(self, edge_points):
        """ computes two ratios: one for each face shared between the edge
            the ratio is between the height / base (edge) of each triangle
            sort handles order ambiguity
        """
        ratios_a = self.get_ratios(edge_points, 0)
        ratios_b = self.get_ratios(edge_points, 3)
        ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
        return np.sort(ratios, axis=0)

    def find_boundary_loop(self):
        """
        For one hole in the mesh, finds the loop of boundary edges\n
        :return: the boundary edge loop
        """
        for idx, edge in enumerate(self.edges):
            if self.is_boundary_edge(idx):
                return self._boundary_loop(idx)
        return []

    def _boundary_loop(self, edge_id: int):
        """
        Starting from a boundary edge, searches for a loop of boundary edges\n
        :param edge_id: index of the boundary edge to start from
        :return: the boundary edge loop
        """

        def recursive(loop: List[int], vert: int, edge: int, stop_vert: int = None):
            """
            Recursively finds a boundary loop.\n
            :param loop: current intermediate result
            :param vert: next vert to add to loop
            :param edge: idx of last edge added to loop
            :param stop_vert: stop recursion when reaching this vert
            :return: a boundary loop
            """
            if vert == loop[0]:
                return loop
            if vert == stop_vert:
                return loop + [stop_vert]
            loop.append(vert)
            candidates = []
            for e in self.ve[vert]:
                if self.is_boundary_edge(e) and e != edge:
                    candidates.append(e)
            verts1, verts2 = [self.edges[e, 0] for e in candidates], [self.edges[e, 1] for e in candidates]
            new_vs = [verts1[i] if verts2[i] == loop[-1] else verts2[i] for i in range(len(candidates))]
            if len(candidates) == 1:
                return recursive(loop, new_vs[0], candidates[0], stop_vert=stop_vert)
            assert len(candidates) == 3, f'{self.filename}: not exactly 3 candidates, but {len(candidates)}'
            subs = [recursive([vert], new_vs[i], cand, stop_vert=loop[0]) for i, cand in enumerate(candidates)]
            sub1, sub2 = remove_duplicate(loop[-2], subs)
            return loop + sub1[1:] + sub2[:-1]

        def remove_duplicate(vert, loops):
            """
            Of the two reversed loops, remove the one that spans a higher angle with vert
            :param vert: the vertex to measure the angle with
            :param loops: the three loops of which one is to be removed
            :return: the two remaining loops
            """
            vect1 = [self.vs[lo[0]] - self.vs[vert] for lo in loops]
            vect2 = [self.vs[lo[1]] - self.vs[lo[0]] for lo in loops]
            angles = [inner_angle(vect1[i], vect2[i]) for i in range(3)]
            if loops[0][1:] == loops[1][:0:-1]:
                if angles[0] < angles[1]:
                    return loops[0], loops[2]
                return loops[1], loops[2]
            if loops[0][1:] == loops[2][:0:-1]:
                if angles[0] < angles[2]:
                    return loops[0], loops[1]
                return loops[2], loops[1]
            if angles[1] < angles[2]:
                return loops[1], loops[0]
            return loops[2], loops[0]

        return recursive([self.edges[edge_id, 0]], self.edges[edge_id, 1], edge_id)

    def is_boundary_edge(self, edge_id: int):
        """
        For a given edge, tests whether it is a boundary edge or not\n
        :param edge_id: index of the edge
        :return: True for a boundary edge, False otherwise
        """
        return -1 in self.gemm_edges[edge_id]

    def update_verts(self, verts):
        """
        update vertex positions only, same connectivity\n
        :param verts: new verts
        """
        self.vs = verts

    def add_verts(self, verts):
        """
        Adds new vertices to the mesh and updates necessary data structures.\n
        :param verts: the vertex to add
        """
        torch.cat([self.vs, verts])
        [self.ve.append([]) for _ in verts]

    def add_edge(self, edge):
        """
        Adds a new edge to the mesh and updates necessary data structures.\n
        :param edge: the edge to add
        """
        edge_tuple = tuple(sorted(edge))
        np.append(self.edges, [sorted(edge)], axis=0)
        self.edge2key[edge_tuple] = self.edges_count
        self.ve[edge[0]].append(self.edges_count)
        self.ve[edge[1]].append(self.edges_count)
        self.edge_faces = np.append(self.edge_faces, [2 * [[0, 0, 0]]], axis=0)
        self.edges_count += 1

    def add_face(self, face):
        """
        Adds a new face to the mesh and updates necessary data structures.\n
        :param face: the face to add
        """
        face = self.correct_order(face)
        self.faces = torch.cat([self.faces, torch.tensor(face).unsqueeze(0)])
        for i in range(3):
            edge_id = self.edge2key[tuple(sorted([face[i], face[(i + 1) % 3]]))]
            if (self.edge_faces[edge_id, 0] == [0, 0, 0]).all():
                self.edge_faces[edge_id, 0] = face
            else:
                self.edge_faces[edge_id, 1] = face

    def correct_order(self, face):
        """
        ensures face normal consistency
        :param face: the input face (shape: (3,))
        :return: the face with correct vertex order (shape: (3,))
        """
        edge = np.array(face[:2])
        other_face = self.edge_faces[self.edge2key[tuple(sorted(edge))], 0]
        for i in range(3):
            if (edge == other_face[[i, (i + 1) % 3]]).all():
                return [face[1], face[0], face[2]]
        return face

    def deep_copy(self):
        new_mesh = Mesh(self.__opt, verts=self.vs, faces=self.faces)
        for attr in dir(self):
            if attr == '__dict__':
                continue
            val = getattr(self, attr)
            if type(val) == np.ndarray:
                setattr(new_mesh, attr, val.copy())
            elif type(val) == torch.Tensor:
                setattr(new_mesh, attr, val.clone())
            elif type(val) in [dict, list]:
                setattr(new_mesh, attr, pickle.loads(pickle.dumps(val, -1)))
            elif type(val) in [str, int, bool, float]:
                setattr(new_mesh, attr, val)
        return new_mesh

    def export(self, file: str = None):
        if file is None:
            if self.export_folder:
                filename, file_extension = os.path.splitext(self.filename)
                file = f'{self.export_folder}/{filename}_{self.pool_count}{file_extension}'
            else:
                return
        if file.endswith('.obj'):
            with open(file, 'w+') as f:
                for v in self.vs:
                    f.write(f'v {v[0]} {v[1]} {v[2]}\n')
                for face in self.faces:
                    f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')
        elif file.endswith('.off'):
            with open(file, 'w+') as f:
                f.write('OFF\n')
                f.write(f'{len(self.vs)} {len(self.faces)} 0\n')
                for v in self.vs:
                    f.write(f'{v[0]} {v[1]} {v[2]}\n')
                for face in self.faces:
                    f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
        else:
            raise ValueError('Invalid file type. Valid types are .obj and .off')

    # methods for pooling:

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.v_mask[edge[1]] = False
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]
        for v in vs:
            if edge_id not in self.ve[v]:
                print(self.ve[v])
                print(self.filename)
            self.ve[v].remove(edge_id)

    def clean(self, edges_mask):
        edges_mask = edges_mask.astype(bool)
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve
