from multiprocessing import Pool
from typing import List, Optional

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from trimesh.primitives import Sphere

from models.layers.mesh import Mesh


class ChamferDistance:
    def __init__(self, opt, from_mesh: bool = True, samples=0):
        if opt is not None:
            self.__num_samples = opt.num_samples
        else:
            self.__num_samples = samples
        self.__from_mesh = from_mesh

    def __call__(self, x_batch, y_batch):
        return self.compute_loss(x_batch, y_batch)

    def compute_loss(self, x_batch, y_batch):
        if self.__from_mesh:
            x_meshes = Meshes(verts=[x.vs.float() for x in x_batch], faces=[x.faces for x in x_batch])
            y_meshes = Meshes(verts=[y.vs.float() for y in y_batch], faces=[y.faces for y in y_batch])
            x_batch = sample_points_from_meshes(x_meshes, num_samples=self.__num_samples)
            y_batch = sample_points_from_meshes(y_meshes, num_samples=self.__num_samples)
            try:
                x_batch.requires_grad = True
            except RuntimeError:
                pass
            try:
                y_batch.requires_grad = True
            except RuntimeError:
                pass
        if type(x_batch) is list:
            loss = torch.tensor(0., requires_grad=True, device=x_batch[0].device)
            for i, x in enumerate(x_batch):
                if 0 not in x.shape and 0 not in y_batch[i].shape:
                    loss = loss + chamfer_distance(x[None].float(), y_batch[i][None].float())[0] / len(x)
            loss = loss / len(x_batch)
            return loss
        return chamfer_distance(x_batch, y_batch)[0]


class Laplacian:
    def __init__(self):
        pass

    def __call__(self, meshes):
        return self.compute_loss(meshes)

    @staticmethod
    def compute_loss(meshes):
        meshes = Meshes(verts=[mesh.vs.float() for mesh in meshes], faces=[mesh.faces for mesh in meshes])
        return mesh_laplacian_smoothing(meshes)


class EdgeLengthLoss:
    def __init__(self):
        sphere = Sphere()
        face, verts = sphere.faces[0], sphere.vertices
        self.__target_length = np.linalg.norm(verts[face[1]] - verts[face[0]])

    def __call__(self, x_batch):
        return self.compute_loss(x_batch)

    def compute_loss(self, x_batch):
        return mesh_edge_loss(x_batch, self.__target_length)


class SelfIntersectionPenalty:
    def __init__(self, opt):
        self.__opt = opt
        self.__rng = np.random.default_rng(opt.seed)

    def __call__(self, batch: List[Mesh], p: float = 1):
        return self.compute_loss(batch)

    @staticmethod
    def compute_loss(batch):
        """
        For given meshes, calculates the number of self-intersections.
        Returns the mean of squares of self-intersections per face
        Checks whether two triangles intersect using the method proposed in:\n
        [1] S. H. Lo: Automatic mesh generation over intersecting surfaces, 1995.\n
        :param batch: the batch of meshes
        :return: loss
        """
        loss = torch.tensor(0., device=batch[0].device, requires_grad=True)
        for mesh in batch:
            loss = loss + SelfIntersectionPenalty._compute_loss(mesh.vs[mesh.faces])
        return loss / len(batch)

    @staticmethod
    def _compute_loss(faces: torch.Tensor):
        """
        Computes number of self-intersections, divided by number of faces for normalization, for one mesh\n
        :param faces: the faces, which are assumed to be triangles, for which to find self-intersections (N x 3 x 3)
        :return: number of self-intersections
        """
        n = len(faces)
        ft = faces.transpose(0, 1)  # 3 x N x 3
        # compute face normals:
        normals = (ft[1] - ft[0]).cross(ft[2] - ft[0])  # N x 3
        # for each face, compute on which side of the face the points lie:
        sides = torch.einsum('ijkl,jl->ijk', ft[:, None] - ft[0, None, :, None], normals)  # 3 x N x N
        # for each face, check for all faces whether 2 points of it lie on different sides:
        opposite_sides = sides * sides[[1, 2, 0]] < 0  # 3 x N x N
        # help variable:
        t = sides / (sides[[1, 2, 0]] - sides)  # 3 x N x N
        # intersection points:
        p = t[..., None] * ft[[1, 2, 0], None] + (1 - t)[..., None] * ft[:, None]  # 3 x N x N x 3
        # check if intersection points are inside the triangles:
        cross_products = torch.cross((ft[[1, 2, 0]] - ft)[:, None, :, None].expand(3, 3, n, n, 3),
                                     p[None] - ft[:, None, :, None],
                                     dim=4)  # 3 x 3 x N x N x 3
        left_of_all_edges = torch.einsum('hijkl,jl->hijk', cross_products, normals) > 0  # 3 x 3 x N x N
        inside = left_of_all_edges.all(dim=0)  # 3 x N x N
        # an edge intersects with a face, iff both verts are on opposite sides and the intersection is inside the face:
        intersect = opposite_sides.logical_and(inside)  # 3 x N x N
        # as one pair of intersecting faces leads to 2 intersecting edges, divide by 2:
        return .5 * (intersect.sum()) / len(faces)

    def compute_loss_parallel(self, batch: List[Mesh], p: float) -> torch.Tensor:
        """
        Previous, inefficient and thus deprecated version of the self-intersection-penalty
        Computes the average of squares of self-intersections for a random subset of each mesh in the batch.
        Uses multiprocessing for speedup.\n
        :param batch: the batch of meshes
        :param p: the fraction of meshes considered for loss calculation
        :return: average number of pairwise intersections
        """
        loss = torch.tensor(0., device=batch[0].device, requires_grad=True)
        faces = [b.faces[self.__rng.choice(range(len(b.faces)), size=int(p * len(b.faces)), replace=False)] for b in
                 batch]
        if self.__opt.multi_processing:
            args = [b.vs[faces[i]].detach() for i, b in enumerate(batch)]
            with Pool(len(batch)) as pool:
                losses = pool.map(self.compute_loss_with_loops, args)
        else:
            losses = [self.compute_loss_with_loops(b.vs[faces[i]].detach()) for i, b in enumerate(batch)]
        for lo in losses:
            loss = loss + torch.tensor(lo, dtype=torch.float32) ** 2
        return loss / len(batch)

    def compute_loss_with_loops(self, faces: torch.Tensor, other: torch.Tensor = None, idx_map: torch.Tensor = None,
                                threshold: int = 10, p=torch.tensor([0, 0, 0]), plane: Optional[str] = 'x'):
        """
        !!! SLOW !!!\n
        For given faces and vertices, calculates the number of self-intersections.
        Uses a divide and conquer approach for more efficient calculation\n
        :param plane: normal of the plane along which to make the cut
        :param faces: the coordinates of the triangles. (shape: nx3)
        :param other: other faces to check for self-intersections with faces. If None, faces are compared to themselves
        :param idx_map: which subset of faces other represents
        :param p: a point in the normals along which to perform the cuts
        :param threshold: at which number of faces to stop further dividing
        :return: the number of self-intersections
        """
        if other is not None:
            loss = 0
            normals = torch.cross(faces[:, 1, :] - faces[:, 0, :], faces[:, 2, :] - faces[:, 1, :])
            for i, face_1 in enumerate(other):
                for j, face_2 in enumerate(faces[idx_map[i] + 1:]):
                    if self.intersect(face_1, face_2, normals[idx_map[i]], normals[j]):
                        loss += 1
            return loss
        if len(faces) <= threshold or plane is None:
            loss = 0
            normals = torch.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
            for i, face_1 in enumerate(faces):
                for j, face_2 in enumerate(faces[i + 1:]):
                    if self.intersect(face_1, face_2, normals[i], normals[j]):
                        loss += 1
            return loss
        norm = {'x': torch.tensor([1, 0, 0]), 'y': torch.tensor([0, 1, 0]), 'z': torch.tensor([0, 0, 1])}[plane]
        sides = [torch.matmul(f - p, norm.to(faces.device).double()) for f in faces.split(1, dim=1)]
        left = (sides[0] < 0).logical_and(sides[1] < 0).logical_and(sides[2] < 0).squeeze()
        right = (sides[0] > 0).logical_and(sides[1] > 0).logical_and(sides[2] > 0).squeeze()
        both = torch.logical_not(left.logical_or(right))
        idx_map = both.nonzero().squeeze(dim=1)
        new_plane = 'y' if plane == 'x' else 'z' if plane == 'y' else None
        loss_left = self.compute_loss_with_loops(faces[left], threshold=threshold, plane=new_plane)
        loss_right = self.compute_loss_with_loops(faces[right], threshold=threshold, plane=new_plane)
        loss_both = self.compute_loss_with_loops(faces, other=faces[both], idx_map=idx_map, threshold=threshold,
                                                 plane=new_plane)
        return loss_left + loss_right + loss_both

    def intersect(self, tri_1: torch.Tensor, tri_2: torch.tensor, norm_1: torch.Tensor, norm_2: torch.Tensor) -> bool:
        """
        Checks whether two triangles intersect using the method proposed in:\n
        [1] S. H. Lo: Automatic mesh generation over intersecting surfaces, 1995.\n
        :param tri_1: coordinates of first triangle (shape: 3x3)
        :param tri_2: coordinates of second triangle (shape: 3x3)
        :param norm_1: face normal of tri_1
        :param norm_2: face normal of tri_2
        :return: True, if tri_1 and tri_2 intersect, False otherwise
        """
        for i in range(3):
            p11, p12 = tri_1[i], tri_1[(i + 1) % 3]
            p21, p22 = tri_2[i], tri_2[(i + 1) % 3]
            if self.opposite_side(p11, p12, tri_2, norm_2):
                p_inter = self.intersection_point(p11, p12, tri_2, norm_2)
                if self.inside_triangle(p_inter, tri_2, norm_2):
                    return True
            if self.opposite_side(p21, p22, tri_1, norm_1):
                p_inter = self.intersection_point(p21, p22, tri_1, norm_1)
                if self.inside_triangle(p_inter, tri_1, norm_1):
                    return True
        return False

    @staticmethod
    def opposite_side(p1: torch.Tensor, p2: torch.Tensor, triangle: torch.Tensor, norm: torch.Tensor) -> bool:
        """
        Checks whether p1 and p2 are on opposite sides of a plane containing triangle.\n
        :param p1: coordinates of first point (shape: 3,)
        :param p2: coordinates of second point (shape: 3,)
        :param triangle: triangle coordinates (shape: 3x3)
        :param norm: face normal of triangle
        :return: True, if p1 and p2 are on the opposite sides, False otherwise
        """
        return torch.matmul(p1 - triangle[0], norm) * torch.matmul(p2 - triangle[0], norm) < 0

    @staticmethod
    def intersection_point(p1: torch.Tensor, p2: torch.Tensor, triangle: torch.Tensor,
                           norm: torch.Tensor) -> torch.Tensor:
        """
        Computes the intersection point of a straight line from p1 to p2 with the plane containing triangle.\n
        :param p1: coordinates of first point (shape: 3,)
        :param p2: coordinates of second point (shape: 3,)
        :param triangle: triangle coordinates (shape: 3x3)
        :param norm: face normal of triangle
        :return: the intersection point
        """
        d = torch.matmul(p1 - triangle[0], norm)
        e = torch.matmul(p2 - triangle[0], norm)
        t = d / (d - e)
        return t * p2 + (1 - t) * p1

    @staticmethod
    def inside_triangle(p: torch.Tensor, triangle: torch.Tensor, norm: torch.Tensor) -> bool:
        """
        Checks whether point p is inside triangle.\n
        :param p: point coordinates (shape: 3,)
        :param triangle: triangle coordinates (shape: 3x3)
        :param norm: face normal of triangle
        :return: True if point is inside triangle, False otherwise
        """
        a, b, c = tuple(triangle)
        return torch.matmul(torch.cross(b - a, p - a), norm) > 0 \
            and torch.matmul(torch.cross(c - b, p - b), norm) > 0 \
            and torch.matmul(torch.cross(a - c, p - c), norm) > 0
