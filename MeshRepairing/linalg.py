from typing import Tuple

import torch


def length_ratio(edge: Tuple[int, int], vert: torch.Tensor) -> torch.Tensor:
    """
    computes length ratio between an edge and distance between a vertex and a line containing the edge in 3D\n
    :param edge: the edge (tuple of elements with shape: (3,))
    :param vert: the vertex (shape: (3,))
    :return: edge length ratio
    """
    return torch.linalg.norm(torch.cross(vert - edge[0], vert - edge[1]))


def inner_angle(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
    computes the angle between two edges in 3D\n
    :param vector1: the first edge (shape: (3,)
    :param vector2: the second edge (shape: (3,)
    :return: the angle in radian
    """
    if type(vector1) is not torch.Tensor:
        vector1 = torch.tensor(vector1)
    if type(vector2) is not torch.Tensor:
        vector2 = torch.tensor(vector2)
    return torch.acos(torch.sum(vector1 * vector2) / (torch.linalg.norm(vector1) * torch.linalg.norm(vector2)))


def dihedral_angle(face1: torch.Tensor, face2: torch.Tensor) -> torch.Tensor:
    """
    computes the angle between two faces in 3D
    :param face1: the first face (shape: (3,3))
    :param face2:  the second face (shape: (3,3))
    :return: the angle between planes containing face1 and face2
    """
    normal1 = torch.cross(face1[1] - face1[0], face1[2] - face1[0])
    normal2 = torch.cross(face2[1] - face2[0], face2[2] - face2[0])
    return inner_angle(normal1, normal2)
