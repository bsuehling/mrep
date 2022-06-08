import torch
import torch.nn as nn
import torch.nn.functional as F


class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, bias=False))

    def forward(self, x, meshes):
        x = self.conv0(x, meshes)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, meshes)
        x += x1
        x = F.relu(x)
        return x


class MeshConv(nn.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    x: edge features (Batch x Features x Edges)
    meshes: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """

    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=bias)
        self.k = k

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, meshes):
        x = x.squeeze(-1)
        g = torch.cat([self.pad_gemm(mesh, x.shape[2], x.device) for mesh in meshes], 0)
        # build 'neighborhood image' and apply convolution
        g = self.create_gemm(x, g)
        x = self.conv(g.float())
        return x

    @staticmethod
    def flatten_gemm_inds(gi):
        (b, ne, nn) = gi.shape
        ne += 1
        batch_n = torch.floor(torch.arange(b * ne, device=gi.device).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # flatten gi
        gi = gi.float() + add_fac[:, 1:, :]
        return gi

    def create_gemm(self, x, gi):
        """ Gathers the edge features (x) from the 1-ring indices (gi).
        Applies symmetric functions to handle order invariance.
        Returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        gi_shape = gi.shape
        # pad the first row of every sample in batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        x = torch.cat((padding, x), dim=2)
        gi = gi + 1  # shift

        # first flatten indices
        gi_flat = self.flatten_gemm_inds(gi)
        gi_flat = gi_flat.view(-1).long()
        #
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=gi_flat)
        f = f.view(gi_shape[0], gi_shape[1], gi_shape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # apply the symmetric functions for an equivariant conv
        x_1 = f[:, :, :, 1] + f[:, :, :, 3]
        x_2 = f[:, :, :, 2] + f[:, :, :, 4]
        x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
        x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
        f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        return f

    @staticmethod
    def pad_gemm(m, xsz, device):
        """ extracts one-ring neighbors (4x) -> m.gemm_edges
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., xsz x 5
        """
        padded_gemm = torch.tensor(m.gemm_edges, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm
