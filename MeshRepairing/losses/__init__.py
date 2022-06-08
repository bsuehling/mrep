from losses.losses import *


class GlobalCriterion:
    def __init__(self, opt):
        self.__opt = opt
        self.__chamfer = ChamferDistance(opt)
        self.__edge_length = EdgeLengthLoss()
        self.__self_inter = SelfIntersectionPenalty(opt)

    def __call__(self, x_batch, y_batch):
        loss = torch.tensor(0., device=x_batch[0].device, requires_grad=True)
        meta = dict()
        if self.__opt.chamfer > 0:
            chamfer = self.__opt.chamfer * self.__chamfer(x_batch, y_batch)
            loss = loss + chamfer
            meta['chamfer'] = chamfer
        if self.__opt.edge_length > 0:
            edge_length = self.__opt.edge_length * self.__edge_length(x_batch)
            loss = loss + edge_length
            meta['edge_length'] = self.__opt.edge_length * self.__edge_length(x_batch)
        if self.__opt.self_inter > 0:
            self_inter = self.__opt.self_inter * self.__self_inter(x_batch)
            loss = loss + self_inter
            meta['self_inter'] = self_inter
        return loss, meta
