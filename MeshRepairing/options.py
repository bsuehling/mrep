import argparse
import random


class Options:
    def __init__(self):
        self.args = None
        self.__parse_args()

    def __parse_args(self):
        p = argparse.ArgumentParser(description='mesh repairing options')
        aa = p.add_argument
        aa('dataroot', help='path to dataset')
        aa('--batch-size', default=1, type=int)
        aa('--chamfer', default=1, type=float, help='weight of chamfer distance for loss computation')
        aa('--continue-train', action='store_true', help='continue training with the latest model saved')
        aa('--convs-glob', default=[8, 16, 32, 48], type=int, help='conv filters for global approach')
        aa('--convs-n-vs', nargs='+', default=[8, 16, 32], type=int, help='conv filters for NVsNet')
        aa('--convs-v-pos', nargs='+', default=[8, 16, 32, 48], type=int, help='conv filters for VertPosNet')
        aa('--convs-smooth', nargs='+', default=[8, 16, 32, 48], type=int, help='conv filters for SmoothNet')
        aa('--data-random', action='store_true',
           help='per epoch, use only one, random, faulty mesh per ground truth mesh')
        aa('--edge-length', default=0, type=float, help='weight of edge length loss for loss computation')
        aa('--epochs', default=100, type=int, help='number of epochs')
        aa('--export-dir', default='./saved_data', help='where to save models')
        aa('--gpu', default=-1, type=int, help='which gpu to use. -1 means cpu')
        aa('--init-weights', default=0.002, type=float, help='initialize NN with this size')
        aa('--laplacian', default=1, type=float, help='weight of laplacian smoothing for loss computation')
        aa('--lr-glob', default=0.01, type=float, help='initial learning rate for global approach')
        aa('--lr-n-vs', default=0.01, type=float, help='initial learning rate for NVsNet')
        aa('--lr-smooth', default=0.01, type=float, help='initial learning rate for SmoothNet')
        aa('--lr-v-pos', default=0.01, type=float, help='initial learning rate for VPosNet')
        aa('--multi-processing', action='store_true',
           help='whether to use multiprocessing for loss computation')
        aa('--n-edges', default=1500, type=int, help='number of input edges (will include dummy edges)')
        aa('--n-verts', default=642, type=int, help='number of input verts')
        aa('--num-samples', default=1000, type=int, help='number of points to sample reconstruction with', metavar='N')
        aa('--num-workers', default=0, type=int, help='how many subprocesses to use for data loading')
        aa('--seed', type=int, help='which random seed to use. If None, one will created randomly')
        aa('--self-inter', default=0, type=float, help='weight of self-intersection loss for loss computation')
        aa('--start-epoch', default=0, type=int,
           help='in which epoch to start training. helpful in combination with --continue-train')
        aa('--test-each-epoch', action='store_true', help='run tests at the end of each epoch')
        aa('--use-meta', action='store_true', help='whether to use meta information about deleted faces and vertices')
        aa('--verts-max', default=8, type=int, help='the maximum number of required vertices predictable by NVertsNet')
        self.args = p.parse_args()
        print(100 * '-')
        print('Arguments passed to program:')
        options = vars(self.args)
        if options['seed'] is None:
            options['seed'] = random.randint(0, 100000)
        for option in options:
            print(f"{option + ':':20} {options[option]}")
