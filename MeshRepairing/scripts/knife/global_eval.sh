#!/usr/bin/env bash

python global_eval.py ~/github/mrep/MeshRepairing/datasets/evaluation/knife \
--batch-size 1 \
--export-dir ./saved_data/evaluation/knife_global \
--gpu 0 \
--n-edges 2328 \
--n-verts 778