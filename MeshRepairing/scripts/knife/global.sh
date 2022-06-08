#!/usr/bin/env bash

python repair_meshes.py ~/github/mrep/MeshRepairing/datasets/knife \
--batch-size 10 \
--epochs 100 \
--export-dir ./saved_data/knife \
--gpu 0 \
--n-edges 2328 \
--n-verts 778 \
--num-samples 5000 \
--self-inter 1 \
--test-each-epoch