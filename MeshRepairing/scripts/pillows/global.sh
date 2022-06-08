#!/usr/bin/env bash

python repair_meshes.py ~/github/mrep/MeshRepairing/datasets/pillows \
--batch-size 10 \
--epochs 100 \
--export-dir ./saved_data/pillows \
--gpu 0 \
--n-edges 2367 \
--n-verts 791 \
--num-samples 5000 \
--self-inter 1 \
--test-each-epoch