#!/usr/bin/env bash

python global_eval.py ~/github/mrep/MeshRepairing/datasets/evaluation/pillows \
--batch-size 1 \
--export-dir ./saved_data/evaluation/pillows_global \
--gpu 0 \
--n-edges 2436 \
--n-verts 814