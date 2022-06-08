#!/usr/bin/env bash

python hole_fill.py ./datasets/knife \
--batch-size 10 \
--convs-n-vs 16 32 64 \
--epochs 100 \
--export-dir ./saved_data/knife \
--gpu 0 \
--lr-n-vs 0.001 \
--lr-v-pos 0.01 \
--n-edges 2328 \
--n-verts 778 \
--test-each-epoch \
--use-meta