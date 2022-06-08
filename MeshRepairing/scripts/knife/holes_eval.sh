#!/usr/bin/env bash

python hole_fill_eval.py ./datasets/evaluation/knife \
--batch-size 1 \
--convs-n-vs 16 32 48 \
--export-dir ./saved_data/evaluation/knife_holes \
--gpu 0 \
--n-edges 2328 \
--n-verts 778 \
--use-meta