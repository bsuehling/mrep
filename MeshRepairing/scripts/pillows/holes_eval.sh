#!/usr/bin/env bash

python hole_fill_eval.py ./datasets/evaluation/pillows \
--batch-size 1 \
--convs-n-vs 16 32 64 \
--export-dir ./saved_data/evaluation/pillows_holes \
--gpu 0 \
--n-edges 2436 \
--n-verts 814 \
--use-meta \
--verts-max 10