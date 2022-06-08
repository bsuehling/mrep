#!/usr/bin/env bash

python hole_fill.py ./datasets/pillows \
--batch-size 10 \
--convs-n-vs 16 32 64 \
--epochs 100 \
--export-dir ./saved_data/pillows \
--gpu 0 \
--lr-n-vs 0.001 \
--lr-v-pos 0.01 \
--n-edges 2367 \
--n-verts 791 \
--test-each-epoch \
--use-meta \
--verts-max 10