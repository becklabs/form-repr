#!/bin/bash

python ../tools/get_embedding.py \
    --checkpoint ../checkpoints/pose3d/latest_epoch.bin \
    --input_path ../data/poses/oliver/ \
    --output_path ../data/embed/oliver/
