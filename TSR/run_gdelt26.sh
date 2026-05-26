#!/bin/bash

python main.py -d GDELT26 --pos_dim 60 --embed_dim 600 \
--temporal_bias 0.001 \
--bs 128 --gpu 0 --seed 0 --path_encode GRU_time
