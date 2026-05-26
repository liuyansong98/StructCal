#!/bin/bash

python main.py -d ICEWS14 --pos_dim 60 --embed_dim 600 \
--temporal_bias 0.01 \
--bs 128 --gpu 0 --seed 0 --path_encode GRU_time
