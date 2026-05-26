export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

PORT=6001
HOST=0.0.0.0

python -m server.TKGR_server \
    --host=$HOST \
    --port=$PORT \
    --model_path ./server/model_ckpts/ICEWS14s_divide/best-model.pth \
    --gpu 0 \
    --data ICEWS14s_divide \
    --temporal_bias 0.01 \
    --path_encode GRU_time \
|tee ./log/tkgr_server/ice14${PORT}.log


