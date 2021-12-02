NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 26, 0, float32

python ../../mscoco.py
MXNET_CUDNN_AUTOTUNE_DEFAULT=0 
MXNET_GPU_MEM_POOL_TYPE=Round 
MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF=32 
python train_mask_rcnn.py --gpus $GPUS --dataset coco \
    --epochs $NUM_EPOCHS $USE_AMP --dtype $DTYPE --network resnet18_v1b --val-interval 2 -j 8 --verbose








