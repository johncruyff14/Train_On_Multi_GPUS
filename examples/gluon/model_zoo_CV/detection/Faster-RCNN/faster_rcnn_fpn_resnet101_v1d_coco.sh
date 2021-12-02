NUM_GPUS=8
CUDNN_AUTOTUNE_DEFAULT=0
MXNET_GPU_MEM_POOL_TYPE=Round 
MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF=32

. ../../set_env_vars.sh
set_env_vars 26 0

python ../../mscoco.py
python train_faster_rcnn.py --gpus $GPUS --dataset coco --network resnet101_v1d \
     --epochs $NUM_EPOCHS $USE_AMP --lr-decay-epoch 17,23 --val-interval 2 --use-fpn



