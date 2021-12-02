NUM_GPUS=4
. ../../set_env_vars.sh
set_env_vars 20 0

python ../../pascal_voc.py
MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python train_faster_rcnn.py --gpus $GPUS \
    --epochs $NUM_EPOCHS $USE_AMP --dataset voc --network resnet50_v1b



