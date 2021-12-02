MODEL=cifar_resnext29_16x64d
NUM_GPUS=4

. ../../../set_env_vars.sh
set_env_vars 320 0 float32

python train_mixup_cifar10.py \
    --num-epochs $NUM_EPOCHS $USE_AMP --mode hybrid --num-gpus $NUM_GPUS -j 4 --batch-size 32 \
    --dtype $DTYPE --wd 0.0005 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 150,225 --model=$MODEL

