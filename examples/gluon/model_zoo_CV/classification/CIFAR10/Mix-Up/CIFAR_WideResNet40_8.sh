MODEL=cifar_wideresnet40_8
NUM_GPUS=2

. ../../../set_env_vars.sh
set_env_vars 220 0 float32

python train_mixup_cifar10.py \
    --num-epochs $NUM_EPOCHS $USE_AMP --mode hybrid --num-gpus $NUM_GPUS -j 2 --batch-size 64 \
    --dtype $DTYPE --wd 0.0005 --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 60,120,160 --model=$MODEL

