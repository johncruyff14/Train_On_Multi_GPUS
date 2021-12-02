MODEL=cifar_resnet110_v2
NUM_GPUS=1

. ../../../set_env_vars.sh
set_env_vars 200 0 float32

python train_mixup_cifar10.py \
    --num-epochs $NUM_EPOCHS $USE_AMP --mode hybrid --num-gpus $NUM_GPUS -j 2 --batch-size 128 \
    --dtype $DTYPE --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model=$MODEL

