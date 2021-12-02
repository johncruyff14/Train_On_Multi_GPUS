MODEL=cifar_wideresnet28_10
NUM_GPUS=1

. ../../../set_env_vars.sh
set_env_vars 200 0 float32

python train_cifar10.py \
    --num-epochs $NUM_EPOCHS $USE_AMP --mode hybrid --num-gpus $NUM_GPUS -j 8 --batch-size 128 \
    --dtype $DTYPE --wd 0.0005 --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 60,120,160 --model=$MODEL
 
