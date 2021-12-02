NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 120 0 float32

python ade20k.py
python ../train.py --dataset ade20k --model-zoo deeplab_resnest200_ade \
    --aux --lr 0.01 --syncbn --ngpus $NUM_GPUS --checkname deeplab_resnest200_ade \
    --epochs $NUM_EPOCHS --warmup-epochs $WARM_EPOCHS --dtype $DTYPE $USE_AMP \
    --base-size 520 --crop-size 480 --workers 48








