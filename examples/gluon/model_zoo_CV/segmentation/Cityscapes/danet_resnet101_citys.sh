NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 240 0 float32

python cityscapes.py
if [ "$?" -ne "0" ]; then
   exit
fi

# cmd for training
CUDA_VISIBLE_DEVICES=$GPUS
python ../train.py \
    --dataset citys \
    --model danet \
    --backbone resnet101 \
    --syncbn \
    --ngpus $NUM_GPUS \
    --lr 0.01 \
	  --epochs $NUM_EPOCHS \
	  --warmup-epochs $WARM_EPOCHS \
	  --dtype $DTYPE \
	  $USE_AMP \
    --base-size 2048 \
    --crop-size 768 \
    --workers 32 \

# cmd for evaluation
python ../test.py \
    --model-zoo danet_resnet101_citys \
    --dataset citys \
    --batch-size 8 \
    --ngpus $NUM_GPUS \
    --eval \
    --pretrained \
    --base-size 2048 \
    --crop-size 768








