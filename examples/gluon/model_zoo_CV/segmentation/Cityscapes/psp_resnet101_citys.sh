NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 240 0 float32

python cityscapes.py
if [ "$?" -ne "0" ]; then
   exit
fi

CUDA_VISIBLE_DEVICES=$GPUS
python ../train.py \
    --dataset citys \
    --model psp \
    --aux \
    --backbone resnet101 \
    --syncbn \
    --ngpus $NUM_GPUS \
    --checkname psp_resnet101_citys \
    --lr 0.01 \
	  --epochs $NUM_EPOCHS \
	  --warmup-epochs $WARM_EPOCHS \
	  --dtype $DTYPE \
	  $USE_AMP \
    --base-size 2048 \
    --crop-size 768 \
    --workers 48




