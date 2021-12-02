NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 1000 0 float32

python cityscapes.py
if [ "$?" -ne "0" ]; then
   exit
fi

# cmd for training
CUDA_VISIBLE_DEVICES=$GPUS
python ../train.py \
	--dataset citys \
	--model fastscnn \
	--aux \
	--ngpus $NUM_GPUS \
	--lr 0.045 \
	--epochs $NUM_EPOCHS \
	--warmup-epochs $WARM_EPOCHS \
	--dtype $DTYPE \
	$USE_AMP \
	--base-size 2048 \
	--crop-size 1024 \
	--workers 32 \
	--batch-size 32

# cmd for evaluation
python ../test.py \
	--model fastscnn \
	--dataset citys \
	--batch-size 8 \
	--ngpus $NUM_GPUS \
	--eval \
	--pretrained \
	--aux \
	--base-size 2048 \
	--height 1024 \
	--width 2048





