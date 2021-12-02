NUM_GPUS=4
. ../../set_env_vars.sh
set_env_vars 30 0 float32

python ../../mscoco.py
CUDA_VISIBLE_DEVICES=$GPUS
python ../train.py --dataset coco --model fcn --aux --backbone resnet101 \
	  --epochs $NUM_EPOCHS --warmup-epochs $WARM_EPOCHS --dtype $DTYPE $USE_AMP \
    --lr 0.01 --syncbn --ngpus $NUM_GPUS --checkname res101







