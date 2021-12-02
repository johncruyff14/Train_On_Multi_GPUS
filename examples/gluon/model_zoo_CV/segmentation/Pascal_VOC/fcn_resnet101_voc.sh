NUM_GPUS=4
. ../../set_env_vars.sh
set_env_vars 50 0 float32

python ../../mscoco.py
python ../../pascal_voc.py

# First finetuning COCO dataset pretrained model on augmented set
# If you would like to train from scratch on COCO, please see fcn_resnet101_coco.sh
CUDA_VISIBLE_DEVICES=$GPUS
python ../train.py --dataset pascal_aug --model-zoo fcn_resnet101_coco --aux \
	  --epochs $NUM_EPOCHS --warmup-epochs $WARM_EPOCHS --dtype $DTYPE $USE_AMP \
    --lr 0.001 --syncbn --ngpus $NUM_GPUS --checkname res101
# Finetuning on original set
python ../train.py --dataset pascal_voc --model fcn --aux --backbone resnet101 \
	  --epochs $NUM_EPOCHS --warmup-epochs $WARM_EPOCHS --dtype $DTYPE $USE_AMP \
    --lr 0.0001 --syncbn --ngpus $NUM_GPUS --checkname res101 --resume runs/pascal_aug/fcn/res101/checkpoint.params







