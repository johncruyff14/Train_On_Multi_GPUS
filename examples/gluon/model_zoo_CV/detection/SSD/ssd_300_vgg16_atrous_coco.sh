NUM_GPUS=4
. ../../set_env_vars.sh
set_env_vars 240 0 float32

python ../../mscoco.py
python train_ssd.py --gpus $GPUS -j 32 --dtype $DTYPE \
     --network vgg16_atrous --data-shape 300 --dataset coco \
     --lr 0.001 --lr-decay-epoch 160,200 --lr-decay 0.1 --epochs $NUM_EPOCHS $USE_AMP




