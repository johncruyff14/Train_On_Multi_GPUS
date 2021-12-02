NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 140 0

python ../../mscoco.py
python train_center_net.py --gpus $GPUS -j 60 --dataset coco --dtype $DTYPE \
     --batch-size 96 --log-interval 10 --epochs $NUM_EPOCHS $USE_AMP --lr-decay-epoch 90,120 \
     --lr 3.75e-4 --wd 0.00001 --momentum 0.9 --wh-weight 0.1 --warmup-epochs $WARM_EPOCHS \
     --val-interval 100  --network resnet50_v1b







