NUM_GPUS=4
. ../../set_env_vars.sh
set_env_vars 140 0

python ../../mscoco.py
python train_center_net.py --gpus $GPUS -j 60 --dataset coco --dtype $DTYPE \
     --batch-size 128 --log-interval 10 --epochs $NUM_EPOCHS $USE_AMP --lr-decay-epoch 90,120 \
     --lr 0.0005 --wd 0.00001 --momentum 0.9 --wh-weight 0.1 --warmup-epochs $WARM_EPOCHS \
     --val-interval 10  --network resnet18_v1b_dcnv2




