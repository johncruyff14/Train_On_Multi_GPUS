NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 280 2 float32

python ../../mscoco.py
python train_yolo3.py --network darknet53 --dataset coco --gpus $GPUS --data-shape 320 --dtype $DTYPE \
    --batch-size 64 -j 32 --log-interval 100 --lr-decay-epoch 220,250 --epochs $NUM_EPOCHS $USE_AMP \
    --syncbn --warmup-epochs $WARM_EPOCHS --mixup --no-mixup-epochs 20 --label-smooth --no-wd




