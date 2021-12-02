NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 280 2 float32

python ../../mscoco.py
python train_yolo3.py --network mobilenet1.0 --dataset coco --gpus $GPUS --data-shape 608 --dtype $DTYPE \
    --batch-size 64 -j 32 --log-interval 100 --lr-decay-epoch 220,250 --epochs $NUM_EPOCHS $USE_AMP \
    --syncbn --warmup-epochs $WARM_EPOCHS


