NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 26 0

python ../../mscoco.py
python train_faster_rcnn.py --gpus $GPUS --dataset coco --batch-size 16 --use-fpn \
    --lr 0.02 --lr-warmup 500 --rpn-smoothl1-rho 0.001 --rcnn-smoothl1-rho 0.001 -j16 --network resnest50 \
    --executor-threads 8 --val-interval 1 --epochs $NUM_EPOCHS $USE_AMP \
    --lr-decay-epoch 20,24 --lr-warmup-factor 0.3333 --norm-layer syncbn --disable-hybridization
