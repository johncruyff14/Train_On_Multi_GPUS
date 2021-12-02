MODEL=vgg11_bn
TRAIN_DATA_DIR=/data/imagenet/train-val-recordio-256
NUM_GPUS=8

. ../../set_env_vars.sh
set_env_vars 100 0 float32

python ../gluon_train_imagenet.py \
  --rec-train $TRAIN_DATA_DIR/train.rec --rec-train-idx $TRAIN_DATA_DIR/train.idx \
  --rec-val $TRAIN_DATA_DIR/val.rec --rec-val-idx $TRAIN_DATA_DIR/val.idx \
  --model $MODEL --mode hybrid $USE_AMP \
  --batch-size 32 --num-gpus $NUM_GPUS -j 64 \
  --num-epochs $NUM_EPOCHS --lr 0.01 --lr-decay 0.1 --lr-decay-epoch 50,80 \
  --dtype $DTYPE --warmup-epochs $WARM_EPOCHS \
  --use-rec --save-dir params_$MODEL



