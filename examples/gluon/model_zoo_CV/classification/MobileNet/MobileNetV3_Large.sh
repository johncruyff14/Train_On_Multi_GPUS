MODEL=mobilenetv3_large
TRAIN_DATA_DIR=/data/imagenet/train-val-recordio-256
NUM_GPUS=8
export MXNET_SAFE_ACCUMULATION=1

. ../../set_env_vars.sh
set_env_vars 360 5 float16

python ../gluon_train_imagenet.py \
  --rec-train $TRAIN_DATA_DIR/train.rec --rec-train-idx $TRAIN_DATA_DIR/train.idx \
  --rec-val $TRAIN_DATA_DIR/val.rec --rec-val-idx $TRAIN_DATA_DIR/val.idx \
  --model $MODEL --mode hybrid $USE_AMP \
  --lr 2.6 --wd 0.00003 --lr-mode cosine \
  --num-epochs $NUM_EPOCHS --batch-size 256 --num-gpus $NUM_GPUS -j 96 \
  --warmup-epochs $WARM_EPOCHS --dtype $DTYPE --no-wd --last-gamma \
  --use-rec --label-smoothing \
  --save-dir params_$MODEL \
  --logging-file $MODEL.log


