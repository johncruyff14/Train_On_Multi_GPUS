NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 200 4 float32

python ../../pascal_voc.py
python train_yolo3.py --network darknet53 --dataset voc --data-shape 416 --dtype $DTYPE \
      --gpus $GPUS --batch-size 64 -j 16 \
      --log-interval 100 --lr-decay-epoch 160,180 \
      --epochs $NUM_EPOCHS $USE_AMP --syncbn --warmup-epochs $WARM_EPOCHS



