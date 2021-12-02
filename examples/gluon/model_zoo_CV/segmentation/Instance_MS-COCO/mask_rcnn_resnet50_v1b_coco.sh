NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 26, 0, float32

python ../../mscoco.py
python train_mask_rcnn.py --gpus $GPUS --epochs $NUM_EPOCHS --dtype $DTYPE $USE_AMP
python eval_mask_rcnn.py --gpus $GPUS --pretrained mask_rcnn_resnet50_v1b_coco_best.params &>> mask_rcnn_resnet50_v1b_coco_train.log


