NUM_GPUS=1
TRAIN_DATA_DIR=data
OUTPUT_DIR=output/snli-intra

if [ $# -ge 1 ]; then
  MODEL=$1
else
  MODEL=da
fi

export SNLI="snli_1.0"

. ../../set_env_vars.sh
set_env_vars 300


../load_DB.sh $TRAIN_DATA_DIR

DATA_SNLI=${TRAIN_DATA_DIR}/${SNLI}

python ../main.py --train-file ${DATA_SNLI}/train.txt --test-file ${DATA_SNLI}/dev.txt --output-dir $OUTPUT_DIR --batch-size 32 --print-interval 5000 --lr 0.025 --epochs $NUM_EPOCHS --gpu-id $GPUS --dropout 0.2 --weight-decay 1e-5 --intra-attention --fix-embedding --model $MODEL

python ../main.py --test-file ${DATA_SNLI}/test.txt --model-dir $OUTPUT_DIR --gpu-id $GPUS --mode test --output-dir ${OUTPUT_DIR}/test --model $MODEL



