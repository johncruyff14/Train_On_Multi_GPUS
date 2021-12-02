NUM_GPUS=1
. ../../../set_env_vars.sh
set_env_vars 200

python ../../sentiment_analysis_cnn.py --gpu $GPUS --batch_size 50 --epochs $NUM_EPOCHS --dropout 0.5 --model_mode static --data_name TREC
