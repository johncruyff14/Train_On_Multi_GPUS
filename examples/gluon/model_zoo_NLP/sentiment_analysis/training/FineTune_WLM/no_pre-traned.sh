NUM_GPUS=1
. ../../../set_env_vars.sh
set_env_vars 3

pip install -U spaCy

python ../../finetune_lm.py --gpu $GPUS --batch_size 16 --bucket_type fixed --epochs $NUM_EPOCHS --dropout 0 --no_pretrained --lr 0.005 --valid_ratio 0.1 --save-prefix imdb_lstm_200  # Test Accuracy 85.60
