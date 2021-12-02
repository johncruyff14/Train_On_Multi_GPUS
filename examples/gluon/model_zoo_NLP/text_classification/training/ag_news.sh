NUM_GPUS=1
TRAIN_DATA_DIR=data
. ../../set_env_vars.sh
set_env_vars 25

../load_DB.sh ag_news

python ../fasttext_word_ngram.py --input $TRAIN_DATA_DIR/ag_news.train \
                                 --output $TRAIN_DATA_DIR/ag_news.gluon \
                                 --validation $TRAIN_DATA_DIR/ag_news.test \
                                 --ngrams 2 --epochs $NUM_EPOCHS --lr 0.1 --emsize 100 --gpu $GPUS


