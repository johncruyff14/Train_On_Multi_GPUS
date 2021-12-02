NUM_GPUS=1
TRAIN_DATA_DIR=data
. ../../set_env_vars.sh
set_env_vars 10

../load_DB.sh yelp_review_polarity

python ../fasttext_word_ngram.py --input $TRAIN_DATA_DIR/yelp_review_polarity.train \
          	                 --output $TRAIN_DATA_DIR/yelp_review_polarity.gluon \
                                 --validation $TRAIN_DATA_DIR/yelp_review_polarity.test \
                                 --ngrams 1 --epochs $NUM_EPOCHS --lr 0.1 --emsize 100 --gpu $GPUS
