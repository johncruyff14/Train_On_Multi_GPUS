NUM_GPUS=1
. ../../../set_env_vars.sh
set_env_vars 750

python ../../word_language_model.py --gpu $GPUS --tied --ntasgd --epochs $NUM_EPOCHS --lr_update_interval 30 --lr_update_factor 0.1 --save awd_lstm_lm_1150_wikitext-2



