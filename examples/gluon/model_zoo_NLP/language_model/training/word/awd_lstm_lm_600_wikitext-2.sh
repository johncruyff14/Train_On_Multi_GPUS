NUM_GPUS=1
. ../../../set_env_vars.sh 750
set_env_vars 750

python ../../word_language_model.py --gpu $GPUS --emsize 200 --nhid 600 --epochs $NUM_EPOCHS --dropout 0.2 --dropout_h 0.1 --dropout_i 0.3 --dropout_e 0.05 --weight_drop 0.2 --tied --ntasgd --lr_update_interval 30 --lr_update_factor 0.1 --save awd_lstm_lm_600_wikitext-2

