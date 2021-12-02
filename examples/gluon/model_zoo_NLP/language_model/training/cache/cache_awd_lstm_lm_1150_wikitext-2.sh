NUM_GPUS=1
. ../../../set_env_vars.sh
set_env_vars

python ../../cache_language_model.py --gpus $GPUS --model_name awd_lstm_lm_1150


