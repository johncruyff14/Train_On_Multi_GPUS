NUM_GPUS=4
. ../../../set_env_vars.sh
set_env_vars

python ../../large_word_language_model.py --gpus $GPUS --clip=10



