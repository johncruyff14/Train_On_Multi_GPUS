NUM_GPUS=8
. ../../../set_env_vars.sh
set_env_vars 60

MXNET_GPU_MEM_POOL_TYPE=Round 
python ../../train_transformer.py --dataset WMT2014BPE --src_lang en --tgt_lang de --batch_size 2700 --optimizer adam --num_accumulated 16 --lr 3.0 --warmup_steps 4000 --save_dir transformer_en_de_u512 --epochs $NUM_EPOCHS --gpus $GPUS --scaled  --average_start 15 --num_buckets 20 --bucket_scheme exp --bleu 13a --log_interval 10
