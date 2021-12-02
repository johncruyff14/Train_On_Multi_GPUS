NUM_GPUS=8
. ../../set_env_vars.sh
set_env_vars 120 0 float32

# Instaling unoficial packages
pip install html5lib --user
pip install googleDriveFileDownloader --user
# Loading Database
python mhp_v1.py

CUDA_VISIBLE_DEVICES=$GPUS
python ../train.py \
	--dataset mhpv1 \
	--model icnet \
	--backbone resnet50 \
	--syncbn \
	--ngpus $NUM_GPUS \
	--optimizer adam \
	--lr 0.00001 \
	--epochs $NUM_EPOCHS \
	--warmup-epochs $WARM_EPOCHS \
	--dtype $DTYPE \
	$USE_AMP \
	--base-size 768 \
	--crop-size 768 \
	--workers 32 \
	--batch-size 16 \
	--log-interval 1



