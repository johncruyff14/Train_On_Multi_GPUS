This directory contains 146 scripts for Classification, Object Detection, and Segmentation, which were taken from GluonCV model_zoo (https://cv.gluon.ai/model_zoo/index.html) and adapted for execution in NVidia MxNet containers on dbcluster.

For training, these scripts use the shared data available in dbcluster (/data/imagenet/train-val-recordio-256) OR datasets downloaded from the Internet. When the environment variable DATASET is defined as some directory name, each script will try to use OR download the corresponding dataset in that location. In that case, these data will be available for ANY docker session. Otherwise, the datasets downloaded from the Internet will be available only during the same docker session, but they could be used by similar scripts. In general, this will save a lot of time because some of the datasets used for training are quite large.

Any *.sh script can be run directly without any changes or setting any environment variables. In this case, you should expect a significant total execution time because the default values for the number of epochs for training are large enough.

You can change the preset values for MODEL, TRAIN_DATA_DIR, NUM_GPUS by setting the environment variables _MODEL, _TRAIN_DATA_DIR, _NUM_GPUS.

To run a quick test of the general functionality of each script, you could set the following environment variables: NUM_EPOCHS, WARM_EPOCHS. For instance, when
export WARM_EPOCHS=0
export NUM_EPOCHS=1

the script will do only one epoch of training.

As of today 12/07/2020 55 scrips for Classification and 24 scripts for Segmentation could be launched for both float16 and float32 data types.
For each such script the data type is defined by default OR by the value of the environment variable DTYPE, which could be set to float16 OR float32.

For all these 146 scripts, the possibility to launch them with OR without AMP is implemented. To do that just set the environment variable USE_AMP to 1 OR 0.

As of today 12/07/2020, the usage of AMP

1. requires DTYPE=float32.
2. is not always successful

The current state of the results obtained for different modes of these scripts could be found in "GluonCV testing.xlsx"





 
