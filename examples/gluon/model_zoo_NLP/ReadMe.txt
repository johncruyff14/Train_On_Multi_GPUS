This directory contains 51 scripts for Language Model (11), Machine Translation(2), Sentiment Analysis (30), Text Classification (3), Natural Language Inference (4), and Word Embedding (1 script, but, in fact, it launches 15 different model trainings), which were taken from Gluon NLP model_zoo (https://nlp.gluon.ai/model_zoo/index.html) and adapted for execution in NVidia MxNet containers on dbcluster.

For training, these scripts use datasets downloaded from the Internet.
Most of the scripts (*.sc files), which could be launched could be found in different subdirectorie of "training" directory for corresponding Model. For instance, see
model_zoo_NLP/machine_translation/training/GoogleNeural/gnmt.sh
OR
model_zoo_NLP/language_model/training/cache/*.sh

The only exception is the script model_zoo_NLP/word_embeddings/run_all.sh, which launches training for 15 different datasets.
We noticed that "run_all.sh' script stops working after the first problem appeared with any dataset. Please, set the environment variable CATCH_EXEPTION to 1, if you want to continue the execution of that script for the next dataset.


Any *.sh script can be run directly without any changes or setting any environment variables. In this case, you should expect a significant total execution time because the default values for the number of epochs for training are large enough.

You can change the preset value for NUM_GPUS by setting the environment variables _NUM_GPUS.

To run a quick test of the general functionality of each script, you could set the following environment variables: NUM_EPOCHS. For instance, when
export NUM_EPOCHS=1

the script will do only one epoch of training.


The current state (as of 03/01/2021) of the results obtained for different modes of these scripts could be found in "GluonNLP testing.xlsx"





 
