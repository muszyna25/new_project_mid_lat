#!/bin/bash

set -u # Exit if there is unintialized variable.

FN_LIST="ERA_data_list.txt"
VAR="x"
OUT_PATH="."

#python 2.py $FN_LIST $VAR $OUT_PATH

#INPUT_PATH="/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/data_classification_regression/raw_data/u/big_hdf5_files/chunk_files/reduced_chunk_files"
INPUT_PATH="/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/data_classification_regression/raw_data/u/big_hdf5_files"
#python 6.py $INPUT_PATH $OUT_PATH "train_0.h5"
TASK_TYPE=1

python 6.py $INPUT_PATH $OUT_PATH "test.h5" $TASK_TYPE



