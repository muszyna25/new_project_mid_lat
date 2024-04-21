#!/bin/bash

module load tensorflow/intel-1.12.0-py36

set -u # Exit if there is unintialized variable.

######################################## Settings ########################################
WORK_DIR="/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/data_classification_regression/data_for_paper/raw_data"
L_FILES="all_fold_files.txt"
#L_FILES="test_data.txt"
OUT_PATH="/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/data_classification_regression/data_for_paper/all_data"

OUT_PATH_A="${OUT_PATH}/source_a"
OUT_PATH_B="${OUT_PATH}/source_b"
OUT_PATH_C="${OUT_PATH}/source_c"
OUT_PATH_D="${OUT_PATH}/source_d"

mkdir -p ${OUT_PATH_A}
mkdir -p ${OUT_PATH_B}
mkdir -p ${OUT_PATH_C}
mkdir -p ${OUT_PATH_D}

SOURCE_A="${WORK_DIR}/u/big_hdf5_files/chunk_files"
SOURCE_B="${WORK_DIR}/v/big_hdf5_files/chunk_files"
SOURCE_C="${WORK_DIR}/pv/big_hdf5_files/chunk_files"
SOURCE_D="${WORK_DIR}/t/big_hdf5_files/chunk_files"
SOURCE_E="${WORK_DIR}/z/big_hdf5_files/chunk_files"

BATCH_START_TIME=$(date)
echo "[+] ------START TIME (ST): $BATCH_START_TIME------"

python merge_files.py ${L_FILES} ${SOURCE_A} ${SOURCE_B} ${OUT_PATH_A} && 
python merge_files.py ${L_FILES} ${OUT_PATH_A} ${SOURCE_C} ${OUT_PATH_B} && 
python merge_files.py ${L_FILES} ${OUT_PATH_B} ${SOURCE_D} ${OUT_PATH_C} && 
python merge_files.py ${L_FILES} ${OUT_PATH_C} ${SOURCE_E} ${OUT_PATH_D} 

