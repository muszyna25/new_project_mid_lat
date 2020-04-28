#!/bin/bash

module load tensorflow/intel-1.12.0-py36

set -u # Exit if there is unintialized variable.

######################################## Settings ########################################
WORK_DIR="/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/data_classification_regression"

## Extract subimages (regions) from ERA data.
FLAG_EXTRACT_LABELS="1" # 1 -- yes; 0 --no.
FLAG_EXTRACT_DATA="1" # 1 -- yes; 0 -- no.

VERSION="new"      # Version of labeling: https://docs.google.com/document/d/1hGE5t1Q6V9UrZ0gSwaetktFtL7j3BSyMvs85kGAI0Hg/edit?usp=sharing

VAR='x'             # Variable in ERA-interim data. If VAR='x' then take all variables.

TASK_TYPE=0         # Regression: 1, Classification 0 

######################################## Set python scripts ########################################

## Print current directory.
PWD="$(pwd)"
echo "$PWD"

# Get file names and create a list of files: 1.py, 2.py, ..., 8.py.
FILES=()
yourfilenames=`ls *.py`
for f in $yourfilenames
do
  FILES+=("$f")
done
echo "${FILES[*]}"

######################################## 1.py ######################################## # Extract lables.
L_BIN_MASKS="ETHZ_data_list.txt"
OUT_LABELS_PATH="${WORK_DIR}/${VERSION}/labels"

if [ $FLAG_EXTRACT_LABELS -eq "1" ]; then
    mkdir -p ${OUT_LABELS_PATH}
fi

######################################## 2.py ######################################## # Extract raw data regions.
L_ERA_FILES="ERA_data_list.txt"
OUT_RAW_DATA_PATH="${WORK_DIR}/raw_data/${VAR}"

if [ $FLAG_EXTRACT_DATA -eq "1" ]; then
    mkdir -p ${OUT_RAW_DATA_PATH}
fi

######################################## 3.py ######################################## # Generate one HDF5 file from data regions.
BIG_FILE_NAME_HDF5="$VAR.h5"
L_REGION_FILES="bin_mask_regions_list.txt"
OUT_BIG_FILES_HDF5="${WORK_DIR}/raw_data/${VAR}/big_hdf5_files"

mkdir -p ${OUT_BIG_FILES_HDF5}

######################################## 4.py ######################################## # Shuffle data in HDF5 file.
OUT_SHUFFLED_0_FILE="${VAR}_shuffled_0.h5"

######################################## 5.py ######################################## # Resample data in HDF5 file.
OUT_RESAMPLED_FILE="${VAR}_resampled.h5"

######################################## 4.py ######################################## # Repeat shuffuling on resampled data.
OUT_SHUFFLED_1_FILE="${VAR}_shuffled_1.h5"

######################################## 6.py ######################################## # Generate k-folds & normalize, and then divide them into m-subsets (list of files for generator).
OUT_CHUNK_FILES_HDF5="${OUT_BIG_FILES_HDF5}/chunk_files"

mkdir -p ${OUT_CHUNK_FILES_HDF5}

######################################## MAIN ########################################

python ${FILES[0]} ${L_BIN_MASKS} ${OUT_LABELS_PATH} && 
python ${FILES[1]} ${L_ERA_FILES} ${VAR} ${OUT_RAW_DATA_PATH} && 
python ${FILES[2]} ${L_REGION_FILES} ${BIG_FILE_NAME_HDF5} ${OUT_LABELS_PATH} ${OUT_RAW_DATA_PATH} ${OUT_BIG_FILES_HDF5} &&
python ${FILES[3]} ${OUT_BIG_FILES_HDF5} ${OUT_BIG_FILES_HDF5} ${BIG_FILE_NAME_HDF5} ${OUT_SHUFFLED_0_FILE} &&
python ${FILES[4]} ${OUT_BIG_FILES_HDF5} ${OUT_BIG_FILES_HDF5} ${OUT_SHUFFLED_0_FILE} ${OUT_RESAMPLED_FILE} &&
python ${FILES[3]} ${OUT_BIG_FILES_HDF5} ${OUT_BIG_FILES_HDF5} ${OUT_RESAMPLED_FILE} ${OUT_SHUFFLED_1_FILE} &&
python ${FILES[5]} ${OUT_BIG_FILES_HDF5} ${OUT_CHUNK_FILES_HDF5} ${OUT_SHUFFLED_1_FILE} ${TASK_TYPE}



