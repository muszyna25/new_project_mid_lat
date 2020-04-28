#!/bin/bash

PWD="$(pwd)"
echo "$PWD"

FILES=()
yourfilenames=`ls *.py`

for f in $yourfilenames
do
  FILES+=("$f")
done

echo "${FILES[*]}"

VAR='z'

ALL_ABOVE_AVG="all_above_avg"

#1.py
L_BIN_MASKS="ETHZ_data_list.txt"
FRAC_AREA=1.0
#OUT_LABELS_PATH="/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/generated_data_full_analysis/${VAR}/labels/$FRAC_AREA"
OUT_LABELS_PATH="/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/generated_data_full_analysis/${ALL_ABOVE_AVG}/${VAR}/labels/$FRAC_AREA"

mkdir -p ${OUT_LABELS_PATH}

#2.py
L_ERA_FILES="ERA_data_list.txt"
OUT_RAW_DATA_PATH="/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/generated_data_full_analysis/${ALL_ABOVE_AVG}/${VAR}/raw_data/$VAR"

mkdir -p ${OUT_RAW_DATA_PATH}

#3.py
BIG_FILE_NAME_HDF5="$VAR.h5"
L_REGION_FILES="bin_mask_regions_list.txt"
OUT_BIG_FILES_HDF5="/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/generated_data_full_analysis/${ALL_ABOVE_AVG}/${VAR}/big_hdf5_files"

mkdir -p ${OUT_BIG_FILES_HDF5}

#4.py
OUT_SHUFFLED_0_FILE="${VAR}_shuffled_0.h5"

#5.py
OUT_RESAMPLED_FILE="${VAR}_resampled.h5"

#4.py
OUT_SHUFFLED_1_FILE="${VAR}_shuffled_1.h5"

#6.py
# nothing --

#7.py
# nothing --

#8.py
OUT_CHUNK_FILES_HDF5="${OUT_BIG_FILES_HDF5}/chunk_files"

mkdir -p ${OUT_CHUNK_FILES_HDF5}

python ${FILES[0]} ${L_BIN_MASKS} ${FRAC_AREA} ${OUT_LABELS_PATH} && 
python ${FILES[1]} ${L_ERA_FILES} ${VAR} ${OUT_RAW_DATA_PATH} &&
python ${FILES[2]} ${L_REGION_FILES} ${BIG_FILE_NAME_HDF5} ${OUT_LABELS_PATH} ${OUT_RAW_DATA_PATH} ${OUT_BIG_FILES_HDF5} &&
python ${FILES[3]} ${OUT_BIG_FILES_HDF5} ${OUT_BIG_FILES_HDF5} ${BIG_FILE_NAME_HDF5} ${OUT_SHUFFLED_0_FILE} &&
python ${FILES[4]} ${OUT_BIG_FILES_HDF5} ${OUT_BIG_FILES_HDF5} ${OUT_SHUFFLED_0_FILE} ${OUT_RESAMPLED_FILE} &&
python ${FILES[3]} ${OUT_BIG_FILES_HDF5} ${OUT_BIG_FILES_HDF5} ${OUT_RESAMPLED_FILE} ${OUT_SHUFFLED_1_FILE} &&
python ${FILES[5]} ${OUT_BIG_FILES_HDF5} ${OUT_BIG_FILES_HDF5} ${OUT_SHUFFLED_1_FILE} &&
python ${FILES[6]} ${OUT_BIG_FILES_HDF5} ${OUT_BIG_FILES_HDF5} ${VAR} &&
python ${FILES[7]} ${OUT_BIG_FILES_HDF5} ${OUT_CHUNK_FILES_HDF5} ${VAR}



