#!/bin/bash

####################################################################################
# Launch batch (job) scripts in a loop for a given model (e.g., Model_X)
# Generate and pass start index & end index of files that we pass to each model's replica.
####################################################################################

set -u

MODEL='Model_D' # Change name when launching different---deeper CNN model (i.e., A, B, C, ..., F).
FILE=${MODEL}'.sh'
PAR_1=0
PAR_2=0
c=0
RANGE=181 # Max index---it corresponds to max number of train files in a list of hdf5 data files.
SHIFT=3 # No of hdf5 subfiles for data generator (streamer).
JUMP=200
WAIT_TIME=10 # Waiting time in order to avoid rejections from the SLURM system.

#This loop is going to run one Model_X for different rounds of CV and launch scripts (jobs) to do hyper-parameters tuning of each model replica. 
for ((a=1; a <= ${RANGE}; a=$((a+${SHIFT})))); do

    echo ${c}
    echo ${a}
    PAR_1=${a} #Start index.
    PAR_2=$((a+${JUMP})) #End index.
    echo ${PAR_1} ${PAR_2}

    #It stops at 201, 401. I can pass this to python script and iterate until 200 and 400.
    sbatch --job-name=${MODEL}'_'${c} ${FILE} ${MODEL}'_'${c} ${PAR_1} ${PAR_2} ${SHIFT}
    sleep ${WAIT_TIME}
    
    # JUST FOR TESTING, AND REMOVE THIS IF STATEMENT WHEN RUNNING ALL ROUNDS FOR CV.
    #if [ $a -eq "41" ]; then
    #    echo ${a}
    #    #It stops at 201, 401. I can pass this to python script and iterate until 200 and 400
    #    sbatch --job-name=${MODEL}'_'${c} ${FILE} ${MODEL}'_'${c} ${PAR_1} ${PAR_2} ${SHIFT}
    #    sleep ${WAIT_TIME}
    #fi

    #Count number of rounds of CV.
    c=$((c+1))

done


