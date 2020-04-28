#!/bin/bash 

#SBATCH --qos premium
#-SBATCH --qos regular 
#-SBATCH --qos interactive

#SBATCH -N 3
#SBATCH -t 00:29:00
#SBATCH -L SCRATCH

#-SBATCH -C knl
#SBATCH -C haswell

#SBATCH --mail-user=gmuszynski@lbl.gov
#SBATCH --mail-type=BEGIN,END
#SBATCH -J Model_A

### Bash options ###
set -u

BATCH_START_TIME=$(date)
echo "[+] ------START TIME (ST): $BATCH_START_TIME------"

echo "Start cluster..."
module load tensorflow/intel-1.13.1-py36

./startCluster.sh & sleep 60 && module load tensorflow/intel-1.13.1-py36 && python test_ipyparallel.py ${1} ${2} & wait

######### Sequential code ###########
BATCH_END_TIME=$(date)
echo "[+] ------END TIME (ET) $BATCH_END_TIME------"

