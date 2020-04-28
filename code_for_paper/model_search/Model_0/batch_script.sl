#!/bin/bash 

#SBATCH --qos regular 
#-SBATCH --qos interactive
#SBATCH -N 10
#SBATCH -t 08:00:00
#SBATCH -L SCRATCH
#SBATCH -C knl
#SBATCH --mail-user=gmuszynski@lbl.gov
#SBATCH --mail-type=BEGIN,END
#SBATCH -J Model_0

### Bash options ###
set -u

BATCH_START_TIME=$(date)
echo "[+] ------START TIME (ST): $BATCH_START_TIME------"

echo "Start cluster..."
./load_module.sh

./startCluster.sh & sleep 60 && module load tensorflow/intel-1.13.1-py36 && python test_ipyparallel.py & wait

#xterm -e ./startCluster.sh & 
#sleep 60 && module load tensorflow/intel-1.13.1-py36 && python test_ipyparallel.py &
#wait

#xterm -e ./startCluster.sh & sleep 60 && python test_ipyparallel.py

######### Sequential code ###########
BATCH_END_TIME=$(date)
echo "[+] ------END TIME (ET) $BATCH_END_TIME------"
