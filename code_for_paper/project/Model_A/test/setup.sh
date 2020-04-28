# Example environment setup script for jupyter deep learning on Cori

# Using our conda environment which has TF, keras, horovod, IPyParallel, etc.
#. /usr/common/software/python/3.6-anaconda-4.4/etc/profile.d/conda.sh
module load tensorflow/intel-1.13.1-py36
#conda activate /global/common/software/dasrepo/JupyterDL
#module load tensorflow/intel-1.12.0-py36
#conda activate /global/cscratch1/sd/sfarrell/conda/isc-ihpc

# Set some threading environment variables
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export NUM_INTER_THREADS=2
export NUM_INTRA_THREADS=32
export OMP_NUM_THREADS=$NUM_INTRA_THREADS
