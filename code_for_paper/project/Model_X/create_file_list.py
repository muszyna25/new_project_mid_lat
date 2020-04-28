import numpy as np
import glob
import os
import subprocess

#os.system('ls -1 *.py | xargs readlink -f >> file_list.txt')
#fd = open('file_list.txt', 'r')
#lf = fd.read()

input_dir = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/data_classification_regression/data_for_paper/all_data_v2/source_c/'

lf_train = [os.path.basename(x) for x in glob.glob(input_dir + '*train*.h5')]
lfs_train = sorted(lf_train,  key = lambda x: (int(x.split('_')[0]), int(x.split('_')[2])))

lf_val = [os.path.basename(x) for x in glob.glob(input_dir + '*val*.h5')]
lfs_val = sorted(lf_val,  key = lambda x: (int(x.split('_')[0]), int(x.split('_')[2])))

lf_test = [os.path.basename(x) for x in glob.glob(input_dir + '*test*.h5')]
lfs_test = sorted(lf_test,  key = lambda x: (int(x.split('_')[0]), int(x.split('_')[2])))

lf = [lfs_train, lfs_val, lfs_test]

print(lf)

with open('file_list.txt', 'w') as filehandle:
    for listitem in lf:
        for item in listitem:
            filehandle.write('%s\n' % item)

