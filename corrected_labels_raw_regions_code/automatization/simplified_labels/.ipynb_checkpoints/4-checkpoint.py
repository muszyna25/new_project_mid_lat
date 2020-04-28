#!/usr/bin/env python

import sys 
import os
import numpy as np
import operator
import h5py
from sklearn.utils import resample
from random import shuffle

def read_data_hdf5(fn):
    sys.stdout.flush()
    
    with h5py.File(fn, 'r') as hf:
        keys = list(hf.keys()) # 'NA', etc.
        print(hf['X'],keys)
        X=hf['X'][:]
        Y=hf['Y'][:]
        print(' done',X.shape, Y.shape)
        print('[+] done...!')
    
    return X, Y

### Save to hdf
def save_to_hdf5_format(X, Y, fname, out_path):

    print('Data shapes: ', X.shape, Y.shape)
    S = X
    L = Y
    print('Selected data shapes: ', S.shape, L.shape)
    fn = out_path + '/' + fname
    print('Create %s' %fn)
    hf = h5py.File(fn, 'w')
    hf.create_dataset('X', data=S)
    hf.create_dataset('Y', data=L)
    hf.close()
    print('[+] done...!')

#### MAIN ####

if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    input_fname = sys.argv[3]
    output_fname = sys.argv[4]
    
    print(sys.argv[0], input_path, output_path, input_fname, output_fname)
    
    #input_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    #output_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    
    #fname = 'resampled_pv_data.h5'

    X, Y = read_data_hdf5(input_path + '/' + input_fname)

    print('Shuffle data...')
    
    ind_list = [i for i in range(X.shape[0])]
    shuffle(ind_list)
    D = X[ind_list,:,:,:]
    L = Y[ind_list,]
    
    save_to_hdf5_format(D, L, output_fname, output_path)
    
    print(sys.argv[0], "[+++] DONE!")

