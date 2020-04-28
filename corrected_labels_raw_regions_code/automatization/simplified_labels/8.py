#!/usr/bin/env python

import sys 
import os
import numpy as np
import operator
import h5py

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

    #input_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    #output_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv/ready_to_use'
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    var = sys.argv[3]
    
    print(sys.argv[0], input_path, output_path, var)
    
    fn_train = var + '_norm_train.h5'
    fn_val = var + '_norm_val.h5'

    datasets = [fn_train, fn_val]
    new_datasets = ['train', 'val']
    
    n_sets = 10
    counter = 0
    for ds in datasets:
        X, Y = read_data_hdf5(input_path + '/' + ds)
        print(ds)
        n=X.shape[0]
        step = int(n/n_sets)
        counter_1 = 0
        for i in range(0, n, step):
            print(i,X.shape,Y.shape)
            save_to_hdf5_format(X[i:i + step,:,:,:], Y[i:i + step,], new_datasets[counter] + '_' + str(counter_1) + '.h5' , output_path)
            counter_1+=1
        counter+=1

    print(sys.argv[0], "[+++] DONE!")


