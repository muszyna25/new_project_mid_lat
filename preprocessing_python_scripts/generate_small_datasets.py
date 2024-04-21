#!/usr/bin/env python

import sys 
import os
import numpy as np
import operator
import h5py

def read_data_hdf5(fn, n=100):
    print(fn)
    hf = h5py.File(fn, 'r')
    keys = list(hf.keys()) # 'NA', etc.
    print(keys)
    #X = hf['X'].value
    #Y = hf['Y'].value
    X = hf['X'][:n]
    Y = hf['Y'][:n]
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

    input_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/balanced_train_val_test'
    output_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/small_datasets'

    fn_train = 'train.h5'
    fn_val = 'val.h5'
    fn_test = 'test.h5'

    datasets = [fn_train, fn_val, fn_test]
    '''
    for ds in datasets:
        X, Y = read_data_hdf5(input_path + '/' + ds, 1000)
        print(ds,X.shape,Y.shape)
        save_to_hdf5_format(X[:,:,:,0:3], Y, ds, output_path)
    '''
    X, Y = read_data_hdf5(input_path + '/' + datasets[0], 10000)
    for i in range(10):
        print(i,X.shape,Y.shape)
        save_to_hdf5_format(X[:,:,:,0:3], Y, str(i)+'.h5' , output_path)


