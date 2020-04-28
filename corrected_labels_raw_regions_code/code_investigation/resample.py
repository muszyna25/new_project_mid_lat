#!/usr/bin/env python

import sys 
import os
import numpy as np
import operator
import h5py
from sklearn.utils import resample

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

    input_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    output_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    
    fname = 'first_shuffle_pv_data.h5'
        
    X, Y = read_data_hdf5(input_path + '/' + fname)
        
    pos_samples = np.where(Y==1)[0]
    neg_samples = np.where(Y==0)[0]
    print('POS: ', pos_samples)
    print('NEG: ', neg_samples)
    
    X_rs = resample(neg_samples, replace=False, n_samples=len(pos_samples), random_state=112) # Down-sampling negative class
    print('Resample: ', X_rs, len(X_rs))
    X_neg = X[X_rs]
    X_pos = X[pos_samples]
    print('X_neg, X_pos', X_neg.shape, X_pos.shape)
    
    data = np.concatenate((np.asarray(X_pos),np.asarray(X_neg)), axis=0)
    print(data.shape)
    Y_neg = Y[X_rs]
    Y_pos = Y[pos_samples]
    labels = np.concatenate((np.asarray(Y_pos),np.asarray(Y_neg)), axis=0)
    print(labels.shape)

    print('Original samples size:', Y.shape)
    print('# of postives and negative samples in the original sample set: ', len(neg_samples), len(pos_samples))
    print('Resampled samples size:', labels.shape)
    
    save_to_hdf5_format(data, labels, 'resampled_pv_data.h5', output_path)