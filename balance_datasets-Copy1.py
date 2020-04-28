#!/usr/bin/env python

import sys 
import os
from netCDF4 import Dataset
import numpy as np
import cv2
from skimage import measure
import matplotlib.pyplot as plt
import operator
import h5py
from sklearn.utils import resample
from sklearn.utils import shuffle

def read_data_hdf5(fn, n=100):
    print(fn)
    hf = h5py.File(fn, 'r')
    keys = list(hf.keys()) # 'NA', etc.
    print(hf['X'],keys)
    X = hf['X'].value
    Y = hf['Y'].value
    #X = hf['X'][:n]
    #Y = hf['Y'][:n]
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

def segregate(data, lab):
    print('Segregate samples...')
    pos = [i for i in range(0, lab.shape[0]) if lab[i]==1]
    neg = [i for i in range(0, lab.shape[0]) if lab[i]==0]

    print(len(pos), len(neg))
    print(pos, neg)

    print('[+] done...!')
    return neg, pos

#### MAIN ####

if __name__ == "__main__":

    #input_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/train_val_test'
    #output_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/balanced_train_val_test'
    input_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/extracted_raw_data_train_val_test/t'
    output_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/extracted_raw_data_train_val_test/t/balanced_data'
    
    
    fn_train = 'train.h5'
    fn_val = 'val.h5'
    fn_test = 'test.h5'

    datasets = [fn_train, fn_val, fn_test]

    for ds in datasets:
        X, Y = read_data_hdf5(input_path + '/' + ds)
        neg_samples, pos_samples = segregate(X,Y)
        X_rs = resample(neg_samples, replace=False, n_samples=len(pos_samples), random_state=123) # Down-sampling negative class
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

        print('Shuffle data...')
        D, L = shuffle(data, labels, random_state=111)
        save_to_hdf5_format(D, L, ds, output_path)

        #save_to_hdf5_format(data, labels, ds, output_path)










