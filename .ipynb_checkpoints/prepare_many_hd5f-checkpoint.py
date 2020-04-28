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

def read_data_hdf5(fn):
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
    fn = out_path + '/' + str(fname) + '.h5'
    print('Create %s' %fn)
    hf = h5py.File(fn, 'w')
    hf.create_dataset('X', data=S)
    hf.create_dataset('Y', data=L)
    hf.close()
    print('[+] done...!')

#### MAIN ####

if __name__ == "__main__":

    input_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/extracted_raw_data_train_val_test/pv/balanced_data/train.h5'
    out_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/time_test_data'
    
    X, Y = read_data_hdf5(input_path)
    
    for i in range(X.shape[0]):
        save_to_hdf5_format(X[i,:,:,:], Y[i], i, out_path)