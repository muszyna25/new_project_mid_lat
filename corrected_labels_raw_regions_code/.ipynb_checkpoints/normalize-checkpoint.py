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
from sklearn.model_selection import train_test_split

#####################################
def read_data_hdf5(fn):
    print(fn)
    hf = h5py.File(fn, 'r')
    keys = list(hf.keys()) # 'NA', etc.
    print(keys)
    X = hf['X'][:]
    Y = hf['Y'][:]
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

###################### MAIN #####################################

if __name__ == "__main__":

    input_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    output_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    
    fnames = ['train.h5', 'val.h5', 'test.h5']
    
    X, Y = read_data_hdf5(input_path + '/' + fnames[0])
    
    print('X, Y', X.shape, Y.shape)
    
    n_lvs = 8
    
    mean_p_lv = []
    for i in range(0,n_lvs):
        mean_p_lv.append([np.mean(x[:,:,i]) for x in X])

    print('Means for each level: ', len(mean_p_lv))
    
    global_p_lv_means = []
    global_p_lv_means = [np.mean(m) for m in mean_p_lv]
    
    print('Means of means for each level: ', global_p_lv_means)
    
    X_n = np.empty((X.shape[0], X.shape[1], X.shape[2], X.shape[3])) 
    
    for i in range(0,n_lvs):
        X_n[:,:,:,i] = X[:,:,:,i] - global_p_lv_means[i]
       
    print('Shape check: ', X_n.shape)
    #print(X)
    #print(X_n)
    
    save_to_hdf5_format(X_n, Y, 'norm_train.h5', output_path)
    
    ### Normalize val and train
    
    X_v, Y_v = read_data_hdf5(input_path + '/' + fnames[1])
    print('X, Y', X_v.shape, Y_v.shape)
    
    X_val = np.empty((X_v.shape[0], X_v.shape[1], X_v.shape[2], X_v.shape[3])) 
    
    for i in range(0,n_lvs):
        X_val[:,:,:,i] = X_v[:,:,:,i] - global_p_lv_means[i]
    
    print('Shape check: ', X_val.shape)
    
    save_to_hdf5_format(X_val, Y_v, 'norm_val.h5', output_path)
    
    X_t, Y_t = read_data_hdf5(input_path + '/' + fnames[2])
    print('X, Y', X_t.shape, Y_t.shape)
    
    X_test = np.empty((X_t.shape[0], X_t.shape[1], X_t.shape[2], X_t.shape[3])) 
    
    for i in range(0,n_lvs):
        X_test[:,:,:,i] = X_t[:,:,:,i] - global_p_lv_means[i]
    
    print('Shape check: ', X_test.shape)
    
    save_to_hdf5_format(X_test, Y_t, 'norm_test.h5', output_path)
    