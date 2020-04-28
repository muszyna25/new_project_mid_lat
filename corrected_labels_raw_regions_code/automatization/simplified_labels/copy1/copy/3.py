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

#####################################
def read_file_list(name): 
    fn_list = []
    with open(name, 'r') as filehandle:
        for line in filehandle:
            fn = line[:-1]
            fn_list.append(fn)
    return fn_list

#####################################
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

#####################################
### Save to hdf
def save_to_hdf5_format(X, Y, fname, out_path):

    print('Data shapes: ', X.shape, Y.shape)

    S = X
    L = Y

    print('Selected data shapes: ', S.shape, L.shape)

    fn = out_path + '/' + fname

    if os.path.exists(fn) == False:
        print('Create %s' %fn)
        hf = h5py.File(fn, 'w')
        hf.create_dataset('X', data=S, maxshape=(None, S.shape[1], S.shape[2], S.shape[3]))
        hf.create_dataset('Y', data=L, maxshape=(None, 1))
        hf.close()
    elif os.path.exists(fn) == True:
        print('Update %s' %fn)
        hf = h5py.File(fn, 'a')
        hf['X'].resize((hf['X'].shape[0] + S.shape[0]), axis = 0)
        hf['X'][-S.shape[0]:] = S
        hf['Y'].resize((hf['Y'].shape[0] + L.shape[0]), axis = 0)
        hf['Y'][-L.shape[0]:] = L
        hf.close()
    print('[+] done...!')

###################### MAIN #####################################

if __name__ == "__main__":

    fn_list = sys.argv[1]
    file_name = sys.argv[2]
    dir_path_bin_masks = sys.argv[3]
    dir_path_era_raw = sys.argv[4]
    out_path = sys.argv[5]
    
    print(sys.argv[0], fn_list, file_name, dir_path_bin_masks, dir_path_era_raw, out_path)
    
    #dir_path_bin_masks = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_labels/test/all_corrected_labels'
    #dir_path_era_raw = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_raw_data/pv'

    df_list = read_file_list(fn_list)
    
    #out_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'

    #file_name = 'pv_data.h5'
    
    #TRAIN SET
    for i in range(0, len(df_list)):
        f = df_list[i]
        print(f)
    
        X_bm, Y_bm = read_data_hdf5(dir_path_bin_masks + '/' + f)
        X_raw, Y_raw = read_data_hdf5(dir_path_era_raw + '/' + f)

        l_skipped = []
        l_skipped = [x for x in range(0,Y_bm.shape[0]) if Y_bm[x] != -1]
        
        X_s = X_raw[l_skipped, :, :, :]
        Y_s = Y_bm[l_skipped,]
        
        Y = np.reshape(Y_s, (Y_s.shape[0],1))
        X_tmp = np.swapaxes(X_s,1,2)
        X = np.swapaxes(X_tmp,2,3)

        save_to_hdf5_format(X, Y, file_name, out_path)
    
    print(sys.argv[0], "[+++] DONE!")

