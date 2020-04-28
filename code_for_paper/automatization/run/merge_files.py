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
        Z=hf['Z'][:]
        A=hf['A'][:]
        print(' done',X.shape, Y.shape, Z.shape, A.shape)
        print('[+] done...!')
    
    return X, Y, Z, A

#####################################
def read_data_hdf5_part(fn):
    sys.stdout.flush()
    
    with h5py.File(fn, 'r') as hf:
        keys = list(hf.keys()) # 'NA', etc.
        print(hf['X'],keys)
        X=hf['X'][:]
        #Y=hf['Y'][:]
        #Z=hf['Z'][:]
        #A=hf['A'][:]
        #print(' done',X.shape, Y.shape, Z.shape, A.shape)
        print(' done',X.shape)
        print('[+] done...!')
    
    return X

#####################################
### Save to hdf
def save_to_hdf5_format(X, Y, Z, A, fname, out_path):

    print('Data shapes: ', X.shape, Y.shape, Z.shape, A.shape)

    S = X
    L = Y
    V = Z
    I = A

    print('Selected data shapes: ', S.shape, L.shape, V.shape, I.shape)

    fn = out_path + '/' + fname
    
    print(os.path.exists(fn), V.shape)

    if os.path.exists(fn) == False:
        print('Create %s' %fn)
        hf = h5py.File(fn, 'w')
        hf.create_dataset('X', data=S, maxshape=(None, S.shape[1], S.shape[2], 40))
        #hf.create_dataset('X', data=S, maxshape=(None, S.shape[1], S.shape[2], S.shape[3]))
        hf.create_dataset('Y', data=L, maxshape=(None, 1))
        hf.create_dataset('Z', data=V, maxshape=(None, V.shape[1]))
        hf.create_dataset('A', data=I, maxshape=(None, I.shape[1]))
        hf.close()
    elif os.path.exists(fn) == True:
        print('Update %s' %fn)
        hf = h5py.File(fn, 'a')
        hf['X'].resize((hf['X'].shape[0] + S.shape[0]), axis = 0)
        hf['X'][-S.shape[0]:] = S
        hf['Y'].resize((hf['Y'].shape[0] + L.shape[0]), axis = 0)
        hf['Y'][-L.shape[0]:] = L
        hf['Z'].resize((hf['Z'].shape[0] + V.shape[0]), axis = 0)
        hf['Z'][-V.shape[0]:] = V
        hf['A'].resize((hf['A'].shape[0] + I.shape[0]), axis = 0)
        hf['A'][-I.shape[0]:] = I
        hf.close()
    print('[+] done...!')

################################################################################################
#   INPUT:
#
#   OUTPUT:
#
################################################################################################

if __name__ == "__main__":

    fn_list = sys.argv[1]
    source_A_path = sys.argv[2]
    source_B_path = sys.argv[3]
    out_path = sys.argv[4]
    
    print(sys.argv[0], fn_list, source_A_path, source_B_path, out_path)
    
    df_list = read_file_list(fn_list)

    for i in range(len(df_list)):
        f = df_list[i]
        print(f)

        X1, Y1, Z1, A1 = read_data_hdf5(source_A_path + '/' + f)
        #X2, Y2, Z2, A2 = read_data_hdf5(source_B_path + '/' + f)
        X2 = read_data_hdf5_part(source_B_path + '/' + f)
    
        X_all_vars = np.concatenate((X1,X2), axis=3)

        print('Check shape X_all_vars: ', X_all_vars.shape)
        save_to_hdf5_format(X_all_vars, Y1, Z1, A1, f, out_path)

    print(sys.argv[0], "[+++] DONE!")

    

