#!/usr/bin/env python
import time
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
import glob

def read_data_hdf5(fn):
    #print(fn)
    hf = h5py.File(fn, 'r')
    #keys = list(hf.keys()) # 'NA', etc.
    #print(hf['X'],keys)
    #X = hf['X'].value
    #Y = hf['Y'].value
    X = hf['X'][:]
    Y = hf['Y'][:]
    #print('[+] done...!')
    return X, Y

dir_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/time_test_data/'

lf = glob.glob(dir_path + '*.h5')

start = time.time()

for i in range(19000):
    X, Y = read_data_hdf5(dir_path + str(i) + '.h5')
    data = X
    labels = Y
    
print('load done, elaT=%.1f sec'%(time.time() - start))