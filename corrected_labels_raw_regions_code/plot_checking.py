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
def plot_imgs(img, reg_img):
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(reg_img)
    plt.show()

def read_netcdf_file(fname, varname, ind): #Variables names: e.g., 'lon', 'lat', 'prw'
    print('read_netcdf_file', fname)
    fh = Dataset(fname, mode='r')
    var_netcdf = fh.variables[varname][ind] #Retrieves a given variable by name.
    fh.close()
    return var_netcdf
    
### MAIN ###

fn_1 = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/dataset_1980-1998-2000-2016/bin_masks/shifted_bin_masks/BLOCKS2000.nc'

fn_0 = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_labels/test/all_corrected_labels/2000_NC.h5'

hf_0 = h5py.File(fn_0, 'r')
print(hf_0['X'].shape)
keys_0 = list(hf_0.keys()) # 'NA', etc.
print(keys_0)

c_ind = 12#1140
X_raw = read_netcdf_file(fn_1, 'FLAG', c_ind)

print(hf_0['Y'][c_ind])

fig, axs = plt.subplots(2, 1, figsize=(18, 8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.5, wspace=.001)

axs = axs.ravel()

X = hf_0['X'][:]
axs[0].imshow(X[c_ind])
#axs[1].imshow(X_raw)
#axs[0].imshow(np.flipud(X[0]))
axs[1].imshow(np.flipud(X_raw))
plt.show()

'''
ind=0
for i in range(0,8):
    X_1 = hf_1['X'][ind,i,:,:]
    axs[i].imshow(X_1)

X_0 = hf_0['X'][ind,:,:]
Y_0 = hf_0['Y'][ind]
print(Y_0)

axs[8].imshow(X_0)

plt.show()
'''













'''
ind = 300
for i in range(ind,ind+10):
    X_0 = hf_0['X'][i,:,:]
    Y_0 = hf_0['Y'][i]
    X_1 = hf_1['X'][i,j,:,:]
    #Y_1 = hf_1['Y'] # It doesn't exist yet..
    print(Y_0)
    plot_imgs(X_0,X_1)
'''