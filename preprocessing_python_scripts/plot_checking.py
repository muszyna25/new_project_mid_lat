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

### MAIN ###

fn_1 = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/extracted_raw_data/pv/2000_NC.h5' #sys.argv[1]
#fn_0 = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/binary_masks_regions/2000_NC.h5' #sys.argv[2]
fn_0 = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_labels/2000_NC.h5'

hf_0 = h5py.File(fn_0, 'r')
print(hf_0['X'].shape)
keys_0 = list(hf_0.keys()) # 'NA', etc.
print(keys_0)

hf_1 = h5py.File(fn_1, 'r')
print(hf_1['X'].shape)
print(hf_1['Y'].value)
keys_1 = list(hf_1.keys()) # 'NA', etc.
print(keys_1)

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

fig, axs = plt.subplots(3,3, figsize=(18, 8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.5, wspace=.001)

axs = axs.ravel()

ind=0
for i in range(0,8):
    X_1 = hf_1['X'][ind,i,:,:]
    axs[i].imshow(X_1)

X_0 = hf_0['X'][ind,:,:]
Y_0 = hf_0['Y'][ind]
print(Y_0)

axs[8].imshow(X_0)

plt.show()