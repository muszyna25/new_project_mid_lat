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
def read_netcdf_file(fname, varname, ind): #Variables names: e.g., 'lon', 'lat', 'prw'
    print('read_netcdf_file', fname)
    fh = Dataset(fname, mode='r')
    var_netcdf = fh.variables[varname][ind] #Retrieves a given variable by name.
    fh.close()
    return var_netcdf

#####################################
def extract_regions(f_var, d_reg_loc, D):
    avg_blob_size = 228
    frac_avg_blob_size = 0.30 * avg_blob_size
    img = np.array(f_var)

    for dk in D.keys():
        keys = list(D[dk].keys())
        #print('dic:', keys,d_reg_loc[dk])
        lats = d_reg_loc[dk][:2] 
        lons = d_reg_loc[dk][2:]
    
        #Concatenate NP_E + NP_W
        #if dk == 'NP' or dk == 'SP':
        if dk == 'NA' or dk == 'SA':
            reg_img_W = img[:, :, lats[0]:lats[1], lons[0][0]:lons[0][1]]
            reg_img_E = img[:, :, lats[0]:lats[1], lons[1][0]:lons[1][1]]
            print('Two regions: ', reg_img_W.shape, reg_img_E.shape)
            reg_img = np.concatenate((reg_img_E, reg_img_W), axis=-1)
        else:
            reg_img = img[:, :, lats[0]:lats[1], lons[0]:lons[1]]

        print('All data:', reg_img.shape)
        D[dk][keys[0]]=reg_img

        #Get size of the largest blob in the region
        D[dk][keys[1]]=[]

########### MAIN #######

'''
fn_0='/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/raw_data_level_500_regions/2015_NA.h5'
hf_0 = h5py.File(fn_0, 'r')
print(hf_0['X'].shape)
keys_0 = list(hf_0.keys()) # 'NA', etc.
print(keys_0)
'''

ind=0
fn_1 = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/dataset_1980-1998-2000-2016/climate_mod_output/ERA2000.nc'
X = read_netcdf_file(fn_1, 'pv', ind)
print(X.shape)

N_lat_st = 20
N_lat_nd = 80
d_reg_loc = {'NP': [N_lat_st, N_lat_nd, 150, 270]}
d_reg_bin_masks = {'NP': {'X': [], 'Y': []}}

#extract_regions(X, d_reg_loc, d_reg_bin_masks)

fig, axs = plt.subplots(4,1, figsize=(18, 12), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.5, wspace=.01)

axs = axs.ravel()

pos=0
#for i in range(0,2):
    #X_0=d_reg_bin_masks['NP']['X'][pos,i,:,:]
    
axs[0].imshow(X[0,:,:])
axs[1].imshow(X[1,:,:])
    
ind=0
fn_2 = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/dataset_1980-1998-2000-2016/bin_masks/shifted_bin_masks/BLOCKS2000.nc'
#fn_2 = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/ab_label_data/BLOCKS2000.nc'
X2 = read_netcdf_file(fn_2, 'FLAG', ind)
print(X2.shape)

#X2 = np.flipud(X2)

pos=0
#aa=X2[20:80, 30:150]
axs[2].imshow(X2)
axs[3].imshow(X2[20:80, 270:360])
#axs[3].imshow(aa)
#axs[3].imshow(X2[20:80, 150:270])


#x = range(2,4)
#for i in range(0,2):
#    axs[x[i]].imshow(X[pos,i-2,:,:])

'''
X_0=d_reg_bin_masks['NP']['X'][pos,6,:,:]
axs[0].imshow(X_0, interpolation='lanczos')
X_0=d_reg_bin_masks['NP']['X'][pos,7,:,:]
axs[1].imshow(X_0, interpolation='lanczos')
axs[2].imshow(X[pos,6,:,:])
axs[3].imshow(X[pos,7,:,:])
'''
plt.show()