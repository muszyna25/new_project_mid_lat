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
def read_netcdf_file(fname, varname): #Variables names: e.g., 'lon', 'lat', 'prw'
    print('read_netcdf_file', fname)
    fh = Dataset(fname, mode='r')
    var_netcdf = fh.variables[varname][:] #Retrieves a given variable by name.
    fh.close()
    return var_netcdf

#####################################
def read_file_list(name): 
    fn_list = []
    with open(name, 'r') as filehandle:
        for line in filehandle:
            fn = line[:-1]
            fn_list.append(fn)
    return fn_list

#####################################
def get_largest_blob_size_in_region(reg_img):

    blobs, n_blobs = measure.label(reg_img, neighbors=4, background=0, return_num=True)
    #print('#blobs ', n_blobs)

    d_blobs = dict(zip(range(1,n_blobs+1), np.zeros(n_blobs, int)))

    for x in range(0, blobs.shape[0]):
        for y in range(0, blobs.shape[1]):
            val = blobs[x][y]
            if val != 0:
                d_blobs[val] += 1
    
    #Get the largest blob size in the region
    if not d_blobs:
        largest_blob = 0
    else:
        largest_blob = max(d_blobs.items(), key=operator.itemgetter(1))[1]
    #print(d_blobs)
    #print('lgest blob:', largest_blob)
    return largest_blob

#####################################
def plot_imgs(img, reg_img):
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(reg_img)
    plt.show()

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

        #print('label:', labels)

#####################################
def save_to_hdf5_format(D, fname, out_path):

    year = fname[3:7]
    for dk in D.keys():
        fn = out_path + '/' + year + '_' + dk + '.h5'
        print('Create %s' %fn)
        hf = h5py.File(fn, 'w')
        keys = list(D[dk].keys()) # 'NA', etc.

        for k in keys:
            #print('dic:', D[dk][k]) # 'X' or 'Y'
            _data = D[dk][k]
            if k == 'Y':
                _data = np.reshape(np.array(_data), (np.array(_data).shape[0],1))
            hf.create_dataset(k, data=_data)

        hf.close()
    print('[+] done...!')

################################################################################################
#   INPUT:
#       fn_list -- text file with list of ETHZ nc files 
#       dir_path -- path location of ETHZ nc datafiles
#
#   OUTPUT:
#       output -- text file with bloc sizes 
################################################################################################

if __name__ == "__main__":

    fn_list = sys.argv[1]
    dir_path = sys.argv[2]
    df_list = read_file_list(fn_list)

    file_info = []
    
    debug = True
    l_blob_sizes = []

    #Latitudes cut-offs
    #N_lat_st = 15
    #N_lat_nd = 75 
    #S_lat_st = 105 
    #S_lat_nd = 165
    
    #d_reg_loc = {'NA': [N_lat_st, N_lat_nd, 90, 210], 'NC': [N_lat_st, N_lat_nd, 210, 330], 'NP': [N_lat_st, N_lat_nd, (0,90), (330,360)],
    #            'SA': [S_lat_st, S_lat_nd, 80, 200], 'SI': [S_lat_st, S_lat_nd, 200, 320], 'SP': [S_lat_st, S_lat_nd, (0,80), (320,360)]}

    N_lat_st = 20
    N_lat_nd = 80 
    S_lat_st = 110 
    S_lat_nd = 170
    
    d_reg_loc = {'NC': [N_lat_st, N_lat_nd, 30, 150], 'NP': [N_lat_st, N_lat_nd, 150, 270], 'NA': [N_lat_st, N_lat_nd, (0,30), (270,360)],
                'SI': [S_lat_st, S_lat_nd, 20, 140], 'SP': [S_lat_st, S_lat_nd, 140, 260], 'SA': [S_lat_st, S_lat_nd, (0,20), (260,360)]}


    d_reg_bin_masks = {'NA': {'X': [], 'Y': []}, 'NC': {'X': [], 'Y': []}, 'NP': {'X': [], 'Y': []}, 
            'SA': {'X': [], 'Y': []}, 'SI': {'X': [], 'Y': []}, 'SP': {'X': [], 'Y': []}}
    
    out_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/extracted_raw_data/v'
    #out_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/raw_data_level_500_regions'

    for i in range(0, len(df_list)):
    #for i in range(1, len(df_list)):
        f = df_list[i]
        print(f)

        #Read time variable
        time_var = read_netcdf_file(dir_path + f, 'time')
        print(time_var.shape[0])

        #Read flag variable (binary mask)
        f_var = read_netcdf_file(dir_path + f, 'v')
        print(f_var.shape)
        
        extract_regions(f_var, d_reg_loc, d_reg_bin_masks)
        #print('Region X size: ', d_reg_bin_masks['SI']['X'].shape)
        
        save_to_hdf5_format(d_reg_bin_masks, f, out_path)
        #l = 0
        #plot_imgs(f_var[200, l, :, :], d_reg_bin_masks['NC']['X'][200])
        #plot_imgs(f_var[200, l, :, :], d_reg_bin_masks['NP']['X'][200])
        #plot_imgs(f_var[200, l, :, :], d_reg_bin_masks['NA']['X'][200])
        #plot_imgs(f_var[200, l, :, :], d_reg_bin_masks['SI']['X'][200])
        #plot_imgs(f_var[200, l, :, :], d_reg_bin_masks['SP']['X'][200])
        #plot_imgs(f_var[200, l, :, :], d_reg_bin_masks['SA']['X'][200])



