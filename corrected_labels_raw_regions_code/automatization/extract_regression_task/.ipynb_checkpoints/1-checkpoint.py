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
import math

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
def get_largest_blob_size_in_region(reg_img, x):

    blobs, n_blobs = measure.label(reg_img, neighbors=4, background=0, return_num=True)
    #print('#blobs ', n_blobs)

    d_blobs = dict(zip(range(1,n_blobs+1), np.zeros(n_blobs, int)))

    # Count number of pixels for each blob label.
    for x in range(0, blobs.shape[0]):
        for y in range(0, blobs.shape[1]):
            val = blobs[x][y]
            if val != 0:
                d_blobs[val] += 1
    
    # Sum all blobs' pixels in binary mask.
    blob_s = 0
    blob_s = (reg_img == 1).sum()
    
    if blob_s == 0:
        return blob_s, [], -2
    
    # Get the largest blob label in the mask.
    if not d_blobs:
        largest_blob = 0
    else:
        largest_blob = max(d_blobs.items(), key=operator.itemgetter(1))[0]
    #print('lgest blob:', largest_blob)
    
    # Remove labels of other smaller blobs in the mask to keep only the largest one.
    for x in range(0, blobs.shape[0]):
        for y in range(0, blobs.shape[1]):
            val = blobs[x][y]
            if val != largest_blob:
                blobs[x][y]=0
                
    # Calculate the mass center and radius of the largest blob in the mask.
    C = []
    radius = 0
    if blob_s != 0:
        contours, hierarchy = cv2.findContours(cv2.convertScaleAbs(blobs), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        #print('#Contours: ', len(contours))
        cnt = contours[0]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        C = [int(x), int(y)]
        #print('Radius: ', math.ceil(radius), int(x), int(y))
    
    return blob_s, C, math.ceil(radius)

#####################################
def plot_imgs(img, reg_img, C, R):
    
    fig,ax = plt.subplots()
    ax.set_aspect('equal')

    # Show the image
    ax.imshow(reg_img)
    ax.scatter(int(C[0]), int(C[1]),color='r')

    # Now, loop through coord arrays, and create a circle at each x,y pair
    circ = plt.Circle((int(C[0]), int(C[1])),R)
    circ.set_facecolor('None')
    circ.set_edgecolor('r')
    ax.add_patch(circ)
    # Show the image
    plt.show()
    
######################################
def extract_regions(f_var, d_reg_loc, D, fn):
    
    avg_blob_size = 228
    frac_avg_blob_size = 1.0 * avg_blob_size
    
    tmp = np.array(f_var)
    img = np.array([np.flipud(x) for x in tmp])

    for dk in D.keys():
        keys = list(D[dk].keys())
        #print('dic:', keys,d_reg_loc[dk])
        lats = d_reg_loc[dk][:2] 
        lons = d_reg_loc[dk][2:]
    
        #Concatenate
        if dk == 'NA' or dk == 'SA':
            reg_img_W = img[:,lats[0]:lats[1], lons[0][0]:lons[0][1]]
            reg_img_E = img[:,lats[0]:lats[1], lons[1][0]:lons[1][1]]
            print('Two regions: ', reg_img_W.shape, reg_img_E.shape)
            reg_img = np.concatenate((reg_img_E, reg_img_W), axis=-1)
            #reg_img = np.concatenate((reg_img_W, reg_img_E), axis=-1)
        else:
            reg_img = img[:,lats[0]:lats[1], lons[0]:lons[1]]

        #plot_imgs(np.flipud(img[0]), np.flipud(reg_img[0]))

        #print(dk)
        print('All data:', reg_img.shape)
        D[dk][keys[0]]=reg_img

        #Get size of the largest blob in the region
        #print('Frac of ave size: ', frac_avg_blob_size)
        #labels = [1 if get_largest_blob_size_in_region(reg_img[x], x) >= frac_avg_blob_size else 0 for x in range(0, reg_img.shape[0])]
        
        labels = []
        reg_vars = []
        aux_info = []
        for x in range(0, reg_img.shape[0]):
            bs, C, R = get_largest_blob_size_in_region(reg_img[x], x)
            
            if bs >= frac_avg_blob_size:
                labels.append(1)
                reg_vars.append(np.array([C[0],C[1],R]))
                aux_info.append(np.array([fn, x, dk], dtype='S'))
                
                # Display extracted mass center and radius.
                #print('C', C)
                #c_ind = 0
                #plot_imgs(img[c_ind], reg_img[c_ind], C, R)
            elif bs > 0:
                labels.append(-1)
                reg_vars.append(np.array([0,0,0])) # Default values.
                aux_info.append(np.array([fn, x, dk], dtype='S'))
            else:
                labels.append(0)
                reg_vars.append(np.array([0,0,0])) # Default values.
                aux_info.append(np.array([fn, x, dk], dtype='S'))
        
        D[dk][keys[1]]=labels
        D[dk][keys[2]]=reg_vars
        D[dk][keys[3]]=aux_info
        
        #print('label: ', labels[:10])
        #print('vars: ', reg_vars[0].shape)

#####################################
def save_to_hdf5_format(D, fname, out_path):

    year = fname[6:10]
    for dk in D.keys():
        fn = out_path + '/' + year + '_' + dk + '.h5'
        print('Create %s' %fn)
        hf = h5py.File(fn, 'w')
        keys = list(D[dk].keys()) # 'NA', etc.

        for k in keys:
            _data = D[dk][k]
            if k == 'Y':
                _data = np.reshape(np.array(_data), (np.array(_data).shape[0],1))
                hf.create_dataset(k, data=_data)
            if k == 'Z':
                _data = np.reshape(np.array(_data), (np.array(_data).shape[0],3))
                hf.create_dataset(k, data=_data)
            if k == 'A':
                _data = np.reshape(np.array(_data), (np.array(_data).shape[0],3))
                hf.create_dataset(k, data=_data)
            if k == 'X':
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
    out_path = sys.argv[2]
    
    print(sys.argv[0], fn_list, out_path)
    
    dir_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/dataset_1980-1998-2000-2016/bin_masks/shifted_bin_masks/'
    df_list = read_file_list(fn_list)
    
    N_lat_st = 20
    N_lat_nd = 80 
    S_lat_st = 110 
    S_lat_nd = 170
    
    d_reg_loc = {'NC': [N_lat_st, N_lat_nd, 30, 150], 'NP': [N_lat_st, N_lat_nd, 150, 270], 'NA': [N_lat_st, N_lat_nd, (0,30), (270,360)],
                 'SI': [S_lat_st, S_lat_nd, 20, 140], 'SP': [S_lat_st, S_lat_nd, 140, 260], 'SA': [S_lat_st, S_lat_nd, (0,20), (260,360)]}
    
    d_reg_bin_masks = {'NC': {'X': [], 'Y': [], 'Z': [], 'A': []}, 'NP': {'X': [], 'Y': [], 'Z': [], 'A': []}, 'NA': {'X': [], 'Y': [], 'Z': [], 'A': []}, 
                       'SI': {'X': [], 'Y': [], 'Z': [], 'A': []}, 'SP': {'X': [], 'Y': [], 'Z': [], 'A': []}, 'SA': {'X': [], 'Y': [], 'Z': [], 'A': []}}
    
    for i in range(len(df_list)):
        f = df_list[i]
        print(f)
    
        #Read time variablee
        time_var = read_netcdf_file(dir_path + f, 'time')
        print(time_var.shape[0])

        #Read flag variable (binary mask)
        f_var = read_netcdf_file(dir_path + f, 'FLAG')
        print(f_var.shape)
        
        extract_regions(f_var, d_reg_loc, d_reg_bin_masks, f)
        print(d_reg_bin_masks)
        
        save_to_hdf5_format(d_reg_bin_masks, f, out_path)
    
    print(sys.argv[0], "[+++] DONE!")

