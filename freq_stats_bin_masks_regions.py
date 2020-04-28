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
def plot_histogram(D, year, idx):
    fig = plt.figure()
    plt.bar(D.keys(), D.values(), color='g', log=True)
    l_vals = list(D.values())
    plt.suptitle('Year: ' + str(year) + ' ' + 'Region: ' +  str(idx) + '\n' +  'AB: ' + str(l_vals[0]) + ' ' + 'non-AB: ' + str(l_vals[1]) , x=0.517)
    plt.ylim((pow(10,0),pow(10,4)))
    plt.savefig('region_' + str(year) + '_' + str(idx) + '.png')
    plt.close(fig) # If close then, then it does not display.
    plt.show()

### MAIN ###

fn_list = sys.argv[1]
dir_path = sys.argv[2]

df_list = read_file_list(fn_list)

stats = {'AB': 0, 'non-AB': 0}
for i in range(0, len(df_list)):
    f = df_list[i]
    print(f)
    hf = h5py.File(dir_path + '/' + f, 'r')
    Y = list(hf['Y'].value)
    AB_count = Y.count(1)
    non_AB_count = Y.count(0)

    stats['AB'] += AB_count
    stats['non-AB'] += non_AB_count

    #stats = {'AB': AB_count, 'non-AB': non_AB_count}
    #year = f[0:4]
    #reg = f[5:-3]
    #print(year, reg)
    #plot_histogram(stats, year, reg)
print(stats)



