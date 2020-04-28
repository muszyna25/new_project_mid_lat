from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os

# External imports
import numpy as np
import tensorflow.python.keras
import matplotlib.pyplot as plt
import glob
import random
import csv
import sys
from random import shuffle
from tensorflow.keras.models import load_model
import pandas as pd
from scipy.stats import wilcoxon
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table
import h5py
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas


#####################################
def read_data_hdf5(fn):

    with h5py.File(fn, 'r') as hf:
        keys = list(hf.keys())
        print(hf['X'],keys)
        #X=hf['X'][:]
        #Y=hf['Y'][:]
        #Z=hf['Z'][:]
        A=hf['A'][:]
        #print(' done',X.shape, Y.shape, Z.shape, A.shape)
        print(' done', A.shape)
        print('[+] done...!')

    #return X, Y, Z, A
    return A

#####################################
def get_train_val_names(fname):

    lfs = []
    with open(fname, 'r') as f:
        lfs = f.read().splitlines()

    return lfs

#####################################
def read_data_from_csv(fname):
    #input_file = csv.DictReader(open(fname))
    with open(fname) as csvfile:
        input_file = csv.reader(csvfile, delimiter=' ')
        #next(input_file, None)
        my_list = []
        for row in input_file:
            my_list.append(row)

    return np.array(my_list)

################## MAIN ##################

data_true= read_data_from_csv('true.csv')
data_pred= read_data_from_csv('prediction.csv')

print(data_true.shape, data_pred.shape)


fig, axs = plt.subplots(3, 1, figsize=(18, 8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.5, wspace=.001)
axs = axs.ravel()

axs[0].plot(range(0,data_true[:,0].shape[0]), data_true[:,0], label='true')
axs[0].plot(range(0,data_pred[:,0].shape[0]), data_pred[:,0], label='pred')
axs[0].set_title('first param')
axs[0].legend()


axs[1].plot(range(0,data_true[:,1].shape[0]), data_true[:,1], label='true')
axs[1].plot(range(0,data_pred[:,1].shape[0]), data_pred[:,1], label='pred')
axs[1].set_title('second param')
axs[1].legend()

axs[2].plot(range(0,data_true[:,2].shape[0]), data_true[:,2], label='true')
axs[2].plot(range(0,data_pred[:,2].shape[0]), data_pred[:,2], label='pred')
axs[2].set_title('third param')
axs[2].legend()
#n_bins = np.linspace(0., 120., 25)
#n_bins = np.linspace(-1.0, 1.0, 30)

#axs[0].hist(Z[:,0], bins=n_bins)
#axs[0].set_yscale('log')
#axs[0].set_title('Mass center: x')
#axs[1].hist(Z[:,1], bins=n_bins)
#axs[1].set_yscale('log')
#axs[1].set_title('Mass center: y')
#axs[2].hist(Z[:,2], bins=n_bins)
#axs[2].set_yscale('log')
#axs[2].set_title('Radius: r')

#plt.yscale('log', nonposy='clip')

plt.show()
   

