# System imports
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

def read_results_from_csv(fname):
    #input_file = csv.DictReader(open(fname))
    with open(fname) as csvfile:
        input_file = csv.reader(csvfile, delimiter=',')
        next(input_file, None)
        my_list = []
        for row in input_file:
            my_list.append(row)
    
    return my_list

##########################
#def get_train_val_names_v2(fname, start_ind, end_ind, n_files):
def get_train_val_names_v2(fname, start_ind, n_files):
    lfs = []
    with open(fname, 'r') as f:
        #lfs = f.readlines() 
        lfs = f.read().splitlines()

    #random.seed(4)
    l_train = lfs[start_ind - 1:start_ind - 1 + n_files]
    #l_val = lfs[end_ind - 1:end_ind - 1 + n_files]
    #print('Original order ', l_train, l_val)
    #shuffle(l_train)
    #shuffle(l_val)

    #return l_train, l_val
    return l_train

#########################
def read_hdf5_file(fname):

    with h5py.File(fname, 'r') as h5f:
        fd = h5f.keys()
        #A=h5f['A'][:]
        #Y=h5f['Y'][:]
        #print(' done',X.shape, Y.shape)
        #print(' done',A.shape)
 
    return fd

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
        input_file = csv.reader(csvfile, delimiter=',')
        #next(input_file, None)
        my_list = []
        for row in input_file:
            my_list.append(row)

    return np.array(my_list)

#####################################
def save_to_hdf5_format(D, fname, out_path):

    #fn = out_path + '/' + fname + '.h5'
    fn = fname + '.h5'
    print('Create %s' %fn)
    hf = h5py.File(fn, 'w')
    # keys = list(D.keys()) # 'NA', etc.

    for dk in D.keys():
        g = hf.create_group(dk)
        for k in D[dk].keys():
            _data = D[dk][k]
            print(dk, k)
            g.create_dataset(k, data=_data)

    hf.close()
    print('[+] done...!')

###### MAIN ##########

out_path = ''

#REG_DATA = {'NC': {'Y_t': [], 'Y_p': []}, 'NP': {'Y_t': [], 'Y_p': []}, 'NA': {'Y_t': [], 'Y_p': []}, 
#        'SI': {'Y_t': [], 'Y_p': []}, 'SP': {'Y_t': [], 'Y_p': []}, 'SA': {'Y_t': [], 'Y_p': []}}

#TIME_DATA = {str(k):{'Y_t': [], 'Y_p': []} for k in range(1980,2017) if k!= 1999} 

#print(TIME_DATA)
#print(REG_DATA)

#path_labels = '../classifier_predict/'

m_type = 'E'
#data = read_data_from_csv(path_labels + 'Model_' + m_type + '.csv') # Check this line for different models

#Y_t = data[:,0]
#Y_p = data[:,1]
#counter = 0

f_name = 'time_model_' + m_type + '.h5'
#f_name = 'reg_model_' + m_type + '.h5'
fd = h5py.File(f_name, 'r')
print(fd)

for i in fd.keys():
    print('Name:', i)

    y_t = fd[i]['Y_t']
    y_p = fd[i]['Y_p']

    print(y_t.shape)
    print(y_p.shape)

    report = classification_report(y_t, y_p, output_dict=True)
    print(report)
    print('ACC: ', accuracy_score(y_t, y_p))

    df = pandas.DataFrame(report).transpose()
    df.to_csv(r'report_' + i + '_' + m_type + '.csv')

fd.close()






'''
    #for j in lf_test[0:1]:
    for j in lf_test:
        X, Y = read_hdf5_file(input_dir + j)
        print(X.shape)
        X_all = np.concatenate([X_all,X])
        Y_all = np.concatenate([Y_all,Y])

    print('X_all: ', X_all.shape)
    num_zeros = (Y_all == 0).sum()
    num_ones = (Y_all == 1).sum()
    print('Y ', num_zeros, num_ones)

    scores = model.evaluate(X_all, Y_all, verbose=0)
    print('scores: ', scores)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    np.savetxt('classifier_predict/' + 'loss_acc_' + i[0] + '.csv', scores, fmt='%.5f')

    y = model.predict(X_all,batch_size=256)
    print(y)
    p_new = np.array([1.0 - i for i in y])
    y_new = np.concatenate([y, p_new], axis=1)
    y_pred = y_new.argmax(axis=-1)
    y_pred_new = [1 if i == 0 else 0 for i in y_pred]
    print(y_pred_new[29:33], Y_all[29:33].flatten())

    report = classification_report(Y_all, y_pred_new, output_dict=True)
    print(report)
    print('ACC: ', accuracy_score(Y_all, y_pred_new))
    #np.savetxt('classifier_predict/' + 'report_' + i[0] + '.csv', report)
    df = pandas.DataFrame(report).transpose()
    df.to_csv(r'classifier_predict/' + 'report_' + i[0] + '.csv')

    # true, pred 
    dat = np.array([Y_all.flatten(), y_pred_new])
    dat = dat.T
    np.savetxt('classifier_predict/' + i[0] + '.csv', dat, fmt='%i,%i')

'''
