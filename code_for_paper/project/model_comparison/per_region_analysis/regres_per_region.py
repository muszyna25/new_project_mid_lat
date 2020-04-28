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
from statistics import mean
from numpy.linalg import norm


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

#####################################
def lins_ccc(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)
    t = y_true.mean()
    p = y_pred.mean()
    St = y_true.var()
    Sp = y_pred.var()
    Spt = np.mean((y_true - t) * (y_pred - p)) 
    return 2 * Spt / (St + Sp + (t - p)**2)

#####################################
def lins_ccc_v2(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)

    #CCC = np.empty((0,1))
    CCC = []

    for i in range(0,3):
        t = y_true[:,i].mean()
        p = y_pred[:,i].mean()
        St = y_true[:,i].var()
        Sp = y_pred[:,i].var()
        Spt = np.mean((y_true[:,i] - t) * (y_pred[:,i] - p)) 
        ccc = 2 * Spt / (St + Sp + (t - p)**2)
        CCC.append(ccc)

    return mean(CCC)

#####################################
def mpe_error(y_t, y_p):
    print(y_t.shape)
    print(y_p.shape)

    #mpe = np.sum(norm(1+y_t-1+y_p)/norm(1+y_t))/y_t.shape[0]
    v1 = np.ones((y_t.shape[0],1))*10
    #mpe = np.sum(norm(y_t-y_p)/norm(v1+y_t))/y_t.shape[0]
    mpe = 100*(np.sum((y_t-y_p)/y_t)/y_t.shape[0])
    print('mpe checkpoint:', mpe.shape)

    return mpe
    
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

f_name = 'loc_reg_model_' + m_type + '.h5'
fd = h5py.File(f_name, 'r')
print(fd)

for i in fd.keys():
    print('Name:', str(i))

    y_t = fd[i]['Y_t']
    y_p = fd[i]['Y_p']

    print(y_t.shape)
    print(y_p.shape)
    print('TRUE:',y_t[:10])
    print('PRED:',y_p[:10])

    #ccc = lins_ccc(np.array(y_t), np.array(y_p))
    ccc = lins_ccc_v2(np.array(y_t), np.array(y_p))
    print('CCC:', ccc)

    #mpe_xy = mpe_error(np.array(y_t[:,:2]), np.array(y_p[:,:2]))
    MPE = []
    for j in range(0,3):
        mpe_xy = mpe_error(np.array(y_t[:,j]), np.array(y_p[:,j]))
        print('MPE%:', mpe_xy)
        MPE.append(mpe_xy)
        
    MPE.append(ccc)
    res = np.array(MPE)
    #print(res,res.shape)
    np.savetxt(m_type + '_' + str(i) + '_regres_mpe_xyr_ccc.csv', res, fmt='%.5f')

    #mpe_R = mpe_error(np.array(y_t[:,-1]), np.array(y_p[:,-1]))


'''
    ccc = lins_ccc(np.array(y_t[:,0]), np.array(y_p[:,0]))
    print(ccc)

    ccc = lins_ccc(np.array(y_t[:,1]), np.array(y_p[:,1]))
    print(ccc)

    ccc = lins_ccc(np.array(y_t[:,2]), np.array(y_p[:,2]))
    print(ccc)
'''

    #report = classification_report(y_t, y_p, output_dict=True)
    #print(report)
    #print('ACC: ', accuracy_score(y_t, y_p))

    #df = pandas.DataFrame(report).transpose()
    #df.to_csv(r'report_' + i + '_' + m_type + '.csv')

fd.close()




