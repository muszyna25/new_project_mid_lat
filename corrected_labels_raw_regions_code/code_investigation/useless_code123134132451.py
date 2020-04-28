#!/usr/bin/env python

import sys 
import os
import numpy as np
import operator
import h5py
from sklearn.utils import resample
#from sklearn.utils import shuffle
from random import shuffle

def read_data_hdf5(fn):
    sys.stdout.flush()
    
    with h5py.File(fn, 'r') as hf:
        keys = list(hf.keys()) # 'NA', etc.
        print(hf['X'],keys)
        X=hf['X'][:]
        Y=hf['Y'][:]
        print(' done',X.shape, Y.shape)
        print('[+] done...!')
    
    return X, Y
    
'''
    print(fn)
    hf = h5py.File(fn, 'r')
    keys = list(hf.keys()) # 'NA', etc.
    print(hf['X'],keys)
    X = hf['X'][:]
    Y = hf['Y'][:]
    #X = hf['X'][:n]
    #Y = hf['Y'][:n]
    print('[+] done...!')
    return X, Y
'''

### Save to hdf
def save_to_hdf5_format(X, Y, fname, out_path):

    print('Data shapes: ', X.shape, Y.shape)
    S = X
    L = Y
    print('Selected data shapes: ', S.shape, L.shape)
    fn = out_path + '/' + fname
    print('Create %s' %fn)
    hf = h5py.File(fn, 'w')
    hf.create_dataset('X', data=S)
    hf.create_dataset('Y', data=L)
    hf.close()
    print('[+] done...!')

def segregate(data, lab):
    print('Segregate samples...')
    pos = [i for i in range(0, lab.shape[0]) if lab[i]==1]
    neg = [i for i in range(0, lab.shape[0]) if lab[i]==0]

    print(len(pos), len(neg))
    print(pos, neg)

    print('[+] done...!')
    return neg, pos

#### MAIN ####

if __name__ == "__main__":

    input_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    output_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    
    fname = 'pv_data.h5'
        
    X, Y = read_data_hdf5(input_path + '/' + fname)
        
    pos_samples = np.where(Y==1)[0]
    neg_samples = np.where(Y==0)[0]
    print('POS: ', pos_samples)
    print('NEG: ', neg_samples)
    
    X_rs = resample(neg_samples, replace=False, n_samples=len(pos_samples), random_state=123) # Down-sampling negative class
    print('Resample: ', X_rs, len(X_rs))
    X_neg = X[X_rs]
    X_pos = X[pos_samples]
    print('X_neg, X_pos', X_neg.shape, X_pos.shape)
    
    data = np.concatenate((np.asarray(X_pos),np.asarray(X_neg)), axis=0)
    print(data.shape)
    Y_neg = Y[X_rs]
    Y_pos = Y[pos_samples]
    labels = np.concatenate((np.asarray(Y_pos),np.asarray(Y_neg)), axis=0)
    print(labels.shape)

    print('Original samples size:', Y.shape)
    print('# of postives and negative samples in the original sample set: ', len(neg_samples), len(pos_samples))
    print('Resampled samples size:', labels.shape)

    print('Shuffle data...')
    #D, L = shuffle(data, labels, random_state=123)
    
    ind_list = [i for i in range(data.shape[0])]
    shuffle(ind_list)
    #D = data[ind_list,:,:,:]
    #L = labels[ind_list,]
    
    save_to_hdf5_format(data[ind_list,:,:,:], labels[ind_list,], 'balanced_pv_data.h5', output_path)



    
#1. Shuffle all dataset
#2. Resampling all dataset          
#3. Split train, val, test 
#4. Split train into 10 sets
#5. Normalizaton per cropped image in each channel separately on training only:
    #a) x_i - mean_i
    #v’ = (v-min)/(max-min) * (newmax-newmin) + newmin
    #1. wybierz zbior train                                                           
    #2. dla kazdej wartwy z osobna policz srednia z  srednich dla kazdego obrazka ze zbioru train

    #kroki dla kazdej warstwy osobno:
    #-policz srednia dla kazdego obrazka z zbioru train
    #- policz srednia z srednich dla kazdego obraka z zbioru train

    #3, dla kazdego warstw odemij odpowiedni srednia srednich
    #dla calego zbioru danych (test, train, val)