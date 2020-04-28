"""
This module contains model and training code for the CNN classifier.
"""

# System
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os

# Big data
import h5py
import numpy as np
import math

# Deep learning
#import keras
import tensorflow.python.keras

#from keras import layers, models, optimizers
from tensorflow.python.keras import layers, models, optimizers

#from keras.models import Sequential
from tensorflow.python.keras.models import Sequential

#from keras import backend as K
from tensorflow.python.keras import backend as K

#from keras.callbacks import Callback
from tensorflow.python.keras.callbacks import Callback

#from keras.utils import Sequence
from tensorflow.python.keras.utils import Sequence

import subprocess
import random
import time
import csv
#import matplotlib.pyplot as plt
import sys
import pandas
from random import shuffle
from sklearn.preprocessing import Normalizer

from sklearn import preprocessing

from sklearn.preprocessing import normalize
#from time import time

#############################################################################
class MyLearningTracker(Callback):
    def __init__(self):
        self.hir=[]   
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr)
        self.hir.append(lr)
        #print(lr)
        
class TimingCallback(Callback):
    """A Keras Callback which records the time of each epoch"""
    def __init__(self):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time.time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time.time() - self.starttime
        self.times.append(epoch_time)
        logs['time'] = epoch_time

#############################################################################
class MyImageGenerator(Sequence):
    # based on https://medium.com/datadriveninvestor/keras-training-on-large-datasets-3e9d9dbc09d4

    #' logic: sequentially open HD5 files form the list, one at a time, close previous'
    def __init__(self, dom, inpPath, fnameL, batch_size, fakeSteps=None, verb=1):
        'Initialization'
        self.dom=dom
        self.verb=verb
        self.batch_size = batch_size
        self.numFiles=len(fnameL)
        self.cnt={'epoch':0,'files':0}
        
        if type(fnameL) is list:
            anyL=[ inpPath+'/'+x for x in fnameL]
        else:
            anyL = [inpPath+'/'+fnameL]
        #print('List of files: ', dom, anyL)
        
        goodL=[]
        for x in anyL:
            if  os.path.exists(x) : goodL.append(x)
        if self.verb:
            print('gen-cnst dom=%s found %d of %d input files, fakeSteps='%(dom,len(goodL),len(anyL)),fakeSteps,self.verb)
        assert(len(goodL)>0)
        shuffle(goodL)
        self.fnameL=goodL
        self.stepTime =[] # history of completed epochs
        self.openH5(0) # prime this generator
        self.__getitem__(0) # to record timestep
        self.numFrames=self.frames_per_file*len(goodL)
        self.fakeSteps=fakeSteps

        if self.verb:
            print('  gen-cnst:%s nFiles=%d nFrames=%d nSteps=%d is set.'%(dom,len(goodL),self.numFrames,self.__len__()))
        self.on_epoch_end()

    #...!...!..................
    def openH5(self,fCnt):        
        assert fCnt < self.numFiles
        inpF=self.fnameL[fCnt]
        self.cnt['files']+=1
        prt= self.verb and self.cnt['files']<2
        #print('read data from hdf5:',inpF)
        h5f = h5py.File(inpF, 'r')
        objD={}
        for x in h5f.keys():
            obj=h5f[x][:]
            if prt and '0' in self.dom: print('readH5 ',x,obj.shape)
            objD[x]=obj
        h5f.close()
        self.finger=(fCnt,0)
        self.data=objD
        if self.cnt['files']==1: # this is the 1st file opened
            self.frames_per_file=obj.shape[0]
    
    #...!...!..................
    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.fakeSteps!=None:
            return self.fakeSteps
        return int(np.floor( self.numFrames/ self.batch_size))

    #...!...!..................
    def __getitem__(self, idx):
        'ignore  requested input index to assure good performance'
        #print('ggIt idx=',idx,'finger(2):',self.finger[:2],self.dom)
        fCnt, pCnt=self.finger
        bs=self.batch_size
        if pCnt+bs > self.frames_per_file:
            fCnt= (fCnt+1)% self.numFiles
            self.openH5(fCnt)
            pCnt=0 # reset pointer

        #X=self.data['X'][pCnt:pCnt+bs]
        #Y=self.data['Y'][pCnt:pCnt+bs]
        #note, AUX is not passed to output
        
        X=self.data['X'][pCnt:pCnt+bs,:,:,:]
        #Y=self.data['Y'][pCnt:pCnt+bs]
        Y=self.data['Z'][pCnt:pCnt+bs]
        
        #for i in range(0,8):
        #    for j in range(0,X.shape[0]):
        #        A=X[j,:,:,i]
        #        X[j,:,:,i]=normalize(A, norm='l2')
        
        pCnt+=bs
        self.finger=(fCnt, pCnt)
        #return (data,labels)
        return (X,Y)

    #...!...!..................
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.cnt['epoch']%10==0 and self.verb:
                print('GEN:%s on_epoch_end cnt:'%self.dom,self.cnt)
        self.cnt['epoch']+=1
        fCnt, pCnt=self.finger
        timeRec=[time.time(),fCnt, self.cnt['epoch']]
        self.stepTime.append(timeRec)
        
#############################################################################
def make_prediction(model, X):
        Yprob = model.predict(X).flatten()
        print('Yprob',Yprob.shape)
        return Yprob
    
#############################################################################
'''
def plot_labeled_scores(Ytrue,Yscore,score_thr=0.5):
        u={0:[],1:[]}
        mAcc={0:0,1:0}
        for ygt,ysc in  zip(Ytrue,Yscore):
            ii=int(ygt[0])
            #print('cc',type(ygt),type(ysc),ii,type(ii))
            u[ii].append(ysc)
            if ysc> score_thr : mAcc[ii]+=1

        mInp={0:len(u[0])+1e-3,1:len(u[1])+1e-3}

        print('Labeled scores found mAcc',mAcc, ' thr=',score_thr)

        bins = np.linspace(0.0, 1., 50)
        txt=''
        txt='TPR=%.2f, '%(mAcc[1]/mInp[1])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(u[1], bins, alpha=0.6,label=txt+'%d POS out of %d'%(mAcc[1],mInp[1]))
        txt='FPR=%.2f, '%(mAcc[0]/mInp[0])
        ax.hist(u[0], bins, alpha=0.5,label=txt+'%d NEG out of %d'%(mAcc[0],mInp[0]))

        ax.axvline(x=score_thr,linewidth=2, color='blue', linestyle='--')

        ax.set(xlabel='predicted score', ylabel='num samples')
        #ax.set_yscale('log')
        ax.grid(True)
        ax.set_title('Labeled scores dom=%s'%(segName))
        ax.legend(loc='upper right', title='score thr > %.2f'%score_thr)
'''     
#############################################################################
def save_results_to_csv(fname, histories):
    keys = histories[0].keys()
    print(keys)
    with open(fname, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(histories)

#############################################################################
def read_results_from_csv(fname):
    input_file = csv.DictReader(open(fname))
    my_list = []
    for row in input_file:
        my_list.append(row)
    return my_list

#############################################################################  
'''
def draw_history(h):
    plt.figure(figsize=(9,4))
    # Loss
    plt.subplot(121)
    plt.plot(h['loss'], label='Training')
    plt.plot(h['val_loss'], label='Validation')
    plt.xlim(xmin=0, xmax=len(h['loss'])+10)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=0)
    plt.grid()
    # Accuracy
    plt.subplot(122)
    plt.plot(h['acc'], label='Training')
    plt.plot(h['val_acc'], label='Validation')
    plt.xlim(xmin=0, xmax=len(h['loss'])+10)
    plt.ylim((0, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc=0)
    plt.grid()
    plt.tight_layout()
'''
#############################################################################
'''
def plot_acc_hist(scores, name):
    plt.figure()
    plt.hist(scores, bins='auto', rwidth=0.3)
    plt.grid()
    plt.xlabel(name);
'''
#############################################################################
def check_early_stopping(done_epochs, n_epochs):
    l_early_stop = []
    for i in range(0, len(done_epochs)):
        if done_epochs[i] < n_epochs:
            l_early_stop.append(1)
        else: 
            l_early_stop.append(0)
    return l_early_stop

#############################################################################
def check_no_params(results):
    l_n_params = []
    for i in range(0, len(results)):
        out = results[i].stdout
        st_pos = out.find('No of parameters:')
        subs_pos = out[st_pos:st_pos + 100].find('\n')
        subs = out[st_pos: st_pos + subs_pos]
        n = int(subs.split(':')[1])
        l_n_params.append(n)
    return l_n_params

#############################################################################        
def get_job_id(cmd):
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception('Failed to run command: %s' % cmd)
    return int(result.stdout.splitlines()[1])

#############################################################################
def get_no_samples(fname, percent):
    np.random.seed(0)
    h5f = h5py.File(fname, 'r')
    X=h5f['X']
    print('Datafile: %s' %fname)
    print('Data shape: %s' %X)
    print('# of samples: %d' %X.shape[0])
    N_s = int(X.shape[0]*percent)
    print('%0.2f of samples: %d' %(percent, N_s))
    if percent != 1:
        I = random.sample(range(X.shape[0]),N_s)
        h5f.close()
        return sorted(I)
    else:
        L = range(0,int(X.shape[0]))
        h5f.close()
        return L

#############################################################################
def save_hyper_params_to_csv(n_hpo_trials, n_convs_layers, conv_sizes, n_fc_layers, fc_sizes, dropout, optimizer, lr, batch_sizes, 
                             n_epochs, repeats_conv, recep_fields, checkpoint_dir, job_id):
    d_hyper_params = dict({'ihp': [], 'n_hpo_trials': [], 'n_convs_layers': [], 'conv_sizes': [], 'n_fc_layers': [], 'fc_sizes': [], 
                  'dropout': [], 'optimizer': [], 'lr': [], 'batch_sizes': [], 'n_epochs': [], 'repeats_conv': [], 'recep_fields': [], 
                  'checkpoint_dir': []})
    l_hyper_params = []
    for ihp in range(n_hpo_trials):
        #print('Hyperparameter trial %i/%d, #conv layers %d, conv %s, #fc layers %d, fc %s, dropout %.4f, opt %s, lr %.4f, batch size %d, #epochs %d, repeats conv layers %s, recep fields %s, checkpoint dir: %s' %
              #(ihp, n_hpo_trials, n_convs_layers, conv_sizes[ihp], n_fc_layers, fc_sizes[ihp], dropout[ihp], optimizer[ihp], lr[ihp], batch_sizes[ihp], n_epochs, repeats_conv, recep_fields, checkpoint_dir))
        d_hyper_params = {'ihp': ihp, 'n_hpo_trials': n_hpo_trials, 'n_convs_layers': n_convs_layers, 'conv_sizes': conv_sizes[ihp], 'n_fc_layers': n_fc_layers, 'fc_sizes': fc_sizes[ihp], 'dropout': dropout[ihp], 'optimizer': optimizer[ihp], 'lr': lr[ihp], 'batch_sizes': batch_sizes[ihp], 'n_epochs': n_epochs, 'repeats_conv': repeats_conv, 'recep_fields': recep_fields, 'checkpoint_dir': checkpoint_dir}
        l_hyper_params.append(d_hyper_params)
    
    keys = list(d_hyper_params.keys())
    print(keys)
    csv_columns = keys
    csv_file = checkpoint_dir + '/hyper_params_' + str(job_id) + '.csv'
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in l_hyper_params:
                writer.writerow(data)
    except IOError:
        print("I/O error") 

#############################################################################
def load_my_data(fname, n):

    #start = time.time()
    #print('load hdf5:',fname, flush=True)
    
    #sys.stdout.flush()
    
    with h5py.File(fname, 'r') as h5f:
        X=h5f['X'][:,:,:,n[0]:n[1]]
        Y=h5f['Y'][:]
        #X=h5f['X'][:]
        #Y=h5f['Y'][:]
        print(' done',X.shape, Y.shape)
 
    #print('load_input_hdf5 done, elaT=%.1f sec'%(time.time() - start))    
    return X, Y

#############################################################################
def read_hdf5_file(fname):

    #print('load hdf5:',fname, flush=True)
    
    #sys.stdout.flush()
    
    with h5py.File(fname, 'r') as h5f:
        X=h5f['X'][:]
        Y=h5f['Y'][:]
        print(' done',X.shape, Y.shape)
 
    return X, Y

#############################################################################
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
#############################################################################
def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
#############################################################################
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#############################################################################
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

#############################################################################
def CC(y_true, y_pred):

    n_y_true = (y_true - K.mean(y_true[:]))
    n_y_pred = (y_pred - K.mean(y_pred[:]))

    top = K.sum(n_y_true[:] * n_y_pred[:])
    bottom = K.sqrt(K.sum(K.pow(n_y_true[:],2))) * K.sqrt(K.sum(K.pow(n_y_pred[:],2)))

    result = top/bottom

    return result

def cc0_for_channel(index):
    def cc0(true,pred):

        #get only the desired class
        true = true[:,index]
        pred = pred[:,index]

        #return dice per class
        return CC(true,pred)
    return cc0

def cc1_for_channel(index):
    def cc1(true,pred):

        #get only the desired class
        true = true[:,index]
        pred = pred[:,index]

        #return dice per class
        return CC(true,pred)
    return cc1

def cc2_for_channel(index):
    def cc2(true,pred):

        #get only the desired class
        true = true[:,index]
        pred = pred[:,index]

        #return dice per class
        return CC(true,pred)
    return cc2

def ccc0_for_channel(index):
    def ccc0(true,pred):

        #get only the desired class
        true = true[:,index]
        pred = pred[:,index]

        #return dice per class
        return CCC(true,pred)
    return ccc0

def ccc1_for_channel(index):
    def ccc1(true,pred):

        #get only the desired class
        true = true[:,index]
        pred = pred[:,index]

        #return dice per class
        return CCC(true,pred)
    return ccc1

def ccc2_for_channel(index):
    def ccc2(true,pred):

        #get only the desired class
        true = true[:,index]
        pred = pred[:,index]

        #return dice per class
        return CCC(true,pred)
    return ccc2

#############################################################################
def CCC(y_true, y_pred):
    # covariance between y_true and y_pred
    n_y_true = y_true - K.mean(y_true[:])
    n_y_pred = y_pred - K.mean(y_pred[:])
    s_xy = K.mean(n_y_true * n_y_pred)

    # means
    x_m = K.mean(y_true)
    y_m = K.mean(y_pred)

    # variances
    s_x_sq = K.mean(K.pow(n_y_true,2))
    s_y_sq = K.mean(K.pow(n_y_pred,2))

    ccc = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)

    return ccc

#############################################################################
def build_model(input_shape,
                conv_sizes=[8, 16, 32], repeats_conv=[(1,2), (2,2), (3,2), (4,2)], recep_field=(3,3),
                fc_sizes=[64, 8],
                dropout=0.5,
                optimizer='Adam', lr=0.001
                ):
    
    #start = time.time()
    
##### MODEL ARCHITECTURE #####    
    # Define the inputs
    inputs = layers.Input(shape=input_shape)
    h = inputs

    for i in range(len(conv_sizes)):
        h = layers.Conv2D(conv_sizes[i], recep_field, input_shape=input_shape, padding='same', activation='relu')(h)
        if repeats_conv != []:
            for j in range(len(repeats_conv)):
                if i == repeats_conv[j][0]:
                    for k in range(repeats_conv[j][1]):
                        h = layers.Conv2D(conv_sizes[i], recep_field, input_shape=input_shape, padding='same', activation='relu')(h)
        h = layers.MaxPooling2D(pool_size=(2,2), padding='same')(h)
    
    #Instantiate an empty model
    #for i in range(len(conv_sizes)):
    #    h = layers.Conv2D(conv_sizes[i], recep_field, input_shape=input_shape, padding='same', activation='relu')(h)
    #    h = layers.MaxPooling2D(pool_size=(2,2), padding='same')(h)
        
    h = layers.Flatten()(h)
    
    #Repeat fc layers and dropout layers in blocks
    for fc in fc_sizes:
        h = layers.Dense(fc, activation='relu')(h)
        h = layers.Dropout(dropout)(h)

    #Last output layer
    #outputs = layers.Dense(1, activation='sigmoid')(h)
    outputs = layers.Dense(3, activation='tanh')(h)

##### MODEL ARCHITECTURE OPTIMIZATION PARAMS #####   

    # Construct the optimizer
    #opt_dict = {'Adam': 'adam', 'Nadam': 'nadam', 'SGD': 'sgd'}
    opt_dict = {'Adam': 'adam', 'Nadam': 'nadam'}
    
    opt = opt_dict[optimizer]

    # Compile the model
    model = models.Model(inputs, outputs)
    
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse', CCC, CC, cc0_for_channel(0), cc1_for_channel(1), cc2_for_channel(2), ccc0_for_channel(0), ccc1_for_channel(1), ccc2_for_channel(2)])
    
    #print('build_model done, elaT=%.1f sec'%(time.time() - start))
    
    return model

#############################################################################
def train_model(model,
                batch_size, n_epochs,
                checkpoint_dir=None, checkpoint_file=None, data_dir_path=None, l_fnames_train=None, l_fnames_val=None, ihp=0,
                verbose=2, callbacks=[]):
    
    """Train the model"""
    rpv_callbacks = []
    
    lrCb=MyLearningTracker()
    rpv_callbacks.append(lrCb)
    
    tcb = TimingCallback()
    rpv_callbacks.append(tcb)
    
    rpv_callbacks.append(
        # Reduce the learning rate if training plateaues.
        tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
            patience=4, factor=0.3, verbose=1, min_lr=0.0, epsilon=0.01)
    )

    
    rpv_callbacks.append(
        tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_delta=0.001)
    )
    
    rpv_callbacks.append(
        tensorflow.keras.callbacks.CSVLogger(checkpoint_dir + '/training_' + str(ihp) + '.log')
    )
        
    if checkpoint_file is not None:
        rpv_callbacks.append(
            #tensorflow.keras.callbacks.callbacks.ModelCheckpoint(checkpoint_dir + file_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=2))
        #rpv_callbacks.append(tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_file))

    if isinstance(list, type(callbacks)) and len(callbacks) > 0:
        rpv_callbacks.extend(callbacks)

    #rpv_callbacks.append(tensorflow.keras.callbacks.LearningRateScheduler(step_decay))
        
    inpGenD = {}
    #print('info: ', data_dir_path, l_fnames_train, l_fnames_val)
    
    inpGenD['train'] = MyImageGenerator(dom='train', inpPath=data_dir_path, 
                                        batch_size=batch_size, fnameL=l_fnames_train, fakeSteps=None, verb=0)
    
    inpGenD['val'] = MyImageGenerator(dom='val', inpPath=data_dir_path, 
                                      batch_size=batch_size, fnameL=l_fnames_val, fakeSteps=None, verb=0)

    m = model.fit_generator(
                generator=inpGenD['train'], 
                validation_data=inpGenD['val'], 
                callbacks=rpv_callbacks, shuffle=True, 
                epochs=n_epochs,
                verbose=1)

    return m



