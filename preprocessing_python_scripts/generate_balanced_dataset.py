from netCDF4 import Dataset
import h5py
import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

#####################################
def get_hdf5_files_list(path, offset=-5):
    l_files = []
    l_files = [fn for fn in glob.glob(path + '*.h5')] # Get list of all netCDF files in the directory.
    assert len(l_files) != 0, 'Directory does not have hdf5 files!' # If there are no netCDF files in the directory.
    l_files = sorted(l_files,  key = lambda x:(x[:-offset], x[4:-3])) # Sort files according to 'year' in the file name.
    return l_files

#####################################
def save_to_hdf5_format(X_train, X_test, X_val, y_train, y_test, y_val, out_path):

    print('Data shapes: ', X_train.shape, X_test.shape, X_val.shape, y_train.shape[0], y_test.shape, y_val.shape)

    y_train_ = np.reshape(y_train, (y_train.shape[0],1))
    y_test_ = np.reshape(y_test, (y_test.shape[0],1))
    y_val_ = np.reshape(y_val, (y_val.shape[0],1))

    if os.path.exists(out_path + 'train.h5') == False:
        print('Create train.h5')
        hf = h5py.File(out_path + 'train.h5', 'w')
        hf.create_dataset('X', data=X_train, maxshape=(None, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        hf.create_dataset('Y', data=y_train_, maxshape=(None, 1))
        hf.close()
    elif os.path.exists(out_path + 'train.h5') == True:
        print('Update train.h5')
        hf = h5py.File(out_path + 'train.h5', 'a')
        hf['X'].resize((hf['X'].shape[0] + X_train.shape[0]), axis = 0)
        hf['X'][-X_train.shape[0]:] = X_train
        hf['Y'].resize((hf['Y'].shape[0] + y_train_.shape[0]), axis = 0)
        hf['Y'][-y_train_.shape[0]:] = y_train_
        hf.close()
    if os.path.exists(out_path + 'test.h5') == False:
        print('Create test.h5')
        hf = h5py.File(out_path + 'test.h5', 'w')
        hf.create_dataset('X', data=X_test, maxshape=(None, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        hf.create_dataset('Y', data=y_test_, maxshape=(None, 1))
        hf.close()
    elif os.path.exists(out_path + 'test.h5') == True:
        print('Update test.h5')
        hf = h5py.File(out_path + 'test.h5', 'a')
        hf['X'].resize((hf['X'].shape[0] + X_test.shape[0]), axis = 0)
        hf['X'][-X_test.shape[0]:] = X_test
        hf['Y'].resize((hf['Y'].shape[0] + y_test_.shape[0]), axis = 0)
        hf['Y'][-y_test_.shape[0]:] = y_test_
        hf.close()
    if os.path.exists(out_path + 'val.h5') == False:
        print('Create val.h5')
        hf = h5py.File(out_path + 'val.h5', 'w')
        hf.create_dataset('X', data=X_val, maxshape=(None, X_val.shape[1], X_val.shape[2], X_val.shape[3]))
        hf.create_dataset('Y', data=y_val_, maxshape=(None, 1))
        hf.close()
    elif os.path.exists(out_path + 'val.h5') == True:
        print('Update val.h5')
        hf = h5py.File(out_path + 'val.h5', 'a')
        hf['X'].resize((hf['X'].shape[0] + X_val.shape[0]), axis = 0)
        hf['X'][-X_val.shape[0]:] = X_val
        hf['Y'].resize((hf['Y'].shape[0] + y_val_.shape[0]), axis = 0)
        hf['Y'][-y_val_.shape[0]:] = y_val_
        hf.close()

#####################################
#
#              MAIN
#
#####################################

input_path = '/global/cscratch1/sd/muszyng/ethz_data/hdf5_data/'
#out_path = '/global/cscratch1/sd/muszyng/ethz_data/train_val_test_hdf5/downsampling/'
out_path = '/global/cscratch1/sd/muszyng/ethz_data/train_val_test_hdf5/downsampling_v2/'
l_files = get_hdf5_files_list(input_path)
print(*l_files, sep='\n')

for i in range(0, len(l_files)):
#for i in range(0, 3): # Read each hdf5 file...
    fn = l_files[i]
    print('File:', fn)
    f = h5py.File(fn, 'r')
    l_keys = list(f.keys()) # List of keys in the file.
    print('Keys: ', l_keys)

    P_patch = f[l_keys[3]]
    N_patch = f[l_keys[1]]
    print('No of samples in P: %d; No of samples in N: %d' %(P_patch.shape[0], N_patch.shape[0]))
    #POS_up_sampled = resample(np.asarray(P_patch), replace=True, n_samples=int(N_patch.shape[0]), random_state=123) # Up-sampling postive class
    NEG_down_sampled = resample(np.asarray(N_patch), replace=False, n_samples=int(P_patch.shape[0]), random_state=123) # Down-sampling negative class
    #print('Upsampled set: ', POS_up_sampled.shape)
    print('Downsampled set: ', NEG_down_sampled.shape)

    X = np.concatenate((np.asarray(P_patch),NEG_down_sampled))
    #X = np.concatenate((np.asarray(N_patch),POS_up_sampled))
    print(X.shape)
    
    #y_ones = np.ones(int(POS_up_sampled.shape[0]))
    #y_zeros = np.zeros(int(N_patch.shape[0]))
    y_ones = np.ones(int(P_patch.shape[0]))
    y_zeros = np.zeros(int(NEG_down_sampled.shape[0]))
    y = np.concatenate((y_zeros, y_ones))
    print(y.shape)

    f.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=123)
    
    save_to_hdf5_format(X_train, X_test, X_val, y_train, y_test, y_val, out_path)


