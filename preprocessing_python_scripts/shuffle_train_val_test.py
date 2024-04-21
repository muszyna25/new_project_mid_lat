import h5py
import time
import os
import sys
from sklearn.utils import shuffle

def load_my_data(fname, n):
    
    print(os.popen('free -g').read())
    start = time.time()
    print('load hdf5:',fname, flush=True)
    
    sys.stdout.flush()
    
    with h5py.File(fname, 'r') as h5f:
        X=h5f['X'][:]
        Y=h5f['Y'][:]
        print(' done',X.shape, Y.shape)

    print('load_input_hdf5 done, elaT=%.1f sec'%(time.time() - start))
    print(os.popen('free -g').read())
    
    return X, Y

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


input_dir = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/copy_shuffled_balanaced_train_val_test/'
output_dir = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/copy_shuffled_balanaced_train_val_test/shuffled_sets'

'''
_train_input, _train_labels = load_my_data(os.path.join(input_dir, 'train.h5'), 0)
train_input, train_labels = shuffle(_train_input, _train_labels, random_state=111)
save_to_hdf5_format(train_input, train_labels, 'train.h5', output_dir)
'''

_val_input, _val_labels = load_my_data(os.path.join(input_dir, 'val.h5'), 0)
val_input, val_labels = shuffle(_val_input, _val_labels, random_state=111)
save_to_hdf5_format(val_input, val_labels, 'val.h5', output_dir)



