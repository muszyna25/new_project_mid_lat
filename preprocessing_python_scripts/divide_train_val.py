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


input_dir = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/copy_shuffled_balanaced_train_val_test/shuffled_sets'
output_dir = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/copy_shuffled_balanaced_train_val_test/shuffled_sets/divided_into_smaller_files' 

data_train, labels_train = load_my_data(os.path.join(input_dir, 'train.h5'), 0)

n_chunks = 3.0

chunk_size = int(data_train.shape[0]/n_chunks)
counter = 0
for i in range(0, data_train.shape[0], chunk_size): 
    print('Chunk: ', counter)
    chunk_X = data[i:i+chunk_size,:]
    chunk_Y = labels[i:i+chunk_size,:]
    save_chunk_to_hdf5(counter, path, chunk_X, chunk_Y)
    save_to_hdf5_format(counter, train_input, train_labels, 'train.h5', output_dir)
    counter+=1
    




