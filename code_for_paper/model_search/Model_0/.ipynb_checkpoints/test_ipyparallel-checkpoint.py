# System imports
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os

# External imports
import ipyparallel as ipp
import numpy as np
import tensorflow.python.keras
import matplotlib.pyplot as plt
import glob
import random
import csv

# Local imports
from model_code import get_job_id, save_hyper_params_to_csv

#############################################################################
def setup_cluster(model_name):
    # Cluster ID taken from job ID above
    job_id = get_job_id('squeue -u muszyng -o %A -n ' + model_name)
    cluster_id = 'cori_{}'.format(job_id)
    print('Job id: %d; Cluster id: %s' %(job_id, cluster_id))

    # Use default profile
    c = ipp.Client(timeout=60, cluster_id=cluster_id)
    print('Worker IDs:', c.ids)
    
    return job_id, cluster_id, c

#############################################################################
def get_train_val_names(input_dir):
    l_fnames_train = [os.path.basename(x) for x in glob.glob(input_dir + 'train*.h5')]
    l_fnames_val = [os.path.basename(x) for x in glob.glob(input_dir + 'val*.h5')] 
    print(list(l_fnames_train), list(l_fnames_val), sep='\n')
    return l_fnames_train, l_fnames_val

#############################################################################
def build_and_train(input_dir, input_size,
                    conv_sizes, fc_sizes, dropout, optimizer, lr,
                    batch_size, n_epochs, repeats_conv, recep_field, 
                    channels, checkpoint_dir=None, checkpoint_file=None, 
                    l_fnames_train=None, l_fnames_val=None, ihp=0, verbose=2):
    
    """Run training for one set of hyper-parameters"""
    import sys
    from model_code import build_model, train_model
    from mlextras import configure_session
    import numpy as np
    import os
    import time
    
    from contextlib import redirect_stdout

    def save_model_summary(i, dir_path):
        with open(dir_path + '/modelsummary_' + str(i) + '.csv', 'w') as f:
            with redirect_stdout(f):
                model.summary()
                n_params = model.count_params()/1000.
                print('No of parameters: %i' %n_params)
            
    def save_model_config(i, dir_path):
        model_yaml = model.to_yaml()
        with open(dir_path + '/model_' + str(i) + '.yaml', 'w') as yaml_file:
            yaml_file.write(model_yaml)
    
    change_model_dbg = True
    
    oldStdout = sys.stdout
    file = open(checkpoint_dir + '/logFile_' + str(ihp) + '.log', 'w')
    sys.stdout = file

    import tensorflow.python.keras

    print('Hyperparameter trial %i, conv %s, fc %s, dropout %.4f, opt %s, lr %.4f, batch size %d, repeats conv layers %s, recep fields %s, checkpoint dir: %s' %
          (ihp, conv_sizes, fc_sizes, dropout, optimizer, lr, batch_size, repeats_conv, recep_field, checkpoint_dir))
    
    sys.stdout.flush()

    # Thread settings
    tensorflow.keras.backend.set_session(configure_session())
    
    # Build the model
    print(fc_sizes)
    model = build_model(input_size,
                        conv_sizes=conv_sizes, repeats_conv=repeats_conv, recep_field=recep_field, fc_sizes=fc_sizes,
                        dropout=dropout, optimizer=optimizer, lr=lr)
    
    n_params = model.count_params()/1000.
    model.summary()
    print('No of parameters: %i' %n_params)
    
    sys.stdout.flush()
    sys.stdout = oldStdout
    
    save_model_config(ihp, checkpoint_dir)
    save_model_summary(ihp, checkpoint_dir)
    
    # Train the model
    sys.stdout.flush()
    start = time.time()
    
    if change_model_dbg:
        history = train_model(model,
                              batch_size=batch_size, n_epochs=n_epochs,
                              checkpoint_dir=checkpoint_dir, checkpoint_file=checkpoint_file, 
                              data_dir_path=input_dir[:-1],
                              l_fnames_train=l_fnames_train, l_fnames_val=l_fnames_val, ihp=ihp, verbose=verbose)
        
        oldStdout = sys.stdout
        sys.stdout = file
        print('train_model done, elaT=%.1f sec'%(time.time() - start))
        sys.stdout.flush()
        sys.stdout = oldStdout
        return history.history

#############################################################################
def test_model_config(input_size, conv_sizes, repeats_conv, 
                      recep_field, fc_sizes,
                      dropout, optimizer, lr):
    
    from model_code import build_model
    
    model = build_model(input_size, conv_sizes=conv_sizes, repeats_conv=repeats_conv, 
                         recep_field=recep_field, fc_sizes=fc_sizes,
                         dropout=dropout, optimizer=optimizer, lr=lr)
    model.summary()

#############################################################################
def plot_walltime(results):
    done_results = [ar for ar in results if ar.ready()]
    times = [(ar.completed - ar.started).total_seconds()/60.0 for ar in done_results]
    plt.figure()
    h = plt.hist(times, bins='auto', rwidth=0.3)
    plt.grid()
    plt.xlabel('Training time [min]')
    plt.ylabel('# of models');
    
############################################################## MAIN #####################################################################################
if __name__ == "__main__":
    
    model_name = 'Model_0'
    job_id, cluster_id, c = setup_cluster(model_name)
    
    # Data path
    path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/generated_data_full_analysis/'
    input_dir = path + 'simplified_0/u/big_hdf5_files/chunk_files/reduced_chunk_files/'
    
    l_fnames_train, l_fnames_val = get_train_val_names(input_dir)
    
    # Temporarily making things reproducible for development
    #np.random.seed(0)

    # Define the hyper-parameter search points
    n_hpo_trials = 10

    n_convs_layers = 5
    layer_sizes = np.random.choice(np.array([8, 16, 32, 64]), size=(n_hpo_trials, n_convs_layers))
    conv_sizes = np.array([sorted(x, reverse=False) for x in layer_sizes])

    n_fc_layers = 3
    dense_sizes = np.array([512, 256, 128, 64])
    dtuple_sizes = np.random.choice(dense_sizes, size=(n_hpo_trials, n_fc_layers))
    fc_sizes = np.array([sorted(x, reverse=True) for x in dtuple_sizes])

    lr = np.zeros((n_hpo_trials,1))
    dropout = np.random.uniform(0.1, 0.5, size=n_hpo_trials)
    optimizer = np.random.choice(['Adam', 'Nadam'], size=n_hpo_trials)
    batch_sizes = np.random.choice([64, 128, 256], size=n_hpo_trials)

    repeats_conv = [(0,0)] #[(2,2), (3,2), (4,2)]
    recep_fields = (3,3)
    channels = (0,0)

    input_size = (60, 120, 8)

    # Training config
    n_epochs = 500
    checkpoint_dir = os.path.join(os.environ['SCRATCH'],'cnn_model_runs/%s/%i' % (model_name, job_id))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(checkpoint_dir)
    
    save_hyper_params_to_csv(n_hpo_trials, n_convs_layers, conv_sizes, 
                         n_fc_layers, fc_sizes, dropout, 
                         optimizer, lr, batch_sizes, 
                         n_epochs, repeats_conv, recep_fields, 
                         checkpoint_dir, job_id)
    
    # Load-balanced view
    lv = c.load_balanced_view()

    # Loop over hyper-parameter sets
    results = []

    for ihp in range(n_hpo_trials):

        print('Hyperparameter trial %i conv %s fc %s dropout %.4f opt %s, lr %.4f, batch sizes %d' %
              (ihp, conv_sizes[ihp], fc_sizes[ihp], dropout[ihp], optimizer[ihp], lr[ihp], batch_sizes[ihp]))

        checkpoint_file = os.path.join(checkpoint_dir, 'model_%i.h5' % ihp)

        print('Checkpoint file: %s' %checkpoint_file)

        result = lv.apply(build_and_train, input_dir, input_size, conv_sizes=conv_sizes[ihp], 
                          fc_sizes=fc_sizes[ihp], dropout=dropout[ihp], optimizer=optimizer[ihp], 
                          lr=lr[ihp], batch_size=batch_sizes[ihp], n_epochs=n_epochs, 
                          repeats_conv=repeats_conv, recep_field=recep_fields, channels=channels, 
                          checkpoint_dir=checkpoint_dir, checkpoint_file=checkpoint_file, l_fnames_train=l_fnames_train, 
                          l_fnames_val=l_fnames_val, ihp=ihp)

        results.append(result)