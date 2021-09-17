'''
    This file contains the data generation environment for lofar-dev
    Misha Mesarcik 2019
'''
import numpy as np
import pandas as pd
import pickle
import datetime
from tqdm import tqdm
import os

import tensorflow as tf
from h5_interface import *


def data_generator(num_files):
    first_flag = False 
    df = pd.read_csv('lofar_data_summary')
    #train_names = list(df[df.Contamination ==0].Name)[:num_files]
    test_names = list(df[df.Contamination > 0.2].Name)[:num_files]

    train_names = test_names[len(test_names)//2:]
    test_names = test_names[:len(test_names)//2]

    train_labels, test_labels = [], []

    for file_name in tqdm(train_names, desc='create train data'):
        vis, flags  = get_cube(file_name)
        mag = np.sqrt(vis[...,0].real**2 + vis[...,0].imag**2)
        mag = np.expand_dims(mag,axis=-1)
        mag = tf.image.resize(mag, (128,64), method='area',antialias=False).numpy()
        flags = tf.image.resize(flags.astype('int'), (128,64), method='area',antialias=False).numpy().astype('bool')

        if not first_flag:
            train_data = mag 
            first_flag = True
            train_masks = flags[...,0]
        else: 
            train_data = np.concatenate((train_data, mag),axis=0)
            train_masks = np.concatenate((train_masks, flags[...,0]),axis=0)
        train_labels.extend(list([f.any() for f in flags[...,0]]))

    first_flag = False 
    for file_name in tqdm(test_names,desc='creating test data'):
        vis, flags = get_cube(file_name)
        mag = np.sqrt(vis[...,0].real**2 + vis[...,0].imag**2)
        mag = np.expand_dims(mag,axis=-1)
        mag = tf.image.resize(mag, (128,64), method='area',antialias=False).numpy()
        flags = tf.image.resize(flags.astype('int'), (128,64), method='area',antialias=False).numpy().astype('bool')
        print(flags.any())

        if not first_flag:
            test_data = mag 
            test_masks = flags[...,0]
            first_flag = True
        else: 
            test_data = np.concatenate((test_data,mag),axis=0)
            test_masks = np.concatenate((test_masks, flags[...,0]),axis=0)
        test_labels.extend([f.any() for f in flags[...,0]]) 

    test_labels, train_labels = np.array(test_labels), np.array(train_labels)

    if not os.path.exists('datasets'):
        os.mkdir('datasets')

    f_name = 'datasets/LOFAR_UNET_dataset_{}.pkl'.format(datetime.datetime.now().strftime("%d-%m-%Y"))

    pickle.dump([train_data, train_labels,train_masks, test_data, test_labels, test_masks],open(f_name, 'wb'), protocol=4)
    print('{} Saved!'.format(f_name))

def main():
    data_generator(-1)

if __name__ == '__main__':
    main()
