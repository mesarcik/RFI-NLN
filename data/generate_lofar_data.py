'''
    This file contains the data generation environment for lofar-dev
    Misha Mesarcik 2019
'''
import numpy as np
import pandas as pd
import pickle
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

import tensorflow as tf
from h5_interface import *

import sys 
sys.path.insert(1,'/home/mmesarcik/RFI-AE/')
from utils import cmd_input 


def data_generator(args):
    """
        Generate the training data for UNET and AEs 
        
    """
    first_flag = False 
    df = pd.read_csv('lofar_data_summary')
    train_names = list(df[(df.Contamination ==0) & (df.Channels>30) &(df.Times>30) & (df.Times <243)].Name)
    test_names = list(df[(df.Contamination > 0.0) & (df.Channels>30) &(df.Times>30) & (df.Times <243)].Name)

    train_labels, test_labels = [], []

    for file_name in tqdm(train_names, desc='create train data'):
        file_path = os.path.join(str(args.data_path), str(file_name))
        print(file_path)
        vis, flags  = get_cube(file_path)
        mag = np.sqrt(vis[...,0].real**2 + vis[...,0].imag**2)
        mag = np.expand_dims(mag,axis=-1)
        mag = tf.image.resize(mag, (128,64), method='area',antialias=False).numpy()

        #phase = np.angle(vis[...,0])
        #phase = np.expand_dims(phase,axis=-1)
        #phase = tf.image.resize(phase, (128,64), method='area',antialias=False).numpy()

        if not first_flag:
            train_data = mag 
            first_flag = True
        else: 
            train_data = np.concatenate((train_data, mag),axis=0)
        train_labels.extend(list([f.any() for f in flags[...,0]]))

    first_flag = False 
    for file_name in tqdm(test_names,desc='creating test data'):
        file_path = os.path.join(str(args.data_path), str(file_name))
        vis, flags = get_cube(file_path)
        mag = np.sqrt(vis[...,0].real**2 + vis[...,0].imag**2)
        mag = np.expand_dims(mag,axis=-1)
        mag = tf.image.resize(mag, (128,64), method='area',antialias=False).numpy()

        #phase = np.angle(vis[...,0])
        #phase = np.expand_dims(phase,axis=-1)
        #phase = tf.image.resize(phase, (128,64), method='area',antialias=False).numpy()

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

    #TODO our broken masks may have to do with this split.
    (unet_train_data, unet_test_data, 
        unet_train_labels, unet_test_labels, 
            unet_train_masks, unet_test_masks) = train_test_split(test_data, 
                                                                  test_labels, 
                                                                  test_masks,
                                                                  test_size=0.50, 
                                                                  random_state=42)

    if not os.path.exists('datasets'):
        os.mkdir('datasets')

    UNET_name = 'datasets/LOFAR_UNET_dataset_{}_small.pkl'.format(datetime.datetime.now().strftime("%d-%m-%Y"))
    pickle.dump([unet_train_data, 
                 unet_train_labels, 
                 unet_train_masks, 
                 unet_test_data, 
                 unet_test_labels, 
                 unet_test_masks],open(UNET_name, 'wb'), protocol=4)


    AE_name = 'datasets/LOFAR_AE_dataset_{}_small.pkl'.format(datetime.datetime.now().strftime("%d-%m-%Y"))
    pickle.dump([train_data, 
                 train_labels, 
                 np.empty([]), 
                 unet_test_data, 
                 unet_test_labels, 
                 unet_test_masks],open(AE_name, 'wb'), protocol=4)


def main():
    args = cmd_input.args
    data_generator(args)

if __name__ == '__main__':
    main()
