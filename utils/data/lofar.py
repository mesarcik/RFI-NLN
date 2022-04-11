import os
import pickle
import numpy as np
from tqdm import tqdm 
from h5py import File
from glob import glob
import tensorflow as tf
from utils.data.defaults import sizes 
from sklearn.model_selection import train_test_split
from model_config import BATCH_SIZE

def _random_crop(image,mask,size):
    output_images = np.empty((len(image), size[0], size[1], 1)).astype('float32')
    output_masks = np.empty((len(mask), size[0], size[1], 1)).astype('bool')
    strt, fnnsh = 0, BATCH_SIZE
    for i in range(0,len(image),BATCH_SIZE):
        stacked_image = np.stack([image[strt:fnnsh,...],
                                  mask[strt:fnnsh,...].astype('float32')],axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2,len(stacked_image[0]), size[0], size[1], 1])
        output_images[strt:fnnsh,...]  = cropped_image[0].numpy()
        output_masks[strt:fnnsh,...]  = cropped_image[1].numpy().astype('bool')
        strt=fnnsh
        fnnsh+=BATCH_SIZE
    return output_images, output_masks

def get_lofar_data(directory, args, num_baselines=400):
    """"
        Walks through LOFAR dataset and returns sampled and cropped data 
        
        directory (str): Directory where LOFAR dataset resides
        args (Namespace): args from utils.cmd_args 
        num_baselines (int): number of baselines to sample 
    """

    with File('/data/mmesarcik/LOFAR/uncompressed/LOFAR_dataset.h5', 'r') as f:
        data = f['visibilities'][:].astype('float32')
        masks = f['aoflags'][:].astype('bool')

    train_data, train_masks = _random_crop(data, masks, (sizes[args.data], sizes[args.data]))

    test_data, test_masks = np.load('/data/mmesarcik/LOFAR/uncompressed/LOFAR_test.npy')
    test_data = test_data.astype('float32')
    test_masks= test_masks.astype('bool')

    train_data = np.concatenate([train_data, np.roll(train_data,
                                                     args.patch_x//2, 
                                                     axis =2)], axis=0)# this is less artihmetically complex then making stride half

    train_masks = np.concatenate([train_masks, np.roll(train_masks,
                                                       args.patch_x//2, 
                                                       axis =2)], axis=0)
    return train_data, train_masks, test_data, test_masks



        
        
