import os
import pickle
import numpy as np
from h5py import File
from tqdm import tqdm 
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

def get_hide_data(args, sigma=5):
    """"
        Walks through the simulated HIDE data and creates a training set generated from:
        hide --strategy-start=2016-01-01-00:00:00 --strategy-end=2016-12-31-23:59:00 --verbose=True hide.config.bleien7m
        seek --file-prefix='./synthesized' --post-processing-prefix='synthesized/seek_cache'\
              --chi-1=20 --overwrite=True seek.config.process_survey_fft

        
        args.data_path (str): Directory where LOFAR dataset resides
        args (Namespace): args from utils.cmd_args 
        num_baselines (int): number of baselines to sample 
    """

    if os.path.exists(os.path.join(args.data_path,'joined_dataset.pickle')):
        print(os.path.join(args.data_path,'joined_dataset.pickle') + ' Loading')
        with open('{}/joined_dataset.pickle'.format(args.data_path),'rb') as f:
            return pickle.load(f)

    else:
        print('Creating joined HIDE dataset')

    mixture_data_files = glob('{}/*.h5'.format(args.data_path))

    rfi_path = args.data_path.split('full_year')
    rfi_path = os.path.join(rfi_path[0], 'rfi_full_year','seek_cache')
    rfi_files = glob('{}/*.h5'.format(rfi_path))

    data = np.empty([len(mixture_data_files), 
                     sizes[args.data], 
                     sizes[args.data], 1], 
                     dtype=np.float32)
    masks = np.empty([len(rfi_files), 
                     sizes[args.data], 
                     sizes[args.data], 1], 
                     dtype=np.bool)
    for i in tqdm(range(len(mixture_data_files))):
        with File(mixture_data_files[i], "r") as f_data, File(rfi_files[i], "r") as f_rfi:
            mixture_data = f_data['data'][:].astype(np.float32)
            rfi = f_rfi['data'][:].astype(np.float32)
            #signal = mixture_data - rfi
            #snr = np.mean(signal)/np.mean(rfi)
            snr = 0.670 # SNR magic number from Sadr et. al. 
            mask = rfi>(sigma*snr)
    
        mixture_data = np.expand_dims(mixture_data, axis=[0,-1])
        mask = np.expand_dims(mask, axis=[0,-1])
        
        mixture_data, mask = _random_crop(mixture_data.astype('float32'),
                                             mask.astype('int'),
                                             (sizes[args.data], sizes[args.data]))
        data[i:i+1,...] = mixture_data
        masks[i:i+1,...] = mask.astype('bool')

    (train_data, test_data,
     train_masks, test_masks) = train_test_split(data, masks, test_size=0.1, random_state=42)# 0.1*365 /aprox 1 month

    pickle.dump((train_data, train_masks, test_data, test_masks), open('{}/joined_dataset.pickle'.format(args.data_path), 'wb'), protocol=4)

    return train_data, train_masks, test_data, test_masks


        
        
