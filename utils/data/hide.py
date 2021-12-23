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
        hide --strategy-start=2016-03-21-00:00:00 --strategy-end=2016-03-21-23:59:00 --verbose=True hide.config.bleien7m

        
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

    files = glob('{}/*/*.h5'.format(args.data_path))
    data = np.empty([len(files)//2, 
                     sizes[args.data], 
                     sizes[args.data], 1], 
                     dtype=np.float32)
    masks = np.empty([len(files)//2, 
                     sizes[args.data], 
                     sizes[args.data], 1], 
                     dtype=np.bool)
    for i in tqdm(range(len(files))):# loop over step size 2 to make data shape > (256,256)
        f = File(files[i])
        temp_data, temp_rfi = f['P/Phase1'][()], f['RFI/Phase0'][()] 
        f.close()
    
        temp_data = np.expand_dims(temp_data, axis=[0,-1])
        temp_rfi = np.expand_dims(temp_rfi, axis=[0,-1])
        snr = (np.mean(temp_data)/np.mean(temp_rfi))

        temp_masks = temp_rfi>(0.1*np.std(temp_data))
        
        temp_data, temp_masks = _random_crop(temp_data.astype('float32'),
                                             temp_masks.astype('int'),
                                             (sizes[args.data], sizes[args.data]))
        data[i:i+1,...] = temp_data
        masks[i:i+1,...] = temp_masks.astype('bool')

    (train_data, test_data,
     train_masks, test_masks) = train_test_split(data, masks, test_size=0.25, random_state=42)

    pickle.dump((train_data, train_masks, test_data, test_masks), open('{}/joined_dataset.pickle'.format(args.data_path), 'wb'), protocol=4)

    return train_data, train_masks, test_data, test_masks


        
        
