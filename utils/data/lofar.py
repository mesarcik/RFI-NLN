import os
import pickle
import numpy as np
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

def get_lofar_data(directory, args, num_baselines=400):
    """"
        Walks through LOFAR dataset and returns sampled and cropped data 
        
        directory (str): Directory where LOFAR dataset resides
        args (Namespace): args from utils.cmd_args 
        num_baselines (int): number of baselines to sample 
    """

    # read each npy file and select, crop it to 512x512 
    # if the training dataset has already been created then return that

    if os.path.exists(os.path.join(directory,'joined_dataset.pickle')):
        print(os.path.join(directory,'joined_dataset.pickle') + ' Loading')
        with open('{}/joined_dataset.pickle'.format(directory),'rb') as f:
            return pickle.load(f)

    else:
        print('Creating joined LOFAR dataset')

    files = glob('{}/*.npy'.format(directory))
    data = np.empty([len(files)*num_baselines, 
                     sizes[args.data], 
                     sizes[args.data], 1], 
                     dtype=np.float32)
    masks = np.empty([len(files)*num_baselines, 
                     sizes[args.data], 
                     sizes[args.data], 1], 
                     dtype=np.bool)

    strt, fnnsh = 0, num_baselines
    for f in tqdm(glob('{}/*.npy'.format(directory))):
        temp_data, temp_flags = np.load(f, allow_pickle=True)
        inds = np.random.choice(range(len(temp_data)), num_baselines, replace=False)
        temp_data, temp_flags  = temp_data[inds], temp_flags[inds]

        temp_data, temp_flags = _random_crop(np.absolute(temp_data[...,0:1]).astype('float32'),
                                             temp_flags[...,0:1].astype('int'),
                                             (sizes[args.data], 
                                             sizes[args.data]))
        data[strt:fnnsh,...] = temp_data
        masks[strt:fnnsh,...] = temp_flags.astype('bool')

        strt=fnnsh
        fnnsh = fnnsh + num_baselines

    (train_data, test_data,
     train_masks, test_masks) = train_test_split(data,
                                                 masks,
                                                 test_size=0.25,
                                                 random_state=42)

    pickle.dump((train_data, train_masks, test_data, test_masks), open('{}/joined_dataset.pickle'.format(directory), 'wb'), protocol=4)

    return train_data, train_masks, test_data, test_masks


        
        
