import os
import errno
import pickle
import numpy as np
import tensorflow as tf
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

def get_lofar_data(args, num_baselines=400):
    """"
        Walks through LOFAR dataset and returns sampled and cropped data 
        
        args (Namespace): args from utils.cmd_args 
        num_baselines (int): number of baselines to sample 
    """
    if os.path.exists(os.path.join(args.data_path,'LOFAR_Full_RFI_dataset.pkl')):
        print(os.path.join(args.data_path,'LOFAR_Full_RFI_dataset.pkl') + ' Loading')
        with open('{}/LOFAR_Full_RFI_dataset.pkl'.format(args.data_path),'rb') as f:
            return  pickle.load(f)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                os.path.join(args.data_path,'joined_dataset.pickle'))
