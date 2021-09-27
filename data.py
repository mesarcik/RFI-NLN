import tensorflow as tf
import numpy as np
import copy
import random
from model_config import BUFFER_SIZE,BATCH_SIZE
from sklearn.model_selection import train_test_split
from utils.data import (get_mvtec_images, 
                        process,
                        rgb2gray,
                        get_patched_dataset,
                        random_rotation,
                        random_crop,
                        resize,
                        sizes)
def load_hera(args):
    """
        Load data from hera

    """
    unet_path = '/data/mmesarcik/hera/HERA_6_24-09-2021.pkl'
    unet_data, unet_labels, unet_masks, _ =  np.load(unet_path, allow_pickle=True)

    ae_path = '/data/mmesarcik/hera/HERA_0_24-09-2021_AE.pkl'
    ae_data, ae_labels, ae_masks, _ =  np.load(ae_path, allow_pickle=True)

    _unet_labels = copy.deepcopy(unet_labels)
    for i,label in enumerate(unet_labels):
        if args.anomaly_class in label: 
            _unet_labels[i] = args.anomaly_class
        else:
            _unet_labels[i] = 'normal'
    unet_labels = np.array(_unet_labels)
    ae_labels = np.array('normal'*len(ae_labels))

    unet_data = np.sqrt(unet_data[...,0]**2 + unet_data[...,1]**2)
    unet_data = np.expand_dims(unet_data, axis=-1)
    mi,ma = np.min(unet_data), np.max(unet_data)
    unet_data = (unet_data - mi)/(ma -mi)
    unet_data = unet_data.astype('float32')

    ae_data = np.sqrt(ae_data[...,0]**2 + ae_data[...,1]**2)
    ae_data = np.expand_dims(ae_data, axis=-1)
    ae_data = (ae_data - mi)/(ma -mi)
    ae_data = ae_data.astype('float32')#process(ae_data, per_image=True)

    unet_masks = np.swapaxes(unet_masks, 1,2)
    unet_masks = np.expand_dims(unet_masks,axis=-1)

    ae_masks = np.swapaxes(ae_masks, 1,2)
    ae_masks = np.expand_dims(ae_masks,axis=-1)

    (unet_train_data, unet_test_data, 
        unet_train_labels, unet_test_labels, 
            unet_train_masks, unet_test_masks) = train_test_split(unet_data, 
                                                                  unet_labels, 
                                                                  unet_masks,
                                                                  test_size=0.25, 
                                                                  random_state=42)
#    if str(args.anomaly_class) is not None:
#        if args.anomaly_type == 'MISO':
#            indicies = np.argwhere(train_labels == str(args.anomaly_class))
#
#            mask_train  = np.invert(train_labels == str(args.anomaly_class))
#        else: 
#            indicies = np.argwhere(train_labels != str(args.anomaly_class))
#
#            mask_train  = train_labels == str(args.anomaly_class)
#
#        train_data= train_data[mask_train]
#        train_labels = train_labels[mask_train]
#        train_masks = train_masks[mask_train]
#
    if args.limit is not None:
        unet_train_data =   unet_train_data[:args.limit,...]
        unet_train_labels = unet_train_labels[:args.limit,...]
        unet_train_masks =  unet_train_masks[:args.limit,...]

        ae_data = ae_data[:args.limit,...]

    unet_train_dataset = tf.data.Dataset.from_tensor_slices(unet_train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    ae_train_dataset = tf.data.Dataset.from_tensor_slices(ae_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return (unet_train_dataset,
            ae_train_dataset,
            unet_train_data, 
            ae_data, 
            unet_train_masks, 
            ae_masks,
            unet_train_labels,
            ae_labels,
            unet_test_data, 
            unet_test_labels, 
            unet_test_masks)

def load_lofar(args, path, unet=False):            
    """
        Load data from lofar 

    """
    [train_data, train_labels, train_masks, test_data, test_labels, test_masks] = np.load(path, allow_pickle=True)
    _train_labels = np.array(['normal']*len(train_labels),dtype='str')
    _test_labels = np.array(['normal']*len(test_labels),dtype='str')

    for i,(_train_label,_test_label) in enumerate(zip(train_labels, test_labels)):
        if _train_label:_train_labels[i] = args.anomaly_class
        else:_train_labels[i] = 'normal'
        if _test_label:_test_labels[i] = args.anomaly_class
        else:_test_labels[i] = 'normal'
    test_labels = np.array(_test_labels)
    train_labels = np.array(_train_labels)

    if args.limit is not None:
        train_data = train_data[:args.limit,...]
        train_labels = train_labels[:args.limit,...]

    train_data = process(train_data, per_image=False)
    test_data = process(test_data, per_image=False)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE,seed=42).batch(BATCH_SIZE)
    if unet:
        return (train_dataset,train_data, train_masks, train_labels, test_data, test_labels, test_masks)
    else:
        return (train_dataset,train_data, train_labels, test_data, test_labels, test_masks)
