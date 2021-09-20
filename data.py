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
    data, labels, masks, _ =  np.load(args.data_path, allow_pickle=True)
    _labels = copy.deepcopy(labels)
    for i,label in enumerate(labels):
        if args.anomaly_class in label: 
            _labels[i] = args.anomaly_class
        else:
            _labels[i] = 'normal'
    labels = np.array(_labels)

    data = np.sqrt(data[...,0]**2 + data[...,1]**2)
    data = np.expand_dims(data, axis=-1)
    data = process(data, per_image=False)
    masks = np.swapaxes(masks, 1,2)
    masks = np.expand_dims(masks,axis=-1)

    (train_data, test_data, 
        train_labels, test_labels, 
            train_masks, test_masks) = train_test_split(data, 
                                                        labels, 
                                                        masks,
                                                        test_size=0.25, 
                                                        random_state=42)

    if str(args.anomaly_class) is not None:
        if args.anomaly_type == 'MISO':
            indicies = np.argwhere(train_labels == str(args.anomaly_class))

            mask_train  = np.invert(train_labels == str(args.anomaly_class))
        else: 
            indicies = np.argwhere(train_labels != str(args.anomaly_class))

            mask_train  = train_labels == str(args.anomaly_class)

        train_data= train_data[mask_train]
        train_labels = train_labels[mask_train]
        train_masks = train_masks[mask_train]

    if args.limit is not None:
        train_data = train_data[:args.limit,...]
        train_labels = train_labels[:args.limit,...]
        train_masks = train_masks[:args.limit,...]

    print('Train data', np.unique(train_labels), train_labels.shape)
    print('Test data', np.unique(test_labels), test_labels.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return (train_dataset,train_data, train_labels, test_data, test_labels, test_masks)

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
