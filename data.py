import tensorflow as tf
import numpy as np
import copy
import random
from tqdm import tqdm
from model_config import BUFFER_SIZE,BATCH_SIZE
from sklearn.model_selection import train_test_split
from utils.flagging import flag_data
from utils.data import (get_lofar_data, 
                        get_hera_data,
                        process,
                        get_patches)


def load_hera(args):
    """
        Load data from hera

    """
    (train_data, test_data, 
         train_masks, test_masks) = get_hera_data(args)


    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]
        train_data  = train_data [train_indx]
        train_masks = train_masks[train_indx]

    test_masks_orig = copy.deepcopy(test_masks)
    if args.rfi_threshold is not None:
        test_masks = flag_data(test_data,args)
        train_masks = flag_data(train_data,args)
        test_masks = np.expand_dims(test_masks,axis=-1) 
        train_masks = np.expand_dims(train_masks,axis=-1) 

    _max = np.mean(test_data[np.invert(test_masks)])+4*np.std(test_data[np.invert(test_masks)])
    _min =  np.absolute(np.mean(test_data[np.invert(test_masks)]) - np.std(test_data[np.invert(test_masks)]))
    test_data = np.clip(test_data, _min, _max)
    test_data = np.log(test_data)
    test_data =  process(test_data, per_image=False)#.astype(np.float16)

    _max = np.mean(train_data[np.invert(train_masks)])+4*np.std(train_data[np.invert(train_masks)])
    _min = np.absolute(np.mean(train_data[np.invert(train_masks)])-np.std(train_data[np.invert(train_masks)]))
    train_data = np.clip(train_data, _min, _max)
    train_data = np.log(train_data)
    train_data = process(train_data, per_image=False)#.astype(np.float16)

    if args.patches:
        p_size = (1,args.patch_x, args.patch_y, 1)
        s_size = (1,args.patch_stride_x, args.patch_stride_y, 1)
        rate = (1,1,1,1)

        train_data = get_patches(train_data, None, p_size,s_size,rate,'VALID')
        train_masks = get_patches(train_masks, None, p_size,s_size,rate,'VALID').astype(np.bool)

        test_data = get_patches(test_data, None, p_size,s_size,rate,'VALID')
        test_masks= get_patches(test_masks.astype('int') , None, p_size,s_size,rate,'VALID').astype(np.bool)

        test_masks_orig = get_patches(test_masks_orig.astype('int') , None, p_size,s_size,rate,'VALID').astype(np.bool)

        train_labels = np.empty(len(train_data), dtype='object')
        train_labels[np.any(train_masks, axis=(1,2,3))] = args.anomaly_class
        train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))] = 'normal'

        test_labels = np.empty(len(test_data), dtype='object')
        test_labels[np.any(test_masks, axis=(1,2,3))] = args.anomaly_class
        test_labels[np.invert(np.any(test_masks, axis=(1,2,3)))] = 'normal'

        ae_train_data  = train_data[np.invert(np.any(train_masks, axis=(1,2,3)))]
        ae_train_labels = train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))]

    unet_train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    ae_train_dataset = tf.data.Dataset.from_tensor_slices(ae_train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


    return (unet_train_dataset,
            train_data, 
            train_labels,
            train_masks, 
            ae_train_dataset,
            ae_train_data, 
            ae_train_labels,
            test_data, 
            test_labels, 
            test_masks,
            test_masks_orig)


def load_lofar(args):            
    """
        Load data from lofar 

    """

    train_data, train_masks, test_data, test_masks = get_lofar_data(args)


    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]
        test_indx = np.random.permutation(len(test_data))[:args.limit]

        train_data  = train_data [train_indx]
        train_masks = train_masks[train_indx]
        #test_data   = test_data  [test_indx]
        #test_masks  = test_masks [test_indx]

    if args.rfi_threshold is not None:
        train_masks = flag_data(train_data,args)
        train_masks = np.expand_dims(train_masks,axis=-1) 

    _max = np.mean(test_data[np.invert(test_masks)])+20*np.std(test_data[np.invert(test_masks)])
    _min =  np.absolute(np.mean(test_data[np.invert(test_masks)]) - np.std(test_data[np.invert(test_masks)]))

    test_data = np.clip(test_data,_min,_max) 
    test_data = np.log(test_data)
    test_data = process(test_data, per_image=False)

    _max = np.mean(train_data[np.invert(train_masks)])+20*np.std(train_data[np.invert(train_masks)])
    _min =  np.absolute(np.mean(train_data[np.invert(train_masks)]) - np.std(train_data[np.invert(train_masks)]))

    train_data = np.clip(train_data, _min,_max)
    train_data = np.log(train_data)
    train_data = process(train_data, per_image=False)

    if args.patches:
        p_size = (1,args.patch_x, args.patch_y, 1)
        s_size = (1,args.patch_stride_x, args.patch_stride_y, 1)
        rate = (1,1,1,1)

        train_data = get_patches(train_data, None, p_size,s_size,rate,'VALID')
        test_data = get_patches(test_data, None, p_size,s_size,rate,'VALID')
        train_masks = get_patches(train_masks, None, p_size,s_size,rate,'VALID').astype(np.bool)
        test_masks= get_patches(test_masks.astype('int') , None, p_size,s_size,rate,'VALID').astype(np.bool)

        train_labels = np.empty(len(train_data), dtype='object')
        train_labels[np.any(train_masks, axis=(1,2,3))] = args.anomaly_class
        train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))] = 'normal'

        test_labels = np.empty(len(test_data), dtype='object')
        test_labels[np.any(test_masks, axis=(1,2,3))] = args.anomaly_class
        test_labels[np.invert(np.any(test_masks, axis=(1,2,3)))] = 'normal'

        ae_train_data  = train_data[np.invert(np.any(train_masks, axis=(1,2,3)))]
        ae_train_labels = train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))]

    unet_train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE,seed=42).batch(BATCH_SIZE)
    ae_train_dataset = tf.data.Dataset.from_tensor_slices(ae_train_data).shuffle(BUFFER_SIZE,seed=42).batch(BATCH_SIZE)

    return (unet_train_dataset,
            train_data, 
            train_labels, 
            train_masks, 
            ae_train_dataset,
            ae_train_data, 
            ae_train_labels, 
            test_data, 
            test_labels, 
            test_masks,
            test_masks)

