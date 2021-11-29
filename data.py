import tensorflow as tf
import numpy as np
import copy
import random
from tqdm import tqdm
#from hera_sim import rfi 
from model_config import BUFFER_SIZE,BATCH_SIZE
from sklearn.model_selection import train_test_split
from utils.data import (get_lofar_data, 
                        _random_crop,
                        process,
                        rgb2gray,
                        get_patched_dataset,
                        get_patches,
                        random_rotation,
                        random_crop,
                        resize,
                        sizes)

def load_hera(args):
    """
        Load data from hera

    """
    data, labels, masks, _ =  np.load(args.data_path, allow_pickle=True)

    data = np.expand_dims(data, axis=-1)
    data = process(data, per_image=True).astype(np.float16)
    masks = np.expand_dims(masks,axis=-1)

    (train_data, test_data, 
     train_labels, test_labels, 
     train_masks, test_masks) = train_test_split(data, 
                                                 labels, 
                                                 masks,
                                                 test_size=0.25, 
                                                 random_state=42)
    if args.percentage_anomaly is not None:
        _m = np.random.random(train_masks.shape)<args.percentage_anomaly
        train_masks[_m] = np.invert(train_masks[_m])

        _m = np.random.random(test_masks.shape)<args.percentage_anomaly
        test_masks_orig = test_masks
        test_masks[_m] = np.invert(test_masks[_m])

    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]

        train_data  = train_data [train_indx]
        train_masks = train_masks[train_indx]

    if args.patches:
        p_size = (1,args.patch_x, args.patch_y, 1)
        s_size = (1,args.patch_stride_x, args.patch_stride_y, 1)
        rate = (1,1,1,1)

        train_data = get_patches(train_data, None, p_size,s_size,rate,'VALID')
        test_data = get_patches(test_data, None, p_size,s_size,rate,'VALID')
        train_masks = get_patches(train_masks, None, p_size,s_size,rate,'VALID').astype(np.bool)
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
    ae_train_dataset = tf.data.Dataset.from_tensor_slices(ae_train)data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    if rfi is None:
        return (unet_train_dataset,
                unet_train_data, 
                unet_train_labels,
                unet_train_masks, 
                ae_train_dataset,
                ae_data, 
                ae_labels,
                test_data, 
                test_labels, 
                test_masks,
                test_masks)
    else:
        return (unet_train_dataset,
                unet_train_data, 
                unet_train_labels,
                unet_train_masks, 
                ae_train_dataset,
                ae_data, 
                ae_labels,
                test_data, 
                test_labels, 
                test_masks,
                test_masks_orig)


def add_HERA_rfi(_data, _masks, args, expand=True, test_labels =None):
    """
        add synthetic RFI to data
        _data (np.array): the data to add rfi to
        _masks (np.array): the masks to add rfi to
        args (Namespace): utils.cmd_args
        expand (bool): expand RFI masks (add noise) or remove masks 

    """
#    args.rfi 
    test_data = copy.deepcopy(_data) 
    test_masks = copy.deepcopy(_masks) 
    fqs = np.linspace(.1,.2,test_data.shape[1],endpoint=False)
    lsts = np.linspace(0,2*np.pi,test_data.shape[1], endpoint=False)
    for i in tqdm(range(len(test_data))): 
        if random.randint(0,1):
            stations = np.absolute(rfi.rfi_stations(fqs, lsts)/200)
            test_data[i,...,0] += args.rfi*stations
            test_masks[i,...,0] = np.logical_or(test_masks[i,...,0].astype('bool'),
                                                stations>0)
            if test_labels is not None:
                test_labels[i] = 'rfi'
        if random.randint(0,1):
            impulse = np.absolute(rfi.rfi_impulse(fqs, 
                                                  lsts, 
                                                  impulse_strength=300, 
                                                  impulse_chance=.05))
            test_data[i,...,0] += args.rfi*impulse 
            test_masks[i,...,0] = np.logical_or(test_masks[i,...,0].astype('bool'),
                                                impulse>0)
            if test_labels is not None:
                test_labels[i] = 'rfi'
        if random.randint(0,1):
            dtv = np.absolute(rfi.rfi_dtv(fqs, 
                                          lsts, 
                                          dtv_strength=500,
                                          dtv_chance=.1))
            test_data[i,...,0] += args.rfi*dtv
            test_masks[i,...,0] = np.logical_or(test_masks[i,...,0].astype('bool'),
                                                dtv>0)
            if test_labels is not None:
                test_labels[i] = 'rfi'

        
    return test_data, test_labels, test_masks 

def load_lofar(args):            
    """
        Load data from lofar 

    """

    train_data, train_masks, test_data, test_masks = get_lofar_data('/data/mmesarcik/LOFAR/uncompressed', args)


    # add RFI to the data masks and labels 
    if args.rfi != 0:
        test_data, _,  test_masks = add_HERA_rfi(test_data, 
                                              test_masks, 
                                              args)
    test_data[test_data==0] = 0.001 # to make log normalisation happy
    test_data = np.nan_to_num(np.log(test_data),nan=0)
    test_data = process(test_data, per_image=False)

    train_data[train_data==0] = 0.001 # to make log normalisation happy
    train_data = np.nan_to_num(np.log(train_data),nan=0)
    train_data = process(train_data, per_image=False)

    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]
        test_indx = np.random.permutation(len(test_data))[:args.limit]

        train_data  = train_data [train_indx]
        train_masks = train_masks[train_indx]
        #test_data   = test_data  [test_indx]
        #test_masks  = test_masks [test_indx]

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

        #test_data  =  test_data[np.invert(np.any(test_masks, axis=(1,2,3)))]
        #test_labels = test_labels[np.invert(np.any(test_masks, axis=(1,2,3)))]
        #test_masks = test_masks[np.invert(np.any(test_masks, axis=(1,2,3)))]

    ae_train_data = ae_train_data.astype(np.float16) 
    train_data = train_data.astype(np.float16) 
    test_data = test_data.astype(np.float16) 

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
            test_masks)

