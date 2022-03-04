import tensorflow as tf
import numpy as np
import copy
import random
from tqdm import tqdm
#from hera_sim import rfi 
from model_config import BUFFER_SIZE,BATCH_SIZE
from sklearn.model_selection import train_test_split
from utils.flagging import flag_data
from utils.data import (get_lofar_data, 
                        get_hera_data,
                        get_hide_data,
                        process,
                        get_patches)

def load_hide(args):            
    """
        Load data from hide 
    """

    (train_data, train_masks, 
            test_data, test_masks) = get_hide_data(args)

    test_data = np.clip(np.fabs(test_data), 0, 500)
    test_data -= np.amin(test_data)
    test_data = process(test_data, per_image=False)

    #mi, ma = np.min(data), np.max(data)
    #output = (data - mi)/(ma -mi)
    #output = output.astype('float32')

    train_data = np.clip(np.fabs(train_data), 0, 500)
    train_data -= np.amin(train_data)
    train_data = process(train_data, per_image=False)

    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]
        test_indx = np.random.permutation(len(test_data))[:args.limit]

        train_data  = train_data [train_indx]
        train_masks = train_masks[train_indx]
        #test_data   = test_data  [test_indx]
        #test_masks  = test_masks [test_indx]

    test_masks_orig = copy.deepcopy(test_masks)
    if args.rfi_threshold is not None:
        test_masks = flag_data(test_data,args)
        train_masks = flag_data(train_data,args)
        test_masks = np.expand_dims(test_masks,axis=-1) 
        train_masks = np.expand_dims(train_masks,axis=-1) 

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
        
        (ae_train_data, 
         ae_train_labels, 
         _)  =  np.load('/home/mmesarcik/data/HERA/HERA_02-03-2022_free.pkl')

        #ae_train_data  = train_data[np.invert(np.any(train_masks, axis=(1,2,3)))]
        #ae_train_labels = train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))]

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
            test_masks,
            test_masks_orig)


def load_hera(args):
    """
        Load data from hera

    """
    (train_data, test_data, 
         train_labels, test_labels, 
         train_masks, test_masks) = get_hera_data(args)


    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]
        train_data  = train_data [train_indx]
        train_masks = train_masks[train_indx]
        train_labels = train_labels[train_indx]

    test_masks_orig = copy.deepcopy(test_masks)
    if args.rfi_threshold is not None:
        test_masks = flag_data(test_data,args)
        train_masks = flag_data(train_data,args)
        test_masks = np.expand_dims(test_masks,axis=-1) 
        train_masks = np.expand_dims(train_masks,axis=-1) 

    test_data[test_data==0] = 0.001 # to make log normalisation happy
    test_data = np.nan_to_num(np.log(test_data),nan=0)

    train_data[train_data==0] = 0.001 # to make log normalisation happy
    train_data = np.nan_to_num(np.log(train_data),nan=0)

    test_data =  process(test_data, per_image=False).astype(np.float16)
    train_data = process(train_data, per_image=False).astype(np.float16)


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

        #(ae_train_data, 
        # ae_train_labels,
        # ae_train_masks)  =  np.load('/data/mmesarcik/hera/HERA_03-03-2022_free.pkl',allow_pickle=True)

        #ae_train_data[ae_train_data==0] = 0.001 # to make log normalisation happy
        #ae_train_data = np.nan_to_num(np.log(ae_train_data),nan=0)
        #ae_train_data = process(ae_train_data, per_image=False).astype(np.float16)
        #ae_train_data = get_patches(ae_train_data, None, p_size,s_size,rate,'VALID')

        #ae_train_masks = get_patches(ae_train_masks, None, p_size,s_size,rate,'VALID').astype(np.bool)

        #train_labels = np.empty(len(ae_train_data), dtype='object')
        #train_labels[np.any(ae_train_masks, axis=(1,2,3))] = args.anomaly_class
        #train_labels[np.invert(np.any(ae_train_masks, axis=(1,2,3)))] = 'normal'


        ae_train_data  = train_data[np.invert(np.any(train_masks, axis=(1,2,3)))]
        ae_train_labels = train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))]

    ae_train_data = ae_train_data.astype(np.float16) 
    train_data = train_data.astype(np.float16) 
    test_data = test_data.astype(np.float16) 

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

    train_data, train_masks, test_data, test_masks = get_lofar_data('/home/mmesarcik/data/LOFAR/uncompressed', args)


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
            test_masks,
            test_masks)

