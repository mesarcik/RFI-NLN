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
    unet_path = '/data/mmesarcik/hera/HERA/HERA_6_27-09-2021_UNET.pkl'
    unet_data, unet_labels, unet_masks, _ =  np.load(unet_path, allow_pickle=True)

    ae_path = '/data/mmesarcik/hera/HERA/HERA_0_27-09-2021_AE.pkl'
    ae_data, ae_labels, ae_masks, _ =  np.load(ae_path, allow_pickle=True)


    unet_data = np.sqrt(unet_data[...,0]**2 + unet_data[...,1]**2)
    unet_data = np.expand_dims(unet_data, axis=-1)
#    mi,ma = np.min(unet_data), np.max(unet_data)
#    unet_data = (unet_data - mi)/(ma -mi)
    unet_data = process(unet_data, per_image=True)#unet_data.astype('float32')

    ae_data = np.sqrt(ae_data[...,0]**2 + ae_data[...,1]**2)
    ae_data = np.expand_dims(ae_data, axis=-1)
#    ae_data = (ae_data - mi)/(ma -mi)
    ae_data = process(ae_data, per_image=True)

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
    _unet_test_labels = copy.deepcopy(unet_test_labels)
    _unet_train_labels = copy.deepcopy(unet_train_labels)
    for i,(train_label,test_label) in enumerate(zip(unet_train_labels,unet_test_labels)):
        if args.anomaly_class in test_label: 
            _unet_test_labels[i] = args.anomaly_class
        else:
            _unet_test_labels[i] = 'normal'

        if args.anomaly_class in train_label: 
            _unet_train_labels[i] = args.anomaly_class
        else:
            _unet_train_labels[i] = 'normal'
        #if 'rfi_stations' in train_label:
        #    _unet_train_labels[i] = 'rfi_stations' 

    unet_test_labels = np.array(_unet_test_labels)
    unet_train_labels = np.array(_unet_train_labels)
    ae_labels = np.array(['normal']*len(ae_labels))
    #remove class from training data
    #unet_train_data = unet_train_data[unet_train_labels != 'rfi_stations']
    #unet_train_masks = unet_train_masks[unet_train_labels != 'rfi_stations']
    #unet_train_labels = unet_train_labels[unet_train_labels != 'rfi_stations']

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
        ae_masks= ae_masks[:args.limit,...]
        ae_labels = ae_labels[:args.limit,...]

    if args.patches:
        data  = get_patched_dataset(unet_train_data,
                                    unet_train_labels,
                                    unet_test_data,
                                    unet_test_labels,
                                    unet_test_masks,
                                    p_size = (1,args.patch_x, args.patch_y, 1),
                                    s_size = (1,args.patch_stride_x, args.patch_stride_y, 1),
                                    central_crop=False)
        (unet_train_data,
         unet_train_labels, 
         unet_test_data, 
         unet_test_labels, 
         unet_test_masks) = data

        unet_train_masks_patches, _ = get_patches(unet_train_masks,
                                                  unet_train_labels,
                                                  (1,args.patch_x, args.patch_y, 1),
                                                  (1,args.patch_stride_x, args.patch_stride_y, 1),
                                                  (1,1,1,1),
                                                  'VALID')
        unet_train_masks = unet_train_masks_patches
        ae_data, ae_labels = get_patches(ae_data,
                                         ae_labels,
                                         (1,args.patch_x, args.patch_y, 1),
                                         (1,args.patch_stride_x, args.patch_stride_y, 1),
                                         (1,1,1,1),
                                         'VALID')
        

    #if args.rotate:
    #    train_images = random_rotation(train_images) 
    #    train_images, train_labels, test_images, test_labels, test_masks = data

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

def add_HERA_rfi(_test_data, _test_masks, args, test_labels =None):
#    args.rfi 
    test_data = copy.deepcopy(_test_data) 
    test_masks = copy.deepcopy(_test_masks) 
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
        train_masks = get_patches(train_masks, None, p_size,s_size,rate,'VALID')
        test_masks= get_patches(test_masks.astype('int') , None, p_size,s_size,rate,'VALID').astype('bool')

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

    ae_train_data = ae_train_data.astype('float32') 
    train_data = train_data.astype('float32') 
    test_data = test_data.astype('float32') 

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

