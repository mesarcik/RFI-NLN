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

def load_lofar(args):            
    """
        Load data from lofar 

    """
    data, masks = np.load(args.data_path, allow_pickle=True)
    data = resize(np.absolute(data[...,0:1]).astype('float32'), (sizes[args.anomaly_class], 
                                                                sizes[args.anomaly_class])) #TODO 
    masks = resize(masks[...,0:1].astype('int'), (sizes[args.anomaly_class], 
                                  sizes[args.anomaly_class])).astype('bool')


    # TODO: determine where to place the normalisation
    data = np.nan_to_num(np.log(data),nan=0)
    data = process(data, per_image=False)


    if args.limit is not None:
        data = data[:args.limit,...]

    if args.patches:
        p_size = (1,args.patch_x, args.patch_y, 1)
        s_size = (1,args.patch_stride_x, args.patch_stride_y, 1)
        rate = (1,1,1,1)

        data_patches = get_patches(data, None, p_size,s_size,rate,'VALID')
        mask_patches = get_patches(masks, None, p_size,s_size,rate,'VALID')

        labels = np.empty(len(data_patches), dtype='object')
        labels[np.any(mask_patches, axis=(1,2,3))] = args.anomaly_class
        labels[np.invert(np.any(mask_patches, axis=(1,2,3)))] = 'normal'

        (train_data, test_data, 
         train_labels,test_labels, 
         train_masks, test_masks) = train_test_split(data_patches, 
                                                     labels, 
                                                     mask_patches,
                                                     test_size=0.25, 
                                                     random_state=42)

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
            test_masks)
