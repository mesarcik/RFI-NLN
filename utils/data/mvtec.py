import pickle
import numpy as np
import os
from tqdm import tqdm 
from imageio import imread
from glob import glob

def get_mvtec_images(SIMO_class, directory):
    """"
        Walks through MVTEC dataset and returns data in the same structure as tf
        
        SIMO_class (str): Anomalous class 
        directory (str): Directory where MVTecAD dataset resides
    """

    train_images, test_images, train_labels ,test_labels, test_masks  = [], [], [], [], []
    
    # if the training dataset has already been created then return that
    if os.path.exists(os.path.join(directory,'{}.pickle'.format(SIMO_class))):
        print(os.path.join(directory,'{}.pickle'.format(SIMO_class)) + ' Loading')
        with open('{}/{}.pickle'.format(directory,SIMO_class),'rb') as f:
            return pickle.load(f)

    print('Creating data for {}'.format(SIMO_class))
    for f in tqdm(glob("{}/{}/train/good/*.png".format(directory,SIMO_class))): 
        img = imread(f)
        train_images.append(img) 
        train_labels.append('non_anomalous')

    for f in tqdm(glob("{}/{}/test/*/*.png".format(directory,SIMO_class))): 
        img = imread(f)
        test_images.append(img)
        if 'good' in f: 
            test_labels.append('non_anomalous')
            test_masks.append(np.zeros([img.shape[0],
                                       img.shape[1]]))
        else:
            test_labels.append(SIMO_class)
            f_ = f.split('/')
            img_mask = imread(os.path.join(f_[0], 
                                           f_[1], 
                                           f_[2], 
                                           f_[3], 
                                           'ground_truth', 
                                           f_[5], 
                                           f_[6].split('.')[0] + '_mask.png'))
            test_masks.append(img_mask)
    
    train_images, train_labels = np.array(train_images), np.array(train_labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)
    test_masks = np.array(test_masks)  

    if (('grid' in SIMO_class) or
        ('screw' in SIMO_class) or 
        ('zipper' in SIMO_class)): 
        test_images = np.expand_dims(test_images,axis=-1)
        train_images = np.expand_dims(train_images,axis=-1)

    pickle.dump(((train_images, train_labels),
                (test_images, test_labels, test_masks)),
                open('{}/{}.pickle'.format(directory,SIMO_class), 'wb'), protocol=1)

    return (train_images, train_labels), (test_images, test_labels, test_masks)
