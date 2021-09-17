import tensorflow as tf
import numpy as np
import copy
from .patches import get_patches

#TODO this certainly is not the most efficient, pythonic, nor tensorflowic way of doing things


def random_rotation(images, masks = None):
    """
        Applies random discrete rotation based augmentations to the test data and to their masks (if applicable)
        The discrete rotations are [0,90,180,270,360] degrees

        Parameters
        ----------
        images (np.array) 
        mask_images (optional, np.array) 
        
        Returns
        -------
        np.array, (optional) np.array

    """
    r_images = copy.deepcopy(images)

    if masks is not None: 
        masks = np.expand_dims(masks,axis=-1)
        r_masks = copy.deepcopy(masks)

    for i in range(images.shape[0]):
        k = np.random.randint(0,5)
        rot_cond = tf.less(tf.random.uniform([], 0, 1.0), .5)
        r_images[i,...] = tf.cond(rot_cond,
                                  lambda: tf.image.rot90(images[i,...],k), 
                                  lambda: images[i,...])
        if masks is not None:
            r_masks[i,...]  = tf.cond(rot_cond, 
                                      lambda: tf.image.rot90(masks[i,...],k), 
                                      lambda: masks  [i,...])

    if masks is not None:
        return np.concatenate([images,r_images],axis=0), np.concatenate([masks,r_masks],axis=0)[...,0]
    else:
        return np.concatenate([images,r_images],axis=0)



def random_crop(images, crop_size):
    """
        Applies central crop and then random crop based augmentations to the test data and to their masks (if applicable)

        Parameters
        ----------
        images (np.array) 
        crop_size (list-like)
        
        Returns
        -------
        np.array

        Raises:
        -------
        ValueError: If the shape of `image` is incompatible with the `masks` 

    """
    assert(images.shape[1] *0.9 > crop_size[0], ValueError, 
                        'X dimension of crop must be greater than X of crop')
    assert(images.shape[2]*0.9 > crop_size[1], ValueError, 
                        'Y dimension of crop must be greater than Y of crop')
    
    images = tf.image.central_crop(images ,0.8).numpy()
    images_,_  = get_patches(images, 
                            np.zeros(len(images)),
                            (1, int(1.7*crop_size[0]), int(1.7*crop_size[1]), 1),
                            (1, int(1.7*crop_size[0]), int(1.7*crop_size[1]), 1),
                            (1,1,1,1),
                            'VALID') 

    r_images = np.empty([images_.shape[0], crop_size[0], crop_size[1], images_.shape[-1]])

    for i in range(images_.shape[0]):
        x_offset = np.random.randint(0, images_.shape[1] -  crop_size[0] + 1)
        y_offset = np.random.randint(0, images_.shape[2] -  crop_size[1] + 1)

        r_images[i,...] = tf.image.crop_to_bounding_box(images_[i,...],
                                                        x_offset,
                                                        y_offset,
                                                        crop_size[0],
                                                        crop_size[1])

    return r_images.astype('uint8')

