import numpy as np 
import tensorflow as tf
from data import *
from utils import cmd_input 
from architectures import *  

def main():
    """
        Reads data and cmd arguments and trains models
    """

    if cmd_input.args.data == 'HERA':
        data  = load_hera(cmd_input.args)
        (train_dataset,
            train_images, 
            train_masks,
            train_labels,
            test_images,
            test_labels, test_masks) = data

    elif cmd_input.args.data == 'LOFAR':
        path = 'data/datasets/LOFAR_AE_dataset_17-09-2021.pkl' 
        ae_data  = load_lofar(cmd_input.args, path)
        (train_dataset,
         train_images, 
         train_labels,
         test_images,
         test_labels, test_masks) = ae_data

        path = 'data/datasets/LOFAR_UNET_dataset_17-09-2021.pkl' 
        unet_data  = load_lofar(cmd_input.args, path)
        (unet_train_dataset,
            unet_train_images, 
            unet_train_masks,
            unet_train_labels,
            unet_test_images,
            unet_test_labels, unet_test_masks) = unet_data




    print(" __________________________________ \n Latent dimensionality {}".format(
                                               cmd_input.args.latent_dim))
    print(" __________________________________ \n Save name {}".format(
                                               cmd_input.args.model_name))
    print(" __________________________________ \n")

    train_unet(unet_train_dataset,unet_train_images,unet_train_labels, unet_train_masks,unet_test_images,unet_test_labels, unet_test_masks, cmd_input.args)
    train_ae(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)
    train_dae(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)
    train_ganomaly(train_dataset,train_images,train_labels,test_images,test_labels,test_masks, cmd_input.args)
    train_vae(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)
    train_aae(train_dataset,train_images,train_labels,test_images,test_labels, test_masks, cmd_input.args)

if __name__ == '__main__':
    main()
