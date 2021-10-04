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
        (unet_train_dataset,
         ae_train_dataset,
         unet_train_data, 
         ae_train_data, 
         unet_train_masks, 
         ae_train_masks,
         unet_train_labels,
         ae_train_labels,
         unet_test_data, 
         unet_test_labels, 
         unet_test_masks) = data


    elif cmd_input.args.data == 'LOFAR':
        path = 'data/datasets/LOFAR_AE_dataset_22-09-2021_small.pkl' 
        ae_data  = load_lofar(cmd_input.args, path, unet=False)
        (train_dataset,
         train_data, 
         train_labels,
         test_data,
         test_labels, test_masks) = ae_data

        path = 'data/datasets/LOFAR_UNET_dataset_22-09-2021_small.pkl' 
        unet_data  = load_lofar(cmd_input.args, path, unet=True)
        (unet_train_dataset,
            unet_train_data, 
            unet_train_masks,
            unet_train_labels,
            unet_test_data,
            unet_test_labels, unet_test_masks) = unet_data


    print(" __________________________________ \n Latent dimensionality {}".format(
                                               cmd_input.args.latent_dim))
    print(" __________________________________ \n Save name {}".format(
                                               cmd_input.args.model_name))
    print(" __________________________________ \n")

    train_unet(unet_train_dataset,
               unet_train_data,
               unet_train_labels, 
               unet_train_masks,
               unet_test_data,
               unet_test_labels, 
               unet_test_masks, 
               cmd_input.args)

    train_ae(ae_train_dataset,
             ae_train_data,
             ae_train_labels,
             unet_test_data,
             unet_test_labels, 
             unet_test_masks, 
             cmd_input.args)
    #train_dae(train_dataset,train_data,train_labels,test_data,test_labels, test_masks, cmd_input.args)
    #train_ganomaly(train_dataset,train_data,train_labels,test_data,test_labels,test_masks, cmd_input.args)
    #train_vae(train_dataset,train_data,train_labels,test_data,test_labels, test_masks, cmd_input.args)
    #train_aae(train_dataset,train_data,train_labels,test_data,test_labels, test_masks, cmd_input.args)

if __name__ == '__main__':
    main()
