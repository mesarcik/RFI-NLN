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
         train_data, 
         ae_train_data, 
         train_masks, 
         ae_train_masks,
         train_labels,
         ae_train_labels,
         test_data, 
         test_labels, 
         test_masks) = data


    elif cmd_input.args.data == 'LOFAR':
        (unet_train_dataset,
            train_data, 
            train_labels, 
            train_masks, 
            ae_train_dataset,
            ae_train_data, 
            ae_train_labels, 
            test_data, 
            test_labels, 
            test_masks) = load_lofar(cmd_input.args)


    print(" __________________________________ \n Latent dimensionality {}".format(
                                               cmd_input.args.latent_dim))
    print(" __________________________________ \n Save name {}".format(
                                               cmd_input.args.model_name))
    print(" __________________________________ \n")

    train_unet(unet_train_dataset,
               train_data,
               train_labels, 
               train_masks,
               test_data,
               test_labels, 
               test_masks, 
               cmd_input.args)

    train_ae(ae_train_dataset,
             ae_train_data,
             ae_train_labels,
             test_data,
             test_labels, 
             test_masks, 
             cmd_input.args)

    train_dae(ae_train_dataset,
             ae_train_data,
             ae_train_labels,
             test_data,
             test_labels, 
             test_masks, 
             cmd_input.args)

    train_ganomaly(ae_train_dataset,
             ae_train_data,
             ae_train_labels,
             test_data,
             test_labels, 
             test_masks, 
             cmd_input.args)

    #train_vae(train_dataset,train_data,train_labels,test_data,test_labels, test_masks, cmd_input.args)
    #train_aae(train_dataset,train_data,train_labels,test_data,test_labels, test_masks, cmd_input.args)

if __name__ == '__main__':
    main()
