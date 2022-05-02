import numpy as np 
import tensorflow as tf
from data import *
from utils import args 
from architectures import *  

def main():
    """
        Reads data and cmd arguments and trains models
    """
    if args.args.data == 'HERA':
        data  = load_hera(args.args)
    elif args.args.data == 'LOFAR':
        data = load_lofar(args.args)
    elif args.args.data == 'HIDE':
        data = load_hide(args.args)

    (unet_train_dataset, train_data, train_labels, train_masks, 
     ae_train_dataset, ae_train_data, ae_train_labels,
     test_data, test_labels, test_masks,test_masks_orig) = data

    print(" __________________________________ \n Save name {}".format(
                                               args.args.model_name))
    print(" __________________________________ \n")
    
    if args.args.model == 'UNET':
        train_unet(unet_train_dataset, train_data, train_labels, train_masks, test_data, test_labels, test_masks, test_masks_orig, args.args)

    if args.args.model == 'RNET':
        train_rnet(unet_train_dataset, train_data, train_labels, train_masks, test_data, test_labels, test_masks, test_masks_orig, args.args)

    elif args.args.model == 'DKNN':
        train_resnet(ae_train_dataset, ae_train_data, ae_train_labels, test_data, test_labels, test_masks, test_masks_orig, args.args)

    elif args.args.model == 'AE':
        train_ae(ae_train_dataset, ae_train_data, ae_train_labels, test_data, test_labels, test_masks, test_masks_orig, args.args)

    elif args.args.model == 'DAE':
        train_dae(ae_train_dataset, ae_train_data, ae_train_labels, test_data, test_labels, test_masks, test_masks_orig,args.args)

    elif args.args.model == 'AE-res':
        train_ganomaly(ae_train_dataset, ae_train_data, ae_train_labels, test_data, test_labels, test_masks, test_masks_orig, args.args)

    elif args.args.model == 'VAE':
         train_vae(ae_train_dataset, ae_train_data, ae_train_labels, test_data, test_labels, test_masks, test_masks_orig, args.args)

    elif args.args.model == 'AAE':
         train_ae(ae_train_dataset, ae_train_data, ae_train_labels, test_data, test_labels, test_masks, test_masks_orig, args.args)

if __name__ == '__main__':
    main()
