import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (roc_curve,
                             auc, 
                             f1_score,
                             accuracy_score, 
                             average_precision_score, 
                             jaccard_score,
                             roc_auc_score, 
                             precision_recall_curve)
from inference import infer, get_error
from utils.data import *
from utils.metrics import nln, get_nln_errors
from matplotlib import pyplot as plt

import time

def evaluate_performance(model,
                         train_images,
                         test_images,
                         test_labels,
                         test_masks,
                         test_masks_orig,
                         model_type,
                         neighbour,
                         args):

    """
        Calculate accuracy metrics for MVTEC AD as reported by the paper

        Parameters
        ----------
        model (tf.keras.Model): the model used
        train_images (np.array): non-anomalous images from train set 
        test_images (np.array): testing images
        test_labels (np.array): labels of testing images
        test_masks (np.array): ground truth masks for the testing images
        model_type (str): the type of model (AE,VAE,...)
        args (Namespace): the argumenets from cmd_args

        Returns
        -------
        seg_auc (float32): segmentation auroc 
        seg_auc_nln (float32): segmentation auroc using NLN
        dists_auroc (float32): detection auroc using dists
        seg_dists_auroc (float32): segmentation auroc using dists
        seg_prc (float32): segmetnation auprc 
        seg_prc_nln (float32): segmentation auprc using nln
        seg_iou (float32): iou score for reconstruction
        seg_iou_nln (float32): iou score for nln
    """
    # Get output from model #TODO: do we want to normalise?
    test_data_recon = patches.reconstruct(test_images, args)
    test_masks_recon = patches.reconstruct(test_masks, args)
    test_masks_orig_recon = patches.reconstruct(test_masks_orig, args)

    if model_type == 'AOFlagger':
        (ao_auroc, true_auroc, 
         ao_auprc, true_auprc,      
         ao_f1, true_f1) = get_metrics(test_masks_recon, 
                                                   test_masks_orig_recon, 
                                                   test_masks_recon)
        return (ao_auroc, true_auroc, 
         ao_auprc, true_auprc,      
         ao_f1, true_f1,
         -1, -1, 
         -1, -1,      
         -1, -1,
         -1, -1, 
         -1, -1,      
         -1, -1,
         [-1], [-1], 
         [-1], [-1],      
         [-1], [-1])



    if model_type =='UNET' or model_type =='RNET' or model_type =='RFI_NET':
        x_hat = infer(model[0], test_images, args, 'AE')
        x_hat_recon = patches.reconstruct(x_hat, args)
        x_hat_recon[x_hat_recon==np.inf] = np.finfo(x_hat_recon.dtype).max

        (unet_ao_auroc, unet_true_auroc, 
         unet_ao_auprc, unet_true_auprc,      
         unet_ao_f1, unet_true_f1) = get_metrics(test_masks_recon, 
                                                   test_masks_orig_recon, 
                                                   x_hat_recon)


        fig, axs = plt.subplots(10,3, figsize=(10,7))
        axs[0,0].set_title('Inp',fontsize=5)
        axs[0,1].set_title('Mask',fontsize=5)
        axs[0,2].set_title('Recon {}'.format(unet_ao_auroc),fontsize=5)

        for i in range(10):
            r = np.random.randint(len(test_data_recon))
            axs[i,0].imshow(test_data_recon[r,...,0].astype(np.float32), interpolation='nearest', aspect='auto')
            axs[i,1].imshow(test_masks_recon[r,...,0].astype(np.float32), interpolation='nearest', aspect='auto')
            axs[i,2].imshow(x_hat_recon[r,...,0].astype(np.float32), interpolation='nearest', aspect='auto')
        plt.savefig('outputs/{}/{}/{}/neighbours.png'.format(model_type,
                                                       args.anomaly_class,
                                                       args.model_name), dpi=300)
        return (unet_ao_auroc, unet_true_auroc, 
         unet_ao_auprc, unet_true_auprc,      
         unet_ao_f1, unet_true_f1,
         -1, -1, 
         -1, -1,      
         -1, -1,
         -1, -1, 
         -1, -1,      
         -1, -1,
         [-1], [-1], 
         [-1], [-1],      
         [-1], [-1])

    elif model_type =='DKNN':
        z_train = infer(model[0], train_images, args, 'DKNN')
        z_test = infer(model[0], test_images, args, 'DKNN')


        neighbours_dist, _, _, _ =  nln(z_train, z_test, None, 'knn', 2, -1)

        dists_recon = get_dists(neighbours_dist, args)

        (dknn_ao_auroc, dknn_true_auroc, 
         dknn_ao_auprc, dknn_true_auprc,      
         dknn_ao_f1,   dknn_true_f1) = get_metrics(test_masks_recon, 
                                                     test_masks_orig_recon, 
                                                     dists_recon)

        fig, axs = plt.subplots(10,3, figsize=(10,7))
        axs[0,0].set_title('Inp',fontsize=5)
        axs[0,1].set_title('Mask',fontsize=5)
        axs[0,2].set_title('Recon {}'.format(dknn_ao_auroc),fontsize=5)

        for i in range(10):
            r = np.random.randint(len(test_data_recon))
            axs[i,0].imshow(test_data_recon[r,...,0].astype(np.float32), interpolation='nearest', aspect='auto')
            axs[i,1].imshow(test_masks_recon[r,...,0].astype(np.float32), interpolation='nearest', aspect='auto')
            axs[i,2].imshow(dists_recon[r,...,0].astype(np.float32), interpolation='nearest', aspect='auto')
        plt.savefig('outputs/{}/{}/{}/neighbours.png'.format(model_type,
                                                       args.anomaly_class,
                                                       args.model_name), dpi=300)

        return (dknn_ao_auroc, dknn_true_auroc, 
                dknn_ao_auprc, dknn_true_auprc,      
                dknn_ao_f1, dknn_true_f1,
                -1, -1, 
                -1, -1,      
                -1, -1,
                -1, -1, 
                -1, -1,      
                -1, -1,
                [-1], [-1], 
                [-1], [-1],      
                [-1], [-1])

    z = infer(model[0].encoder, train_images, args, 'encoder')
    z_query = infer(model[0].encoder, test_images, args, 'encoder')

    x_hat_train  = infer(model[0], train_images, args, 'AE')
    x_hat = infer(model[0], test_images, args, 'AE')
    x_hat_recon = patches.reconstruct(x_hat, args)

    error = get_error('AE', test_images, x_hat,mean=False)

    error_recon, labels_recon  = patches.reconstruct(error, args, test_labels) 

    (ae_ao_auroc, ae_true_auroc, 
     ae_ao_auprc, ae_true_auprc,      
     ae_ao_f1,   ae_true_f1) = get_metrics(test_masks_recon, 
                                             test_masks_orig_recon, 
                                             error_recon)



    neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask =  nln(z, 
                                                                        z_query, 
                                                                        x_hat_train, 
                                                                        args.algorithm, 
                                                                        neighbour,
                                                                        -1)
    nln_error = get_nln_errors(model,
                       'AE',
                       z_query,
                       z,
                       test_images,
                       x_hat_train,
                       neighbours_idx,
                       neighbour_mask,
                       args)
    if args.model == 'fine_tune':
        nln_error = infer(model[1], nln_error, args, 'AE')


    if args.patches:
        if nln_error.ndim ==4:
            nln_error_recon = patches.reconstruct(nln_error, args)
        else:
            nln_error_recon = patches.reconstruct_latent_patches(nln_error, args)
    else: nln_error_recon = nln_error
    

    dists_recon = get_dists(neighbours_dist, args)


    (nln_ao_auroc, nln_true_auroc, 
     nln_ao_auprc, nln_true_auprc,      
     nln_ao_f1,   nln_true_f1) = get_metrics(test_masks_recon, 
                                               test_masks_orig_recon, 
                                               nln_error_recon)


    (dists_ao_auroc, dists_true_auroc, 
     dists_ao_auprc, dists_true_auprc,      
     dists_ao_f1,   dists_true_f1) = get_metrics(test_masks_recon, 
                                                   test_masks_orig_recon, 
                                                   dists_recon)

    combined_true_aurocs, combined_true_auprcs, combined_true_f1s= [], [],[]
    combined_ao_aurocs, combined_ao_auprcs, combined_ao_f1s= [], [],[]
    for alpha in args.alphas:
        if args.data == 'LOFAR':
            combined_recon =  nln_error_recon*np.array([d > np.percentile(d,70) for d in dists_recon])#
        elif args.data == 'HERA':
            combined_recon =  nln_error_recon*np.array([d > np.percentile(d,10) for d in dists_recon])#

        combined_recon = np.nan_to_num(combined_recon)
        (combined_ao_auroc, combined_true_auroc, 
         combined_ao_auprc, combined_true_auprc,      
         combined_ao_f1,   combined_true_f1) = get_metrics(test_masks_recon, 
                                                       test_masks_orig_recon, 
                                                       combined_recon)
        combined_true_aurocs.append(combined_true_auroc)
        combined_true_auprcs.append(combined_true_auprc)
        combined_true_f1s.append(combined_true_f1)
        combined_ao_aurocs.append(combined_ao_auroc)
        combined_ao_auprcs.append(combined_ao_auprc)
        combined_ao_f1s.append(combined_ao_f1)

    fig, axs = plt.subplots(10,7, figsize=(10,8))
    axs[0,0].set_title('Inp',fontsize=5)
    axs[0,1].set_title('Mask',fontsize=5)
    axs[0,2].set_title('Recon {}'.format(ae_ao_auroc),fontsize=5)
    axs[0,3].set_title('NLN {} {}'.format(nln_ao_auroc, neighbour),fontsize=5)
    axs[0,4].set_title('Dist {} {}'.format(dists_ao_auroc, neighbour),fontsize=5)
    axs[0,5].set_title('Combined {} {}'.format(combined_ao_aurocs[0], neighbour),fontsize=5)
    axs[0,6].set_title('Recon {} {}'.format(combined_ao_aurocs[0], neighbour),fontsize=5)

    for i in range(10):
        r = np.random.randint(len(test_data_recon))
        axs[i,0].imshow(test_data_recon[r,...,0].astype(np.float32),  vmin=0, vmax=1,interpolation='nearest', aspect='auto')
        axs[i,1].imshow(test_masks_recon[r,...,0].astype(np.float32), vmin=0, vmax=1,interpolation='nearest', aspect='auto')
        axs[i,2].imshow(error_recon[r,...,0].astype(np.float32),      vmin=0, vmax=1,interpolation='nearest', aspect='auto')
        axs[i,3].imshow(nln_error_recon[r,...,0].astype(np.float32),  vmin=0, vmax=1,interpolation='nearest', aspect='auto')
        axs[i,4].imshow(dists_recon[r,...,0].astype(np.float32),      interpolation='nearest', aspect='auto')
        axs[i,5].imshow(combined_recon[r,...,0].astype(np.float32),   vmin=0, vmax=1,interpolation='nearest', aspect='auto')
        axs[i,6].imshow(x_hat_recon[r,...,0].astype(np.float32),      vmin=0, vmax=1,interpolation='nearest', aspect='auto')
    plt.savefig('outputs/{}/{}/{}/neighbours_{}.png'.format(model_type,
                                                   args.anomaly_class,
                                                   args.model_name,
                                                   neighbour), dpi=300)


    return (ae_ao_auroc,  ae_true_auroc, 
            ae_ao_auprc,  ae_true_auprc,      
            ae_ao_f1,    ae_true_f1,
            nln_ao_auroc, nln_true_auroc, 
            nln_ao_auprc, nln_true_auprc,      
            nln_ao_f1,   nln_true_f1,
            dists_ao_auroc, dists_true_auroc, 
            dists_ao_auprc, dists_true_auprc,      
            dists_ao_f1,   dists_true_f1,
            combined_ao_aurocs, combined_true_aurocs, 
            combined_ao_auprcs, combined_true_auprcs,      
            combined_ao_f1s,   combined_true_f1s)

def get_metrics(test_masks_recon,test_masks_orig_recon, error_recon):

    # AUROC True 
    fpr,tpr, thr = roc_curve(test_masks_orig_recon.flatten()>0, 
                             error_recon.flatten())
    true_auroc = auc(fpr, tpr)

    # AUPRC True 
    precision, recall, thresholds = precision_recall_curve(test_masks_orig_recon.flatten()>0, 
                                                           error_recon.flatten())
    true_auprc = auc(recall, precision)

    f1_scores = 2*recall*precision/(recall+precision)
    true_f1 = np.max(f1_scores)

    return -1, true_auroc, -1, true_auprc, -1, true_f1
    

def normalise(x):
    """
        Returns normalised input between 0 and 1

        Parameters
        ----------
        x (np.array): 1D array to be Normalised

        Returns
        -------
        y (np.array): Normalised array
    """
    y = []
    for _x in x:
        y.append((_x- np.min(_x))/(np.max(_x) - np.min(_x)))
    return np.array(y)

def get_dists(neighbours_dist, args):
    """
        Reconstruct distance vector to original dimensions when using patches

        Parameters
        ----------
        neighbours_dist (np.array): Vector of per neighbour distances
        args (Namespace): cmd_args 

        Returns
        -------
        dists (np.array): reconstructed patches if necessary

    """

    dists = np.mean(neighbours_dist, axis = tuple(range(1,neighbours_dist.ndim)))
    if args.patches:
        dists = np.array([[d]*args.patch_x**2 for i,d in enumerate(dists)]).reshape(len(dists), args.patch_x, args.patch_y)
        dists_recon = reconstruct(np.expand_dims(dists,axis=-1),args)
        return dists_recon
    else:
        return dists 

