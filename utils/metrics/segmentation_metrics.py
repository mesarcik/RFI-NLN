import tensorflow as tf
import numpy as np
from sklearn.metrics import (roc_curve,
                             auc, 
                             accuracy_score, 
                             average_precision_score, 
                             jaccard_score,
                             roc_auc_score, 
                             precision_recall_curve)
from inference import infer, get_error
from utils import cmd_input 
from utils.data import *
from utils.metrics import nln, get_nln_errors
from reporting import plot_neighs
from matplotlib import pyplot as plt

import time

def accuracy_metrics(model,
                     train_images,
                     test_images,
                     test_labels,
                     test_masks,
                     test_masks_orig,
                     model_type,
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

    if model_type =='UNET':
        x_hat = infer(model[0], test_images, args, 'AE')
        x_hat_recon = patches.reconstruct(x_hat, args)
        unet_auroc, unet_auprc, unet_iou = get_metrics(test_masks_recon, test_masks_orig_recon, x_hat_recon)

        fig, axs = plt.subplots(10,3, figsize=(10,7))
        axs[0,0].set_title('Inp',fontsize=5)
        axs[0,1].set_title('Mask',fontsize=5)
        axs[0,2].set_title('Recon {}'.format(unet_auroc),fontsize=5)

        for i in range(10):
            r = np.random.randint(len(test_data_recon))
            axs[i,0].imshow(test_data_recon[r,...,0].astype(np.float32))
            axs[i,1].imshow(test_masks_recon[r,...,0].astype(np.float32))
            axs[i,2].imshow(x_hat_recon[r,...,0].astype(np.float32))
        plt.savefig('outputs/{}/{}/{}/neighbours.png'.format(model_type,
                                                       args.anomaly_class,
                                                       args.model_name), dpi=300)

        return (unet_auroc, unet_auprc, unet_iou, -1,-1,-1,-1,-1,-1, -1, -1, -1)

    elif model_type =='DKNN':
        z_train = infer(model[0], train_images, args, 'DKNN')
        z_test = infer(model[0], test_images, args, 'DKNN')


        neighbours_dist, _, _, _ =  nln(z_train, z_test, None, 'knn', 2, -1)

        dists_recon = get_dists(neighbours_dist, args)
        dknn_auroc, dknn_auprc, dknn_iou = get_metrics(test_masks_recon,  
                                                      test_masks_orig_recon,
                                                      dists_recon)

        fig, axs = plt.subplots(10,3, figsize=(10,7))
        axs[0,0].set_title('Inp',fontsize=5)
        axs[0,1].set_title('Mask',fontsize=5)
        axs[0,2].set_title('Recon {}'.format(dknn_auroc),fontsize=5)

        for i in range(10):
            r = np.random.randint(len(test_data_recon))
            axs[i,0].imshow(test_data_recon[r,...,0].astype(np.float32))
            axs[i,1].imshow(test_masks_recon[r,...,0].astype(np.float32))
            axs[i,2].imshow(dists_recon[r,...,0].astype(np.float32))
        plt.savefig('outputs/{}/{}/{}/neighbours.png'.format(model_type,
                                                       args.anomaly_class,
                                                       args.model_name), dpi=300)

        return (dknn_auroc, dknn_auprc, dknn_iou, -1,-1,-1,-1,-1,-1,-1,-1, -1)

    z = infer(model[0].encoder, train_images, args, 'encoder')
    z_query = infer(model[0].encoder, test_images, args, 'encoder')

    x_hat_train  = infer(model[0], train_images, args, 'AE')
    x_hat = infer(model[0], test_images, args, 'AE')

    error = get_error('AE', test_images, x_hat,mean=False)

    error_recon, labels_recon  = patches.reconstruct(error, args, test_labels) 

    ae_auroc, ae_auprc, ae_iou = get_metrics(test_masks_recon, test_masks_orig_recon, error_recon)
    nln_aurocs, dists_aurocs, combined_aurocs = [], [], []
    for n in args.neighbors:
        print('Neighbours = {}'.format(n))
        neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask =  nln(z, 
                                                                            z_query, 
                                                                            x_hat_train, 
                                                                            args.algorithm, 
                                                                            n,
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


        if args.patches:
            if nln_error.ndim ==4:
                nln_error_recon = patches.reconstruct(nln_error, args)
            else:
                nln_error_recon = patches.reconstruct_latent_patches(nln_error, args)
        else: nln_error_recon = nln_error
        

        dists_recon = get_dists(neighbours_dist, args)
        alpha=0.3
        combined_recon = alpha*normalise(nln_error_recon) + (1-alpha)*normalise(dists_recon)

        nln_auroc, nln_auprc, nln_iou = get_metrics(test_masks_recon, test_masks_orig_recon, nln_error_recon)
        dists_auroc, dists_auprc, dists_iou = get_metrics(test_masks_recon, test_masks_orig_recon, dists_recon)
        combined_auroc, combined_auprc, combined_iou = get_metrics(test_masks_recon, test_masks_orig_recon, combined_recon)
        nln_aurocs.append(nln_auroc)
        dists_aurocs.append(dists_auroc)
        combined_aurocs.append(combined_auroc)

    dists_auroc = np.max(dists_aurocs)    
    n_dist = args.neighbors[np.argmax(dists_aurocs)]
    nln_auroc = np.max(nln_aurocs)    
    n_nln = args.neighbors[np.argmax(nln_aurocs)]
    combined_auroc = np.max(combined_aurocs)    
    n_combined= args.neighbors[np.argmax(combined_aurocs)]

    fig, axs = plt.subplots(10,6, figsize=(10,7))
    axs[0,0].set_title('Inp',fontsize=5)
    axs[0,1].set_title('Mask',fontsize=5)
    axs[0,2].set_title('Recon {}'.format(ae_auroc),fontsize=5)
    axs[0,3].set_title('NLN {} {}'.format(nln_auroc, n_nln),fontsize=5)
    axs[0,4].set_title('Dist {} {}'.format(dists_auroc, n_dist),fontsize=5)
    axs[0,5].set_title('Combined {} {}'.format(combined_auroc, n_combined),fontsize=5)

    for i in range(10):
        r = np.random.randint(len(test_data_recon))
        axs[i,0].imshow(test_data_recon[r,...,0].astype(np.float32))
        axs[i,1].imshow(test_masks_recon[r,...,0].astype(np.float32))
        axs[i,2].imshow(error_recon[r,...,0].astype(np.float32))
        axs[i,3].imshow(nln_error_recon[r,...,0].astype(np.float32))
        axs[i,4].imshow(dists_recon[r,...,0].astype(np.float32))
        axs[i,5].imshow(combined_recon[r,...,0].astype(np.float32))
    plt.savefig('outputs/{}/{}/{}/neighbours.png'.format(model_type,
                                                   args.anomaly_class,
                                                   args.model_name), dpi=300)


    return (ae_auroc, 
            ae_auprc, 
            ae_iou, 
            nln_auroc, 
            nln_auprc, 
            nln_iou, 
            dists_auroc, 
            dists_auprc, 
            dists_iou,
            combined_auroc, 
            combined_auprc, 
            combined_iou)

def get_metrics(test_masks_recon,test_masks_orig_recon, error_recon):
    fpr,tpr, thr = roc_curve(test_masks_recon.flatten()>0, error_recon.flatten())
    auroc = auc(fpr, tpr)
    iou = iou_score(error_recon, test_masks_recon, fpr, tpr, thr)

    #fpr,tpr, thr = roc_curve(test_masks_orig_recon.flatten()>0, error_recon.flatten())
    #auprc = auc(fpr, tpr)

    return auroc, -1, iou
    
def get_threshold(fpr,tpr,thr,flag,test_labels,error,anomaly_class):
    """
        Returns optimal threshold

        Parameters
        ----------
        fpr (np.array): false positive rate
        tpr (np.array): true positive rate
        thr (np.array): thresholds for AUROC
        flag (str): method of calculating threshold
        anomaly_class (str):  name of anomalous class 
        error (np.array): input-output
        test_labels (np.array): ground truth labels  
    
        Returns
        -------
        thr (float32): Optimal threshold  
    """
    if flag == 'MD':# MD = Maximise diff
        idx = np.argmax(tpr-fpr) 
    if flag == 'MA': # MA = Maximise average
        idx, temp = None, 0
        for i,t in enumerate(thr):
            normal_accuracy = accuracy_score(test_labels == 'non_anomalous', error < t)
            anomalous_accuracy = accuracy_score(test_labels == anomaly_class, error > t)
            m = np.mean([anomalous_accuracy, normal_accuracy])
            if  m > temp:
                idx = i
                temp = m
    return thr[idx]

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
    y = (x- np.min(x))/(np.max(x) - np.min(x))

    return y

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

def iou_score(error, test_masks,fpr,tpr,thr):
    """
        Get jaccard index or IOU score

        Parameters
        ----------
        error (np.array): input-output
        test_masks (np.array): ground truth mask 

        Returns
        -------
        max_iou (float32): maximum iou score for a number of thresholds

    """

    idx = np.argmax(tpr-fpr) 
    thresholded = np.mean(error,axis=-1) >=thr[idx]
    iou = jaccard_score(test_masks.flatten()>0, thresholded.flatten())
    return iou
    #iou = []
    #for threshold in np.linspace(np.min(thr), np.max(thr),10):
    #    thresholded =np.mean(error,axis=-1) >=threshold
    #    iou.append(jaccard_score(test_masks.flatten()>0, thresholded.flatten()))

    #return max(iou) 

