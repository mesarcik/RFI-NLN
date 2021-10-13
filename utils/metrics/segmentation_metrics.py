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
                     model_type,
                     max_neighbours,
                     max_radius,
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
        max_neighbours (int): number of neighbours resulting in best AUROC
        max_radius (double): size of radius resulting in best AUROC

        Returns
        -------
        seg_auc (float32): segmentation auroc 
        seg_auc_nln (float32): segmentation auroc using NLN
        dists_auc (float32): detection auroc using dists
        seg_dists_auc (float32): segmentation auroc using dists
        seg_prc (float32): segmetnation auprc 
        seg_prc_nln (float32): segmentation auprc using nln
        seg_iou (float32): iou score for reconstruction
        seg_iou_nln (float32): iou score for nln
    """
    # Get output from model #TODO: do we want to normalise?
    test_data_recon = patches.reconstruct(test_images, args)
    test_masks_recon = patches.reconstruct(test_masks, args)

    if model_type =='UNET':
        x_hat = infer(model[0], test_images, args, 'AE')
        x_hat_recon = patches.reconstruct(x_hat, args)
        unet_auc = round(roc_auc_score(test_masks_recon.flatten()>0, x_hat_recon.flatten()),3)

        fig, axs = plt.subplots(10,3, figsize=(10,7))
        axs[0,0].set_title('Inp',fontsize=5)
        axs[0,1].set_title('Mask',fontsize=5)
        axs[0,2].set_title('Recon {}'.format(unet_auc),fontsize=5)

        for i in range(10):
            r = np.random.randint(len(test_data_recon))
            axs[i,0].imshow(test_data_recon[r,...,0])
            axs[i,1].imshow(test_masks_recon[r,...,0])
            axs[i,2].imshow(x_hat_recon[r,...,0])
        plt.savefig('outputs/{}/{}/{}/neighbours.png'.format(model_type,
                                                       args.anomaly_class,
                                                       args.model_name), dpi=300)
        return unet_auc, -1, -1

    z = infer(model[0].encoder, train_images, args, 'encoder')
    z_query = infer(model[0].encoder, test_images, args, 'encoder')

    x_hat_train  = infer(model[0], train_images, args, 'AE')
    x_hat = infer(model[0], test_images, args, 'AE')

    error = get_error('AE', test_images, x_hat,mean=False)

    if args.patches:
        error_recon, labels_recon  = patches.reconstruct(error, args, test_labels) 
        masks_recon = patches.reconstruct(test_masks, args)
    else: 
        error_recon, labels_recon, masks_recon  = error, test_labels, test_masks 

    error_auc = round(roc_auc_score(test_masks_recon.flatten()>0, error_recon.flatten()),3)
    nln_aucs, dist_aucs = [], [] 
    for n in args.neighbors:
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

        nln_auc =   round(roc_auc_score(test_masks_recon.flatten()>0, nln_error_recon.flatten()),3)
        dists_auc = round(roc_auc_score(test_masks_recon.flatten()>0, dists_recon.flatten()),3)
        nln_aucs.append(nln_auc)
        dist_aucs.append(dists_auc)

    dists_auc = np.max(dist_aucs)    
    n_dist = args.neighbors[np.argmax(dist_aucs)]
    nln_auc = np.max(nln_aucs)    
    n_nln = args.neighbors[np.argmax(nln_aucs)]

    fig, axs = plt.subplots(10,5, figsize=(10,7))
    axs[0,0].set_title('Inp',fontsize=5)
    axs[0,1].set_title('Mask',fontsize=5)
    axs[0,2].set_title('Recon {}'.format(error_auc),fontsize=5)
    axs[0,3].set_title('NLN {} {}'.format(nln_auc, n_nln),fontsize=5)
    axs[0,4].set_title('Dist {} {}'.format(dists_auc, n_dist),fontsize=5)

    for i in range(10):
        r = np.random.randint(len(test_data_recon))
        axs[i,0].imshow(test_data_recon[r,...,0])
        axs[i,1].imshow(test_masks_recon[r,...,0])
        axs[i,2].imshow(error_recon[r,...,0])
        axs[i,3].imshow(nln_error_recon[r,...,0])
        axs[i,4].imshow(dists_recon[r,...,0])
    plt.savefig('outputs/{}/{}/{}/neighbours.png'.format(model_type,
                                                   args.anomaly_class,
                                                   args.model_name), dpi=300)
    return error_auc, nln_auc, dists_auc


def get_segmentation(error, test_masks, test_labels, args):
    """
        Calculates AUROC result of segmentation

        Parameters
        ----------
        error (np.array): input-output
        test_masks (np.array): ground truth segmentation masks 
        test_labels (np.array): ground truth labels  
        args (Namespace): cmd_input args

        Returns
        -------
        auc (float32): AUROC for segmentation
        prc (float32): AUPRC for segmentation
        
    """
    fpr, tpr, thr  = roc_curve(test_masks.flatten()>0, np.max(error,axis=-1).flatten())
    precision, recall, thresholds = precision_recall_curve(test_masks.flatten()>0, np.max(error,axis=-1).flatten())
    prc = auc(recall, precision)
    AUC= roc_auc_score(test_masks.flatten()>0, np.mean(error,axis=-1).flatten())

    return AUC,prc

def get_acc(anomaly_class, test_labels, error):
    """
        Calculates get accuracy for anomaly detection  

        Parameters
        ----------
        anomaly_class (str):  name of anomalous class 
        error (np.array): input-output
        test_labels (np.array): ground truth labels  

        Returns
        -------
        auc (float32): AUROC for segmentation
        normal_accuracy (float32): Accuracy of detecting normal samples
        anomalous_accuracy(float32): Accuracy of detecting anomalous samples 
        
    """
    # Find AUROC threshold that optimises max(TPR-FPR)
    print(anomaly_class)
    fpr, tpr, thr  = roc_curve(test_labels == anomaly_class, error)
    AUC= roc_auc_score(test_labels==anomaly_class,error)

    thr = get_threshold(fpr,tpr,thr,'MD',test_labels, error,anomaly_class)

    # Accuracy of detecting anomalies and non-anomalies using this threshold
    normal_accuracy = accuracy_score(test_labels == 'non_anomalous', error < thr)
    anomalous_accuracy = accuracy_score(test_labels == anomaly_class, error > thr)

    print('Anomalous Accuracy = {}'.format(anomalous_accuracy))
    print('Normal Accuracy = {}'.format(normal_accuracy))

    return AUC, normal_accuracy, anomalous_accuracy

    
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

def iou_score(error, test_masks):
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
    fpr,tpr, thr = roc_curve(test_masks.flatten()>0, np.mean(error,axis=-1).flatten())

    iou = []
    for threshold in np.linspace(np.min(thr), np.max(thr),10):
        thresholded =np.mean(error,axis=-1) >=threshold
        iou.append(jaccard_score(test_masks.flatten()>0, thresholded.flatten()))

    return max(iou) 

