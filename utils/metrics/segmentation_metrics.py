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
from utils.data import *
from utils.metrics import nln, get_nln_errors
from reporting import plot_neighs
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

    if model_type =='UNET':
        x_hat = infer(model[0], test_images, args, 'AE')
        x_hat_recon = patches.reconstruct(x_hat, args)

        (unet_ao_auroc, unet_true_auroc, 
         unet_ao_auprc, unet_true_auprc,      
         unet_ao_iou, unet_true_iou) = get_metrics(test_masks_recon, 
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
         unet_ao_iou, unet_true_iou,
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
         dknn_ao_iou,   dknn_true_iou) = get_metrics(test_masks_recon, 
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
                dknn_ao_iou, dknn_true_iou,
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
     ae_ao_iou,   ae_true_iou) = get_metrics(test_masks_recon, 
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

    if args.patches:
        if nln_error.ndim ==4:
            nln_error_recon = patches.reconstruct(nln_error, args)
        else:
            nln_error_recon = patches.reconstruct_latent_patches(nln_error, args)
    else: nln_error_recon = nln_error
    

    dists_recon = get_dists(neighbours_dist, args)


    (nln_ao_auroc, nln_true_auroc, 
     nln_ao_auprc, nln_true_auprc,      
     nln_ao_iou,   nln_true_iou) = get_metrics(test_masks_recon, 
                                               test_masks_orig_recon, 
                                               nln_error_recon)


    (dists_ao_auroc, dists_true_auroc, 
     dists_ao_auprc, dists_true_auprc,      
     dists_ao_iou,   dists_true_iou) = get_metrics(test_masks_recon, 
                                                   test_masks_orig_recon, 
                                                   dists_recon)

    combined_true_aurocs, combined_true_auprcs, combined_true_ious  = [], [],[]
    combined_ao_aurocs, combined_ao_auprcs, combined_ao_ious  = [], [],[]
    for alpha in args.alphas:
       # combined_recon = normalise(nln_error_recon*np.array([d > 3*np.median(d) for d in dists_recon]))
        combined_recon = np.clip(nln_error_recon, nln_error_recon.mean() + nln_error_recon.std()*5,1.0)*np.array([d > np.percentile(d,60) for d in dists_recon])
        combined_recon = np.nan_to_num(combined_recon)
        combined_recon = np.nan_to_num(combined_recon)
        (combined_ao_auroc, combined_true_auroc, 
         combined_ao_auprc, combined_true_auprc,      
         combined_ao_iou,   combined_true_iou) = get_metrics(test_masks_recon, 
                                                       test_masks_orig_recon, 
                                                       combined_recon)
        combined_true_aurocs.append(combined_true_auroc)
        combined_true_auprcs.append(combined_true_auprc)
        combined_true_ious.append(combined_true_iou)
        combined_ao_aurocs.append(combined_ao_auroc)
        combined_ao_auprcs.append(combined_ao_auprc)
        combined_ao_ious.append(combined_ao_iou)

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
            ae_ao_iou,    ae_true_iou,
            nln_ao_auroc, nln_true_auroc, 
            nln_ao_auprc, nln_true_auprc,      
            nln_ao_iou,   nln_true_iou,
            dists_ao_auroc, dists_true_auroc, 
            dists_ao_auprc, dists_true_auprc,      
            dists_ao_iou,   dists_true_iou,
            combined_ao_aurocs, combined_true_aurocs, 
            combined_ao_auprcs, combined_true_auprcs,      
            combined_ao_ious,   combined_true_ious)

def get_metrics(test_masks_recon,test_masks_orig_recon, error_recon):

    ## AUROC AOFlagger  
#    fpr,tpr, thr = roc_curve(test_masks_recon.flatten()>0, 
#                             error_recon.flatten())
#    ao_auroc = auc(fpr, tpr)
#
    # AUROC True 
    fpr,tpr, thr = roc_curve(test_masks_orig_recon.flatten()>0, 
                             error_recon.flatten())
    true_auroc = auc(fpr, tpr)

    # IOU AOFlagger  

#    ao_iou = iou_score(error_recon, 
#                       test_masks_recon, 
#                       fpr, 
#                       tpr, 
#                       thr)
    # IOU True
    true_iou = iou_score(error_recon, 
                         test_masks_orig_recon, 
                         fpr, 
                         tpr, 
                         thr)

    # AUPRC AOFlagger  
#    precision, recall, thresholds = precision_recall_curve(test_masks_recon.flatten()>0, 
#                                                           error_recon.flatten())
#
#    ao_auprc = auc(recall, precision)

    # AUPRC True 
    precision, recall, thresholds = precision_recall_curve(test_masks_orig_recon.flatten()>0, 
                                                           error_recon.flatten())
    true_auprc = auc(recall, precision)


    #return ao_auroc, true_auroc, ao_auprc, true_auprc, ao_iou, true_iou
    return -1, true_auroc, -1, true_auprc, -1, true_iou
    
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

