import tensorflow as tf
import numpy as np
import os 
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score
from math import isnan
from inference import infer, get_error
from utils import cmd_input 
from utils.data import reconstruct


def calculate_metrics(error,  
                      test_labels,
                      args):
    """
        Returns the AUROC of a particular model 

        Parameters
        ----------
        error (np.array): the reconstruction error of a given model
        test_labels (np.array): the test labels from the testing set
        args (Namespace):  arguments from utils.cmd_input

        Returns
        -------
        _auc (np.float32): return the auc
    """
    if args.anomaly_type == 'MISO':
        _auc = roc_auc_score(test_labels==args.anomaly_class, error)
    else:
        _auc = roc_auc_score(test_labels!=args.anomaly_class, error)

    return _auc


def get_classifcation(model_type,
                      model,
                      test_images,
                      test_labels,
                      args):
    """
        Returns the AUROC score of a particular model 

        Parameters
        ----------
        model_type (str): type of model (AE,VAE,....)
        model (list): the model used
        test_images (np.array): the test images from the testing set
        test_labels (np.array): the test labels from the testing set
        args (Namespace):  arguments from utils.cmd_input

        Returns
        -------
        auc (np.float32): return the auc
    """
    x_hat = infer(model[0], test_images, args, 'AE')

    if args.patches :
        error = get_error('AE', test_images, x_hat, mean=False)
        error, test_labels = reconstruct(error, args, test_labels)
        error =  error.mean(axis=tuple(range(1,error.ndim)))

    else:
        error = get_error('AE', test_images, x_hat, mean=True)

    auc = calculate_metrics(error,test_labels,args)
    return auc

def save_metrics(model_type,
                test_masks,
                test_masks_orig,
                 args,
                 **kwargs):
    
    """
        Either appends or saves a new .csv file with the top K 

        Parameters
        ----------
        model_type (str): type of model (vae,ae,..)
        args (Namespace):  arguments from utils.cmd_input
        ... (optional arguments)

        Returns
        -------
        nothing
    """
    if not os.path.exists('outputs/results_{}_{}.csv'.format(args.data,
                                                             args.seed)):
        df = pd.DataFrame(columns = ['Model',
                                     'Name',
                                     'Latent_Dim',
                                     'Patch_Size',
                                     'Class',
                                     'Type',
                                     'Percentage Anomaly',
                                     'RFI',

                                     'AUROC_AO',
                                     'AUROC_TRUE',
                                     'AUPRC_AO',
                                     'AUPRC_TRUE',
                                     'IOU_AO',
                                     'IOU_TRUE',

                                     'NLN_AUROC_AO',
                                     'NLN_AUROC_TRUE',
                                     'NLN_AUPRC_AO',
                                     'NLN_AUPRC_TRUE',
                                     'NLN_IOU_AO',
                                     'NLN_IOU_TRUE',


                                     'DISTS_AUROC_AO',
                                     'DISTS_AUROC_TRUE',
                                     'DISTS_AUPRC_AO',
                                     'DISTS_AUPRC_TRUE',
                                     'DISTS_IOU_AO',
                                     'DISTS_IOU_TRUE',

                                     'COMBINED_AUROC_AO',
                                     'COMBINED_AUROC_TRUE',
                                     'COMBINED_AUPRC_AO',
                                     'COMBINED_AUPRC_TRUE',
                                     'COMBINED_IOU_AO',
                                     'COMBINED_IOU_TRUE'])
    else:  
        df = pd.read_csv('outputs/results_{}_{}.csv'.format(args.data,
                                                            args.seed))

    perc = round(((np.sum(test_masks) - np.sum(test_masks_orig))/np.prod(test_masks_orig.shape)),3)
    df = df.append({'Model':model_type,
                    'Name':args.model_name,
                    'Latent_Dim':cmd_input.args.latent_dim,
                    'Patch_Size':args.patch_x,
                    'Class':args.anomaly_class,
                    'Type':args.anomaly_type,
                    'Percentage Anomaly':perc,
                    'RFI':args.rfi,


                     'AUROC_AO':   kwargs['ae_ao_auroc']  ,
                     'AUROC_TRUE': kwargs['ae_true_auroc'] ,
                     'AUPRC_AO':   kwargs['ae_ao_auprc']  ,
                     'AUPRC_TRUE': kwargs['ae_true_auprc'] ,
                     'IOU_AO':     kwargs['ae_ao_iou']    ,
                     'IOU_TRUE':   kwargs['ae_true_iou']  ,

                     'NLN_AUROC_AO':   kwargs['nln_ao_auroc']  ,
                     'NLN_AUROC_TRUE': kwargs['nln_true_auroc'] ,
                     'NLN_AUPRC_AO':   kwargs['nln_ao_auprc']  ,
                     'NLN_AUPRC_TRUE': kwargs['nln_true_auprc'] ,
                     'NLN_IOU_AO':     kwargs['nln_ao_iou']    ,
                     'NLN_IOU_TRUE':   kwargs['nln_true_iou']  ,

                     'DISTS_AUROC_AO':   kwargs['dists_ao_auroc']  ,
                     'DISTS_AUROC_TRUE': kwargs['dists_true_auroc'] ,
                     'DISTS_AUPRC_AO':   kwargs['dists_ao_auprc']  ,
                     'DISTS_AUPRC_TRUE': kwargs['dists_true_auprc'] ,
                     'DISTS_IOU_AO':     kwargs['dists_ao_iou']    ,
                     'DISTS_IOU_TRUE':   kwargs['dists_true_iou']  ,

                     'COMBINED_AUROC_AO':   kwargs['combined_ao_auroc']  ,
                     'COMBINED_AUROC_TRUE': kwargs['combined_true_auroc'] ,
                     'COMBINED_AUPRC_AO':   kwargs['combined_ao_auprc']  ,
                     'COMBINED_AUPRC_TRUE': kwargs['combined_true_auprc'] ,
                     'COMBINED_IOU_AO':     kwargs['combined_ao_iou']    ,
                     'COMBINED_IOU_TRUE':   kwargs['combined_true_iou']
                      }, ignore_index=True)

    df.to_csv('outputs/results_{}_{}.csv'.format(args.data,
                                                 args.seed),index=False)
