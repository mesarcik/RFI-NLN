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
                                     'AUROC',
                                     'AUROC_ORIG',
                                     'IOU',
                                     'NLN_AUROC',
                                     'NLN_AUROC_ORIG',
                                     'NLN_IOU',
                                     'DIST_AUROC',
                                     'DIST_AUROC_ORIG',
                                     'DIST_IOU',
                                     'COMBINED_AUROC',
                                     'COMBINED_AUROC_ORIG',
                                     'COMBINED_IOU'])
    else:  
        df = pd.read_csv('outputs/results_{}_{}.csv'.format(args.data,
                                                            args.seed))

    perc = 1- round(np.sum(test_masks)/np.sum(test_masks_orig),3)
    df = df.append({'Model':model_type,
                    'Name':args.model_name,
                    'Latent_Dim':cmd_input.args.latent_dim,
                    'Patch_Size':args.patch_x,
                    'Class':args.anomaly_class,
                    'Type':args.anomaly_type,
                    'Percentage Anomaly':perc,
                    'RFI':args.rfi,
                    'AUROC':kwargs['ae_auroc'],
                    'AUROC_ORIG':kwargs['ae_auprc'],
                    'IOU':kwargs['ae_iou'],
                    'NLN_AUROC':kwargs['nln_auroc'],
                    'NLN_AUROC_ORIG':kwargs['nln_auprc'],
                    'NLN_IOU':kwargs['nln_iou'],
                    'DIST_AUROC':kwargs['dists_auroc'],
                    'DIST_AUROC_ORIG':kwargs['dists_auprc'],
                    'DIST_IOU':kwargs['dists_iou'],
                    'COMBINED_AUROC':kwargs['combined_auroc'],
                    'COMBINED_AUROC_ORIG':kwargs['combined_auprc'],
                    'COMBINED_IOU':kwargs['combined_iou']
                    },
                     ignore_index=True)

    df.to_csv('outputs/results_{}_{}.csv'.format(args.data,
                                                 args.seed),index=False)
