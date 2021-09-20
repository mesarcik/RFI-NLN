import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt
from models import Autoencoder, UNET
from data import load_lofar
from sklearn import metrics,neighbors
from inference import infer, get_error
from utils.data import reconstruct, reconstruct_latent_patches, sizes, process
from utils.metrics import  accuracy_metrics
from model_config import *

MODEL_NAME=  'able-sandy-dugong-of-romance'
models = ['UNET', 'AE', 'DAE_disc', 'GANomaly']

LD =128
PATCH=128

class Namespace:
     def __init__(self, **kwargs):
         self.__dict__.update(kwargs)
     def set_class(self,clss):
         self.anomaly_class = clss

args = Namespace(input_shape=(128, 64, 1),
                 rotate=False,
                 crop=False,
                 epochs=-1,
                 patches=False,
                 percentage_anomaly=0,
                 limit= None,
                 patch_x = PATCH,
                 patch_y=PATCH,
                 patch_stride_x = PATCH,
                 patch_stride_y = PATCH,
                 crop_x=PATCH,
                 crop_y=PATCH,
                 latent_dim=128,
                 data_path = 'NONE',
                 # NLN PARAMS
                 model_name = MODEL_NAME,
                 data ='LOFAR',
                 anomaly_class='rfi',
                 anomaly_type='MISO',
                 radius= [10],
                 neighbors= [1,5,10,20],
                 algorithm = 'knn')

def main():
    first_flag = True
    fig, axs = plt.subplots(10,6,figsize=(10,5))
    axs[0,0].set_title('Input',fontsize=5)
    axs[0,1].set_title('Mask',fontsize=5)
    axs[0,2].set_title('UNET',fontsize=5)
    axs[0,3].set_title('AE',fontsize=5)
    axs[0,4].set_title('DAE',fontsize=5)
    axs[0,5].set_title('',fontsize=5)

    for model_type in models:
        if model_type== 'UNET':
            path = 'data/datasets/LOFAR_UNET_dataset_17-09-2021.pkl'
            unet_data  = load_lofar(args, path, unet=True)
            (train_dataset,
                train_images, 
                train_masks,
                train_labels,
                test_images,
                test_labels, 
                test_masks) = unet_data
            model = UNET()
            model.load_weights('outputs/{}/rfi/{}/training_checkpoints/checkpoint_full_model_unet'
                                 .format(model_type, MODEL_NAME))

        else:
            path = 'data/datasets/LOFAR_AE_dataset_17-09-2021.pkl' 
            ae_data  = load_lofar(args, path, unet=False)
            (train_dataset,
             train_images, 
             train_labels,
             test_images,
             test_labels, 
             test_masks) = ae_data
            model = Autoencoder(args)
            model.load_weights('outputs/{}/rfi/{}/training_checkpoints/checkpoint_full_model_ae'
                                 .format(model_type, MODEL_NAME))
        if first_flag:
            rs = np.round(np.random.random([10])*len(test_images)).astype('int')

        x_hat  = infer(model, test_images, args, 'AE')
        if model_type == 'UNET':
            error = x_hat 
        else:
            d = accuracy_metrics([model],
                     train_images,
                     test_images,
                     test_labels,
                     test_masks,
                     model_type,
                     args)
            k_max = 0
            ind = 0
            for key in d.keys():
                if d[key][0] > k_max:
                    k_max = d[key][0]
                    ind = key
                print(key, d[key][0])
            error = d[ind][1]

        fpr, tpr, thr  = metrics.roc_curve(test_masks.flatten(), error[...,0].flatten())
        threshold =thr[np.argmax(tpr-fpr)]
        auroc = metrics.auc(fpr,tpr)
        thresholded = error[...,0]>threshold
        auprc = metrics.average_precision_score(test_masks.flatten(), error[...,0].flatten())
        jaccard = metrics.jaccard_score(test_masks.flatten(), thresholded.flatten())
        f1 = metrics.f1_score(test_masks.flatten(), thresholded.flatten())

        print("{}: AUROC = {}, AUPRC = {}, Jaccard = {}, f1 = {}".
                format(model_type, 
                        auroc,
                        auprc,
                        jaccard,
                        f1))

        
        for i in range(10):
            r = rs[i]
            if first_flag:
                axs[i,0].imshow(test_images[r,...,0].transpose());
                axs[i,1].imshow(test_masks[r,...].transpose(), vmax=1, vmin=0);
            if model_type == 'UNET':
                axs[i,2].imshow(np.abs(error[r,...,0]).transpose()>threshold, vmax=1, vmin=0);
            elif model_type == 'AE':
                axs[i,3].imshow(np.abs(error[r,...,0]).transpose()>threshold, vmax=1, vmin=0);
            elif model_type == 'DAE_disc':
                axs[i,4].imshow(np.abs(error[r,...,0]).transpose()>threshold, vmax=1, vmin=0);
            elif model_type == 'GANomaly':
                axs[i,5].imshow(np.abs(error[r,...,0]).transpose()>threshold, vmax=1, vmin=0);
        first_flag = False

    plt.savefig('/tmp/temp',dpi=300)   


if __name__ == '__main__':
    main()
