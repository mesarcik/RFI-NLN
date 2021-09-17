import tensorflow as tf
import numpy as np
import os 
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, jaccard_score 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

import time

from inference import infer, get_error
from data import *
from utils.metrics import *
from models import Encoder,Autoencoder, Discriminator_x



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def set_class(self,clss):
        self.anomaly_class = clss
    def set_name(self,clss):
        self.model_name= clss
    def set_input_shape(self,input_shape):
        self.input_shape= input_shape 
    def set_anomaly_type(self,anomaly_type):
        self.anomaly_type = anomaly_type 
    def set_ld(self,ld):
        self.latent_dim = ld 


args = Namespace(
    data='LOFAR',
    data_path='/data/mmesarcik/LOFAR/LOFAR_training_data/datasets/LOFAR_dataset_14-09-2021.pkl',
    seed='',
    input_shape=(128, 64, 1),
    rotate=False,
    crop=False,
    patches=False,
    percentage_anomaly=0,
    model_name=None,
    limit=None,
    latent_dim=128,
    # NLN PARAMS
    anomaly_class='point_source',
    anomaly_type='MISO',
    neighbors= [1, 2, 5, 10, 16, 20],
    algorithm = 'knn'
)

def main(cmd_args):
    df_out = pd.DataFrame(columns=('Model','Class','Latent_Dim','Neighbours','Alpha','NLN','LOF','IF','OCSVM'))
    indxer = 0
    df_ = pd.read_csv('outputs/results_{}_{}.csv'.format(cmd_args.data, cmd_args.seed))
    

    models = list(pd.unique(df_.Model))  

    for model_type in models:
        results = {}
        # find optimal latent dimensions 
        #means = [df_[(df_.Model == model_type) & (df_.Latent_Dim == ld)]['Mul_Recon_NLN_Dist'].mean() for ld in list(pd.unique(df_.Latent_Dim))]
        #ld = list(pd.unique(df_.Latent_Dim))[means.index(max(means))]
        for ld in list(pd.unique(df_.Latent_Dim)):
            args.set_ld(ld)
            df = df_[df_.Latent_Dim == ld]
            model_names = list(pd.unique(df.Name)) 
                    
            for i,clss in enumerate(list(pd.unique(df.Class))):
                detections, lofs, svms, ifs, n_arr = [], [], [], [], []
                args.set_class(clss)
                args.set_anomaly_type(df.iloc[i].Type)
                (train_dataset, train_images, train_labels, test_images, test_labels, test_masks) = load_lofar(args)

                model_name = model_names[i]
                args.set_name(model_name)
                ae = Autoencoder(args)
                p = 'outputs/{}/{}/{}/training_checkpoints/checkpoint_full_model_ae'.format(model_type, clss, model_name)
                ae.load_weights(p)

                if model_type == 'GANomaly':
                    encoder = tf.keras.Sequential(Encoder(args))
                    p = 'outputs/{}/{}/{}/training_checkpoints/checkpoint_full_model_encoder'.format(model_type, clss, model_name)
                    encoder.load_weights(p)
                    model = [ae,None,encoder]

                elif model_type == 'DAE_disc':
                    disc = Discriminator_x(args)
                    p = 'outputs/{}/{}/{}/training_checkpoints/checkpoint_full_model_disc'.format(model_type, clss, model_name)
                    disc.load_weights(p)
                    model = [ae,disc]

                else: model = [ae]
                
                x_hat  = infer(model[0], test_images, args, 'AE')
                x_hat_train  = infer(model[0], train_images, args, 'AE')
                z_query = infer(model[0].encoder, test_images, args, 'encoder') 
                z = infer(model[0].encoder, train_images, args, 'encoder')

                error = get_error('AE', test_images, x_hat, mean=False) 

                for nneigh in args.neighbors:
                    neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask = nln(z, 
                                                                                       z_query, 
                                                                                       x_hat_train, 
                                                                                       args.algorithm, 
                                                                                       nneigh, 
                                                                                       radius=None)
                    nln_error = get_nln_errors(model,
                                               model_type,
                                               z_query,
                                               z,
                                               test_images,
                                               x_hat_train,
                                               neighbours_idx,
                                               neighbour_mask,
                                               args)


                    dists = get_dists(neighbours_dist, args)

                    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.00001, novelty=True)
                    clf.fit(z);
                    lof = abs(clf.score_samples(z_query))

                    svm_model = OneClassSVM(kernel='poly',gamma='auto')
                    svm_model.fit(z);
                    svm = svm_model.score_samples(z_query)
                          

                    IF_model = IsolationForest(random_state=0,contamination=0)
                    IF_model.fit(z);
                    IF = abs(IF_model.score_samples(z_query))

                    ####### DETECTION

                    if args.anomaly_type == 'MISO':
                        AUC_lof = roc_auc_score(test_labels== args.anomaly_class, lof)
                        lofs.append(AUC_lof)

                        AUC_svm= roc_auc_score(test_labels== args.anomaly_class, svm)
                        svms.append(AUC_svm)

                        AUC_IF = roc_auc_score(test_labels== args.anomaly_class, IF)
                        ifs.append(AUC_IF)

                    if args.anomaly_type == 'SIMO':
                        AUC_detection = roc_auc_score(test_labels!= args.anomaly_class, add_norm.flatten())
                        detections.append(AUC_detection)

                        AUC_lof = roc_auc_score(test_labels!= args.anomaly_class, lof)
                        lofs.append(AUC_lof)

                        AUC_svm= roc_auc_score(test_labels!= args.anomaly_class, svm)
                        svms.append(AUC_svm)

                        AUC_IF = roc_auc_score(test_labels!= args.anomaly_class, IF)
                        ifs.append(AUC_IF)
                    nln_norm = process(nln_error, per_image=False)
                    recon_norm = process(error, per_image=False)
                    dists_norm = process(dists, per_image=False)

                    error_agg = aggregate(recon_norm,method='max') 
                    nln_error_agg = aggregate(nln_norm, method='max')
                    dists_agg = aggregate(dists_norm, method='max')

                    for alpha in [0.00, 0.25, 0.50, 0.75, 1.00]:
                        add_norm = (1-alpha)*nln_error_agg + alpha*dists_agg

                        if args.anomaly_type == 'MISO':
                            AUC_detection = roc_auc_score(test_labels== args.anomaly_class, add_norm.flatten())
                            detections.append(AUC_detection)

                        if args.anomaly_type == 'SIMO':
                            AUC_detection = roc_auc_score(test_labels!= args.anomaly_class, add_norm.flatten())
                            detections.append(AUC_detection)

                        df_out.loc[indxer] = (model_type, args.anomaly_class, ld, nneigh, alpha, detections[-1], lofs[-1], ifs[-1], svms[-1]) 
                        indxer += 1

                    n_arr.append(nneigh)
                    results[clss]  =  {'neighbour': n_arr,
                                       'detection':detections,
                                       'lof':lofs,
                                       'svm':svms,
                                       'IF':ifs}
                    print(results[clss])
        filename = 'outputs/{}_{}_{}.csv'.format(args.data,model_type,cmd_args.seed)
        df_out.to_csv(filename)
#    find_best(models,cmd_args.seed)

def aggregate(xs, method='avg'):
    y = np.empty(xs.shape[0])
    if method =='avg':
        for i,x in enumerate(xs):
            y[i] = np.mean(x)
    elif method == 'max':
        for i,x in enumerate(xs):
            y[i] = np.max(x)
    elif method == 'med':
        for i,x in enumerate(xs):
            y[i] = np.median(x)
    return y

def find_best(models,seed):
    results = {}
    for model_type in models:
        filename = 'outputs/{}_{}_{}.csv'.format(args.data,model_type,seed)
        d = np.load(filename, allow_pickle=True)
        df = pd.DataFrame(columns = ['class', 'neighbour', 'detection'])
        for key in d.keys():
            df_temp = pd.DataFrame(d[key], columns =  ['class', 'neighbour', 'detection'])
            df_temp['class'] = key
            df = df.append(df_temp)

        df_group = df.groupby(['neighbour']).agg({'detection':'mean'}).reset_index()

        results[model_type] = [df_group.detection.max()]


    print(results)
                

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


if __name__ == '__main__':
    main(args)
