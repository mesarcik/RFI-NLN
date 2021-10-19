from utils import save_training_curves
from utils.plotting import save_training_curves
from utils.metrics import get_classifcation, get_nln_metrics, save_metrics, accuracy_metrics

def end_routine(train_images, test_images, test_labels, test_masks, model, model_type, args):

#    return None 
    save_training_curves(model,args,test_images,test_labels,model_type)

    auc_latent, dists_auc, sum_auc, mul_auc = args.limit, -1, -1, -1
    neighbour =5
    radius =10
    #(auc_latent, neighbour, radius, dists_auc,
    #    sum_auc, mul_auc, x_hat, x_hat_train,
    #    neighbours_idx, neighbours_dist) = get_nln_metrics(model,   
    #                                                       train_images, 
    #                                                       test_images, 
    #                                                       test_labels, 
    #                                                       model_type, 
    #                                                       args)
    
    if args.data !='LOFAR':
        auc_recon = get_classifcation(model_type,
                                      model,
                                      test_images,
                                      test_labels,
                                      args)
    else: auc_recon = 'n/a'
    if (args.data == 'MVTEC') or (args.data == 'HERA') or (args.data == 'LOFAR'):

        (ae_auroc, ae_auprc, ae_iou, nln_auroc, nln_auprc, 
                nln_iou, dists_auroc, dists_auprc, dists_iou) = accuracy_metrics(model,
                                                                                 train_images,
                                                                                 test_images,
                                                                                 test_labels,
                                                                                 test_masks,
                                                                                 model_type,
                                                                                 neighbour,
                                                                                 radius,
                                                                                 args)

    

    save_metrics(model_type,
                 args,
                 ae_auroc=ae_auroc,
                 ae_auprc=ae_auprc,
                 ae_iou=ae_iou,
                 nln_auroc=nln_auroc,
                 nln_auprc=nln_auprc, 
                 nln_iou=nln_iou,
                 dists_auroc=dists_auroc,
                 dists_auprc=dists_auprc, 
                 dists_iou=dists_iou)

