from utils import save_training_curves
from utils.plotting import save_training_curves
from utils.metrics import get_classifcation, get_nln_metrics, save_metrics, accuracy_metrics

def end_routine(train_images, test_images, test_labels, test_masks, model, model_type, args):

#    return None 
    if model_type != 'DKNN':
        save_training_curves(model,args,test_images,test_labels,model_type)

    
    (ae_auroc, ae_auprc, ae_iou, nln_auroc, nln_auprc, 
            nln_iou, dists_auroc, dists_auprc, dists_iou,
            combined_auroc, combined_auprc, combined_iou) = accuracy_metrics(model,
                                                                             train_images,
                                                                             test_images,
                                                                             test_labels,
                                                                             test_masks,
                                                                             model_type,
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
                 dists_iou=dists_iou,
                 combined_auroc=combined_auroc,
                 combined_auprc=combined_auprc, 
                 combined_iou=combined_iou)

