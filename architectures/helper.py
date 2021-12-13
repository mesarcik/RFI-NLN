from utils.plotting import save_training_curves
from utils.metrics import get_classifcation, get_nln_metrics, save_metrics, accuracy_metrics

def end_routine(train_images, 
                test_images, 
                test_labels, 
                test_masks, 
                test_masks_orig, 
                model, 
                model_type, 
                args):

#    return None 
#    if model_type != 'DKNN':
#        save_training_curves(model,args,test_images,test_labels,model_type)
    (ae_ao_auroc,  ae_true_auroc, 
     ae_ao_auprc,  ae_true_auprc,      
     ae_ao_iou,    ae_true_iou,
     nln_ao_auroc, nln_true_auroc, 
     nln_ao_auprc, nln_true_auprc,      
     nln_ao_iou,   nln_true_iou,
     dists_ao_auroc, dists_true_auroc, 
     dists_ao_auprc, dists_true_auprc,      
     dists_ao_iou,   dists_true_iou,
     combined_ao_auroc, combined_true_auroc, 
     combined_ao_auprc, combined_true_auprc,      
     combined_ao_iou,   combined_true_iou) = accuracy_metrics(model,
                                                              train_images,
                                                              test_images,
                                                              test_labels,
                                                              test_masks,
                                                              test_masks_orig,
                                                              model_type,
                                                              args)

    

    save_metrics(model_type,
                 test_masks,
                 test_masks_orig,
                 args,

                 ae_ao_auroc=   ae_ao_auroc,
                 ae_true_auroc= ae_true_auroc,
                 ae_ao_auprc=   ae_ao_auprc,
                 ae_true_auprc= ae_true_auprc,
                 ae_ao_iou=     ae_ao_iou,
                 ae_true_iou=   ae_true_iou,

                 nln_ao_auroc=   nln_ao_auroc,
                 nln_true_auroc= nln_true_auroc,
                 nln_ao_auprc=   nln_ao_auprc,
                 nln_true_auprc= nln_true_auprc,
                 nln_ao_iou=     nln_ao_iou,
                 nln_true_iou=   nln_true_iou,

                 dists_ao_auroc=   dists_ao_auroc,
                 dists_true_auroc= dists_true_auroc,
                 dists_ao_auprc=   dists_ao_auprc,
                 dists_true_auprc= dists_true_auprc,
                 dists_ao_iou=     dists_ao_iou,
                 dists_true_iou=   dists_true_iou,

                 combined_ao_auroc=   combined_ao_auroc,
                 combined_true_auroc= combined_true_auroc,
                 combined_ao_auprc=   combined_ao_auprc,
                 combined_true_auprc= combined_true_auprc,
                 combined_ao_iou=     combined_ao_iou,
                 combined_true_iou=   combined_true_iou)

