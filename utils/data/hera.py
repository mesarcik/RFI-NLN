import numpy as np
from sklearn.model_selection import train_test_split


rfi_models = ['rfi_stations', 'rfi_dtv', 'rfi_impulse', 'rfi_scatter']

def get_hera_data(args):
    if args.rfi is not None:
        (test_data, 
         test_labels, 
         test_masks)  =  np.load('/home/mmesarcik/data/HERA/HERA_04-03-2022_{}.pkl'.format(args.rfi),
                                                                                   allow_pickle=True)
        test_data[test_data==np.inf] = np.finfo(test_data.dtype).max
        rfi_models.remove(args.rfi)

        (train_data, 
         train_labels, 
         train_masks)  =  np.load('/home/mmesarcik/data/HERA/HERA_04-03-2022_{}.pkl'.format('-'.join(rfi_models)),
                                                                                            allow_pickle=True)
        train_data[train_data==np.inf] = np.finfo(train_data.dtype).max

    else:
        data, labels, masks =  np.load(args.data_path, allow_pickle=True)
        data[data==np.inf] = np.finfo(data.dtype).max

        (train_data, test_data, 
         train_labels, test_labels, 
         train_masks, test_masks) = train_test_split(data, 
                                                     labels, 
                                                     masks,
                                                     test_size=0.25, 
                                                     random_state=42)

    train_data = np.concatenate([train_data, np.roll(train_data,
                                                     args.patch_x//2, 
                                                     axis =2)], axis=0)# this is less artihmetically complex then making stride half

    train_masks = np.concatenate([train_masks, np.roll(train_masks,
                                                       args.patch_x//2, 
                                                       axis =2)], axis=0)
    train_labels = np.concatenate([train_labels,
                                   train_labels],
                                   axis=0)
    return (train_data, test_data, train_labels, test_labels, train_masks, test_masks)
