import numpy as np
from sklearn.model_selection import train_test_split


rfi_models = ['rfi_stations', 'rfi_dtv', 'rfi_impulse', 'rfi_scatter']

def get_hera_data(args):
    if args.rfi is not None:
        (test_data, 
         test_labels, 
         test_masks)  =  np.load('/home/mmesarcik/data/HERA/HERA_24-02-2022_{}.pkl'.format(args.rfi),
                                                                                   allow_pickle=True)
        rfi_models.remove(args.rfi)

        (train_data, 
         train_labels, 
         train_masks)  =  np.load('/home/mmesarcik/data/HERA/HERA_24-02-2022_{}.pkl'.format('-'.join(rfi_models)),
                                                                                            allow_pickle=True)

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

    return (train_data, test_data, train_labels, test_labels, train_masks, test_masks)
