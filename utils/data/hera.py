import numpy as np
from sklearn.model_selection import train_test_split


rfi_models = ['rfi_stations', 'rfi_dtv', 'rfi_impulse', 'rfi_scatter']

def get_hera_data(args):
    if args.rfi is not None:
        (_,_,test_data, 
         test_masks)  =  np.load('{}/HERA_28-07-2023_{}.pkl'.format(args.data_path,
                                                                    args.rfi),
                                                                    allow_pickle=True)
        rfi_models.remove(args.rfi)

        (train_data, 
         train_masks,_,_)  =  np.load('{}/HERA_28-07-2023_{}.pkl'.format(args.data_path, 
                                                                        '-'.join(rfi_models)),
                                                                         allow_pickle=True)

    else:
        (train_data, train_masks, 
          test_data, test_masks) = np.load('{}/HERA_28-07-2023_all.pkl'.format(args.data_path), 
                                                                             allow_pickle=True)
    train_data[train_data==np.inf] = np.finfo(train_data.dtype).max
    test_data[test_data==np.inf] = np.finfo(test_data.dtype).max


    return (train_data.astype('float32'), test_data.astype('float32'),  train_masks, test_masks)
