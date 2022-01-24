import numpy as np
from sklearn.model_selection import train_test_split


def get_hera_data(args):
    data, labels, masks, _ =  np.load(args.data_path, allow_pickle=True)

    data = np.expand_dims(data, axis=-1)
    masks = np.expand_dims(masks,axis=-1) 

    (train_data, test_data, 
     train_labels, test_labels, 
     train_masks, test_masks) = train_test_split(data, 
                                                 labels, 
                                                 masks,
                                                 test_size=0.25, 
                                                 random_state=42)
    return (train_data, test_data, train_labels, test_labels, train_masks, test_masks)
