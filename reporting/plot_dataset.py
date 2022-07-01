import numpy as np
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import image

train_data, train_masks, _, _= np.load('joined_dataset.pickle', allow_pickle=True)
r = np.load('training_indexes.npy')
_files = []
for n in tqdm(np.sort(glob('/tmp/flags/*.png'))):
    _files.append(int(n.split('/')[-1].split('.')[0]))
_files = np.array(_files)
r = r[_files]
_train_data, _train_masks = train_data[r], train_masks[r]

_train_data = np.clip(1.75e6, 50e6, _train_data)
_train_data = np.log(_train_data)


for i in [17,47]:
    f = _files[i]
    fig, axs = plt.subplots(figsize=(5,5))
    axs.imshow(_train_data[i,...,0].T,aspect='auto', interpolation='nearest');
    axs.set_xlabel('Time [s]',fontsize=12)
    axs.set_ylabel('Cropped Frequency Bins',fontsize=12)
    plt.tight_layout()
    plt.savefig('/tmp/LOFAR/{}_train_data'.format(i), dpi=300)
    plt.close('all')

    fig, axs = plt.subplots(figsize=(5,5))
    axs.imshow(np.invert(_train_masks[i,...,0].T>0), cmap='gray',aspect='auto', interpolation='nearest');
    axs.set_xlabel('Time [s]',fontsize=12)
    axs.set_ylabel('Cropped Frequency Bins',fontsize=12)
    plt.tight_layout()
    plt.savefig('/tmp/LOFAR/{}_train_mask'.format(i), dpi=300)
    plt.close('all')

    fig, axs = plt.subplots(figsize=(5,5))
    im =  np.array(Image.open('/tmp/flags/{}.png'.format(f)))
    axs.imshow(np.invert(im.T>0), cmap='gray',aspect='auto', interpolation='nearest');
    axs.set_xlabel('Time [s]',fontsize=12)
    axs.set_ylabel('Cropped Frequency Bins' ,fontsize=12)
    plt.tight_layout()
    plt.savefig('/tmp/LOFAR/{}_test_masks'.format(i),dpi=300)
    plt.close('all')
