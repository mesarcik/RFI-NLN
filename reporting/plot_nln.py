from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np
from models import Autoencoder
from utils.metrics import nln, get_nln_errors, get_dists
from inference import infer
from data import load_lofar
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
from utils.data import patches 

def make_grid(img):
    dx, dy = 32,32
    grid_color = -1
    img[:,::dy] = grid_color
    img[::dx,:] = grid_color
    return img

class Namespace:
     def __init__(self, **kwargs):
         self.__dict__.update(kwargs)
PATCH  = 32
LD =32
args = Namespace(input_shape=(PATCH, PATCH, 1),
              rotate=False,
              rfi_threshold=None,
              rfi=0,
              data_path ='/home/mmesarcik/data/LOFAR/uncompressed/RFI/',
              crop=False,
              epochs=-1,
              patches=True,
              percentage_anomaly=None,
              limit= None,
              patch_x = PATCH,
              patch_y=PATCH,
              patch_stride_x = PATCH,
              patch_stride_y = PATCH,
              crop_x=PATCH,
              crop_y=PATCH,
              latent_dim=LD,
              # NLN PARAMS
              data ='LOFAR',
              anomaly_class='rfi',
              anomaly_type='MISO',
              radius= [10],
              alphas= [0],
              neighbors= [5],
              algorithm = 'knn')

ae = Autoencoder(args)
ae.load_weights('outputs/DAE_disc/rfi/fuzzy-charming-skink-of-rain/training_checkpoints/checkpoint_full_model_ae')
(_,_train_data,_train_labels,_train_masks,train_dataset,
    train_data,train_labels,test_data,test_labels,test_masks,test_masks) = load_lofar(args)

z = infer(ae.encoder, train_data, args, 'encoder')
z_query = infer(ae.encoder, test_data, args, 'encoder')

x_hat_train  = infer(ae, train_data, args, 'AE')
x_hat = infer(ae, test_data, args, 'AE')
x_hat_recon = patches.reconstruct(x_hat, args)

neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask =  nln(z, z_query, x_hat_train, 'knn', 20, -1)
nln_error = get_nln_errors([ae],'AE',z_query,z,test_data, x_hat_train,neighbours_idx, neighbour_mask, args)
nln_error_recon = patches.reconstruct(nln_error, args)

test_masks_recon = patches.reconstruct(test_masks, args)
test_data_recon = patches.reconstruct(test_data, args)
dists_recon = get_dists(neighbours_dist, args)
neighbours = x_hat_train[neighbours_idx]

neighbour_0 = neighbours[:,0,...]
neighbour_0_recon = patches.reconstruct(neighbour_0, args)

neighbour_10 = neighbours[:,10,...]
neighbour_10_recon = patches.reconstruct(neighbour_10, args)

neighbour_19 = neighbours[:,19,...]
neighbour_19_recon = patches.reconstruct(neighbour_19, args)

combined_recon =  np.clip(nln_error_recon,nln_error_recon.mean()+nln_error_recon.std()*5,1.0)*np.array([d > np.percentile(d,66) for d in dists_recon])
precision, recall, thresholds = precision_recall_curve(test_masks_recon.flatten()>0, combined_recon.flatten())
f1_scores = 2*recall*precision/(recall+precision)
indx_max = np.argmax(f1_scores)
t = thresholds[indx_max]

fs =12
ind = 30

fig, axs = plt.subplots(figsize=(5,5))
_max = np.max(test_data_recon[ind,...,0])/2
axs.imshow(make_grid(test_data_recon[ind,...,0].T),    vmin=0, vmax=_max,interpolation='nearest', aspect='auto')
axs.set_xlabel('Time [s]',fontsize=fs)
axs.set_ylabel('Cropped Frequency Bins',fontsize=fs)
plt.tight_layout()
plt.savefig('/tmp/tmp/input'.format(ind),dpi=300)

fig, axs = plt.subplots(figsize=(5,5))
axs.imshow(make_grid(neighbour_0_recon[ind,...,0].T),  vmin=0, vmax=_max,interpolation='nearest', aspect='auto')
axs.set_xlabel('Time [s]',fontsize=fs)
axs.set_ylabel('Cropped Frequency Bins',fontsize=fs)
plt.tight_layout()
plt.savefig('/tmp/tmp/neighbour_0'.format(ind),dpi=300)

fig, axs = plt.subplots(figsize=(5,5))
axs.imshow(make_grid(neighbour_10_recon[ind,...,0].T), vmin=0, vmax=_max,interpolation='nearest', aspect='auto')
axs.set_xlabel('Time [s]',fontsize=fs)
axs.set_ylabel('Cropped Frequency Bins',fontsize=fs)
plt.tight_layout()
plt.savefig('/tmp/tmp/neighbour_10'.format(ind),dpi=300)

fig, axs = plt.subplots(figsize=(5,5))
axs.imshow(make_grid(neighbour_19_recon[ind,...,0].T), vmin=0, vmax=_max,interpolation='nearest', aspect='auto')
axs.set_xlabel('Time [s]',fontsize=fs)
axs.set_ylabel('Cropped Frequency Bins',fontsize=fs)
plt.tight_layout()
plt.savefig('/tmp/tmp/neighbour_19'.format(ind),dpi=300)

fig, axs = plt.subplots(figsize=(5,5))
axs.imshow(make_grid(nln_error_recon[ind,...,0].T),    vmin=0, vmax=_max,interpolation='nearest', aspect='auto')
axs.set_xlabel('Time [s]',fontsize=fs)
axs.set_ylabel('Cropped Frequency Bins',fontsize=fs)
plt.tight_layout()
plt.savefig('/tmp/tmp/nln_error'.format(ind),dpi=300)

fig, axs = plt.subplots(figsize=(5,5))
axs.imshow(make_grid(dists_recon[ind,...,0].T), interpolation='nearest', aspect='auto')
axs.set_xlabel('Time [s]',fontsize=fs)
axs.set_ylabel('Cropped Frequency Bins',fontsize=fs)
plt.tight_layout()
plt.savefig('/tmp/tmp/dists'.format(ind),dpi=300)

fig, axs = plt.subplots(figsize=(5,5))
axs.imshow(combined_recon[ind,...,0].T>t,vmin=0.0, vmax=_max, interpolation='nearest', aspect='auto')
axs.set_xlabel('Time [s]',fontsize=fs)
axs.set_ylabel('Cropped Frequency Bins',fontsize=fs)
plt.tight_layout()
plt.savefig('/tmp/tmp/combined'.format(ind),dpi=300)

fig, axs = plt.subplots(figsize=(5,5))
axs.imshow((x_hat_recon[ind,...,0].T), interpolation='nearest', aspect='auto')
axs.set_xlabel('Time [s]',fontsize=fs)
axs.set_ylabel('Cropped Frequency Bins',fontsize=fs)
plt.tight_layout()
plt.savefig('/tmp/tmp/x_hat'.format(ind),dpi=300)
