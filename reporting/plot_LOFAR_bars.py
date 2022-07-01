import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rcParams.update({'font.size': 12})


labels = ['Full Dataset', 'L629174', 'L631961']

nln_mean = [0.8555, 0.8420, 0.8590]
nln_std =  [0.0040, 0.0009, 0.0026]

unet_mean, rfinet_mean, rnet_mean  = [0.8017, 0.7462, 0.6949], [0.8109, 0.7638, 0.7650], [0.8262, 0.7842, 0.6791]
unet_std, rfinet_std, rnet_std =     [0.0058, 0.0421, 0.1303], [0.0037, 0.0028, 0.0036], [0.0072, 0.0422, 0.1551]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(5,3))
rects3 = ax.bar(x, rfinet_mean, yerr=rfinet_std,       width=width, label='RFI-Net')
rects1 = ax.bar(x - 2*width, nln_mean, yerr=nln_std, width=width, label='NLN')
rects2 = ax.bar(x - 1*width, unet_mean, yerr=unet_std, width=width, label='U-Net')
rects4 = ax.bar(x+ 1*width, rnet_mean, yerr=rnet_std,       width=width, label='RNet')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUROC', fontsize=12)
ax.set_xlabel('Training set', fontsize=12)
ax.set_xticks(x, labels)
ax.grid()
ax.legend(handles = [rects1,rects2,rects3,rects4] , labels=['NLN', 'UNET','RFI-Net','RNET'], loc='lower right')

#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig('/tmp/temp', dpi=300)
