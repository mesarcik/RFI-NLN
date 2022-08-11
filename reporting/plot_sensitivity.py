from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np

import matplotlib
matplotlib.rcParams.update({'font.size': 12})
#df = pd.read_csv('outputs/results_LOFAR_06-09-2022-02-31_ff1cc4.csv')
df = pd.read_csv('outputs/results_LOFAR_07-27-2022-11-54_d3f418.csv')
#df = df[df.Latent_Dim != 2]


xs = list(pd.unique(df.Latent_Dim))
xs.sort(reverse=True)
ys = np.sort(list(pd.unique(df.Patch_Size)))
zs = np.sort(list(pd.unique(df.Neighbour)))
z = 16


for metric  in ['COMBINED_AUROC_TRUE', 'COMBINED_AUPRC_TRUE', 'COMBINED_F1_TRUE']:
    plt.close('all')
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    output = np.zeros([len(xs), len(ys)])

    vals =  df[metric].values
    norm = matplotlib.colors.Normalize(vmin=vals.min(), vmax=vals.max())
    for ix,x in enumerate(xs):
        for iy,y in enumerate(ys):
                val = df[(df.Latent_Dim == x)&( df.Patch_Size == y)&(df.Neighbour == z)][metric].values
                output[ix,iy] = val
                #im = ax.scatter(x, y, z, c=val, norm=norm, cmap='viridis')
    if metric == 'COMBINED_AUROC_TRUE':
        _min, _max  = 0.65, 0.865
    elif metric == 'COMBINED_AUPRC_TRUE':
        _min, _max  = 0.35, 0.65
    else:
        _min, _max  = 0.35 , 0.53
    im = ax.imshow(output,aspect='auto',interpolation='nearest', vmin =_min, vmax=_max)

    ax.set_ylabel('Latent Dimensions',fontsize=12)
    ax.set_yticks(np.arange(0,len(xs)),xs,fontsize=12)

    ax.set_xlabel('Patch Size',fontsize=12)
    ax.set_xticks(np.arange(0,len(ys)),ys,fontsize=12)

    #ax.set_ylabel('# Neighbours')
    #ax.set_yticks(zs)
    #ax.set_yticklabels(zs,fontsize=8)

    fig.colorbar(im)#, shrink=0.9, aspect=10,pad = 0.2)
    plt.tight_layout()
    plt.savefig('/tmp/{}'.format(metric.split('_')[1]), dpi=300)
##############################################################
#df = df[(df.Latent_Dim != 2)]# & (df.Latent_Dim != 256)]

#xs = list(pd.unique(df.Latent_Dim))
ys = np.sort(list(pd.unique(df.Patch_Size)))
x=32
zs = list(pd.unique(df.Neighbour))



for metric  in ['COMBINED_AUROC_TRUE', 'COMBINED_AUPRC_TRUE', 'COMBINED_F1_TRUE']:
    plt.close('all')
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    output = np.zeros([len(zs), len(ys)])

    vals =  df[metric].values
    norm = matplotlib.colors.Normalize(vmin=vals.min(), vmax=vals.max())
    for iy,y in enumerate(ys):
        for iz,z in enumerate(zs):
                val = df[(df.Latent_Dim == x)&( df.Patch_Size == y)&(df.Neighbour == z)][metric].values
                output[iz,iy] = val
                #im = ax.scatter(x, y, z, c=val, norm=norm, cmap='viridis')
    if metric == 'COMBINED_AUROC_TRUE':
        _min, _max  = 0.65, 0.865
    elif metric == 'COMBINED_AUPRC_TRUE':
        _min, _max  = 0.35, 0.65
    else:
        _min, _max  = 0.35 , 0.53

    im = ax.imshow(output,aspect='auto',interpolation='nearest', vmin =_min, vmax=_max)

    ax.set_xlabel('Patch Size',fontsize=12)
    ax.set_xticks(np.arange(0,len(ys)),ys,fontsize=12)

    ax.set_ylabel('# Neighbours',fontsize=12)
    ax.set_yticks(np.arange(0,len(zs)),zs[::-1],fontsize=12)

    #ax.set_ylabel('# Neighbours')
    #ax.set_yticks(zs)
    #ax.set_yticklabels(zs,fontsize=8)

    fig.colorbar(im)#, shrink=0.9, aspect=10,pad = 0.2)
    plt.tight_layout()
    plt.savefig('/tmp/{}_neighbours'.format(metric.split('_')[1]), dpi=300)

