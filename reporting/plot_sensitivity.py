from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np

import matplotlib
matplotlib.rcParams.update({'font.size': 12})
df = pd.read_csv('outputs/results_LOFAR_06-09-2022-02-31_ff1cc4.csv')

plt.close('all')
fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(projection='3d')
ax = fig.add_subplot()

xs = list(pd.unique(df.Latent_Dim))
xs.sort(reverse=True)
ys = np.sort(list(pd.unique(df.Patch_Size)))
zs = np.sort(list(pd.unique(df.Neighbour)))

output = np.zeros([len(xs), len(ys)])

vals =  df['COMBINED_AUROC_TRUE'].values
norm = matplotlib.colors.Normalize(vmin=vals.min(), vmax=vals.max())
z = 20
for ix,x in enumerate(xs):
    for iy,y in enumerate(ys):
            val = df[(df.Latent_Dim == x)&( df.Patch_Size == y)&(df.Neighbour == z)]['COMBINED_AUROC_TRUE'].values
            output[ix,iy] = val
            #im = ax.scatter(x, y, z, c=val, norm=norm, cmap='viridis')

im = ax.imshow(output,aspect='auto',interpolation='nearest')

ax.set_ylabel('Latent Dimensions',fontsize=12)
ax.set_yticks([0,1,2,3,4],xs,fontsize=12)

ax.set_xlabel('Patch Size',fontsize=12)
ax.set_xticks([0,1,2,3],ys,fontsize=12)

#ax.set_ylabel('# Neighbours')
#ax.set_yticks(zs)
#ax.set_yticklabels(zs,fontsize=8)

fig.colorbar(im)#, shrink=0.9, aspect=10,pad = 0.2)
plt.tight_layout()
plt.savefig('/tmp/AUROC', dpi=300)

########## 2
plt.close('all')
fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(projection='3d')
ax = fig.add_subplot()

vals =  df['COMBINED_AUPRC_TRUE'].values
norm = matplotlib.colors.Normalize(vmin=vals.min(), vmax=vals.max())


output = np.zeros([len(xs), len(ys)])
z=20
for ix,x in enumerate(xs):
    for iy,y in enumerate(ys):
            val = df[(df.Latent_Dim == x)&( df.Patch_Size == y)&(df.Neighbour == z)]['COMBINED_AUPRC_TRUE'].values
            output[ix,iy] = val
            #im = ax.scatter(x, y, z, c=val, norm=norm, cmap='viridis')

im = ax.imshow(output,aspect='auto',interpolation='nearest')

ax.set_ylabel('Latent Dimensions',fontsize=12)
ax.set_yticks([0,1,2,3,4],xs,fontsize=12)

ax.set_xlabel('Patch Size',fontsize=12)
ax.set_xticks([0,1,2,3],ys,fontsize=12)

#ax.set_ylabel('# Neighbours')
#ax.set_yticks(zs)
#ax.set_yticklabels(zs,fontsize=8)

fig.colorbar(im)#, shrink=0.9, aspect=10,pad = 0.2)
plt.tight_layout()
plt.savefig('/tmp/AUPRC', dpi=300)
#
########## 3
plt.close('all')
fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(projection='3d')
ax = fig.add_subplot()

vals =  df['COMBINED_F1_TRUE'].values
norm = matplotlib.colors.Normalize(vmin=vals.min(), vmax=vals.max())


output = np.zeros([len(xs), len(ys)])
z=20
for ix,x in enumerate(xs):
    for iy,y in enumerate(ys):
            val = df[(df.Latent_Dim == x)&( df.Patch_Size == y)&(df.Neighbour == z)]['COMBINED_F1_TRUE'].values
            output[ix,iy] = val
            #im = ax.scatter(x, y, z, c=val, norm=norm, cmap='viridis')

im = ax.imshow(output,aspect='auto',interpolation='nearest')

ax.set_ylabel('Latent Dimensions',fontsize=12)
ax.set_yticks([0,1,2,3,4],xs,fontsize=12)

ax.set_xlabel('Patch Size',fontsize=12)
ax.set_xticks([0,1,2,3],ys,fontsize=12)

#ax.set_ylabel('# Neighbours')
#ax.set_yticks(zs)
#ax.set_yticklabels(zs,fontsize=8)

fig.colorbar(im)#, shrink=0.9, aspect=10,pad = 0.2)
plt.tight_layout()
plt.savefig('/tmp/F1', dpi=300)
#
######### 3
#plt.close('all')
#fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(projection='3d')
#
#vals =  df['COMBINED_F1_TRUE'].values
#norm = matplotlib.colors.Normalize(vmin=vals.min(), vmax=vals.max())
#
#for x in xs:
#    for y in ys:
#        for z in zs:
#                val = df[(df.Latent_Dim == x)&( df.Patch_Size == y)&(df.Neighbour == z)]['COMBINED_F1_TRUE'].values
#                im = ax.scatter(x, y, z, c=val, norm=norm, cmap='viridis')
#
#ax.set_xlabel('Latent Dimensions')
#ax.set_xticks(xs)
#ax.set_xticklabels(xs,fontsize=8)
#
#ax.set_ylabel('Patch Size')
#ax.set_yticks(ys)
#ax.set_yticklabels(ys,fontsize=8)
#
#ax.set_zlabel('# Neighbours')
#ax.set_zticks(zs)
#ax.set_zticklabels(zs,fontsize=8)
#
#fig.colorbar(im, shrink=0.5, aspect=10, pad = 0.2)
##plt.tight_layout()
#plt.savefig('/tmp/F1', dpi=300)
