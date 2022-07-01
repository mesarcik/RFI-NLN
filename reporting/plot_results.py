import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def swap_index(df,i0,i1):
    b, c = df.iloc[i0], df.iloc[i1]
               
    temp = df.iloc[i0].copy()
    df.iloc[i0] = c
    df.iloc[i1] = temp
    return df

#df = pd.read_csv('outputs/results_HERA_05-01-2022-01-57_860ccb.csv').iloc[:60]
df = pd.read_csv('outputs/results_HERA_OOD_Final.csv')
df = df.groupby(['Model','Patch_Size','Neighbour','OOD_RFI'],as_index=False).agg({'AUROC_TRUE':['mean','std'],
                                                                                  'AUPRC_TRUE':['mean','std'],
                                                                                  'F1_TRUE':['mean','std'],
                                                                                  'NLN_AUROC_TRUE':['mean','std'],
                                                                                  'NLN_AUPRC_TRUE':['mean','std'],
                                                                                  'NLN_F1_TRUE':['mean','std']})
# Swap Impulse with DTV
df = swap_index(df,0,1)
df = swap_index(df,4,5)
df = swap_index(df,8,9)
df = swap_index(df,12,13)
df = swap_index(df,16,17)
df = swap_index(df,20,21)


ind = np.arange(len(pd.unique(df.OOD_RFI)))  # the x locations for the groups
width = 0.2  # the width of the bars

fig, axs = plt.subplots(3,1, figsize=(5,7), sharex=True)

for model in list(pd.unique(df.Model)):
    if model == 'DAE_disc':
        _filter = (df.Model == model) & (df.Patch_Size==32) & (df.Neighbour==10)
        _label='NLN'
        _auroc, _auprc, _f1 = 'NLN_AUROC_TRUE', 'NLN_AUPRC_TRUE', 'NLN_F1_TRUE'

        nln_rects0 = axs[0].bar(ind - width, df[_filter][_auroc]['mean'], width, yerr=df[_filter][_auroc]['std'], label=_label)
        nln_rects1 = axs[1].bar(ind - width, df[_filter][_auprc]['mean'], width, yerr=df[_filter][_auprc]['std'], label=_label)
        nln_rects2 = axs[2].bar(ind - width, df[_filter][_f1]['mean'], width, yerr=df[_filter][_f1]['std'], label=_label)

    elif model == 'UNET':
        _filter = (df.Model == model) & (df.Patch_Size==32) 
        _label='UNET'
        _auroc, _auprc, _f1 = 'AUROC_TRUE', 'AUPRC_TRUE', 'F1_TRUE'
        unet_rects0 = axs[0].bar(ind , df[_filter][_auroc]['mean'], width, yerr=df[_filter][_auroc]['std'], label=_label)
        unet_rects1 = axs[1].bar(ind , df[_filter][_auprc]['mean'], width, yerr=df[_filter][_auprc]['std'], label=_label)
        unet_rects2 = axs[2].bar(ind , df[_filter][_f1]['mean'], width, yerr=df[_filter][_f1]['std'], label=_label)

    elif model == 'AOFlagger':
        _filter = (df.Model == model) 
        _label='AOFlagger'
        _auroc, _auprc, _f1 = 'AUROC_TRUE', 'AUPRC_TRUE', 'F1_TRUE'
        aoflagger_rects0 = axs[0].bar(ind + width, df[_filter][_auroc]['mean'], width, yerr=df[_filter][_auroc]['std'], label=_label)
        aoflagger_rects1 = axs[1].bar(ind + width, df[_filter][_auprc]['mean'], width, yerr=df[_filter][_auprc]['std'], label=_label)
        aoflagger_rects2 = axs[2].bar(ind + width, df[_filter][_f1]['mean'], width, yerr=df[_filter][_f1]['std'], label=_label)

axs[0].set_ylabel('AUROC',fontsize=12)
#axs[0].set_xlabel('OOD RFI',fontsize=12)
#axs[0].set_xticks(ind)
#axs[0].set_xticklabels(['Narrow-band Burst','Blips','Broad-band Transient', 'Broad-band Continuous'],rotation=20, ha='right',fontsize=12)
axs[0].grid()

axs[1].set_ylabel('AUPRC',fontsize=12)
#axs[1].set_xlabel('OOD RFI',fontsize=12)
#axs[1].set_xticks(ind)
#axs[1].set_xticklabels(['Narrow-band Burst','Blips','Broad-band Transient', 'Broad-band Continuous'],rotation=20, ha='right',fontsize=12)
axs[1].grid()

axs[2].set_ylabel('F1 Score',fontsize=12)
axs[2].set_xlabel('OOD RFI',fontsize=12)
axs[2].set_xticks(ind)
axs[2].set_xticklabels(['Blips','Narrow-band Burst','Broad-band Transient', 'Broad-band Continuous'],rotation=12, ha='right',fontsize=12)
axs[2].grid()

axs[2].legend(handles = [nln_rects1,unet_rects1,aoflagger_rects1] , labels=['NLN', 'UNET','AOFlagger'], ncol=3, loc='lower center')
#loc='upper center',bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False,
plt.tight_layout()

plt.savefig('/tmp/OOD',dpi=300)

##########
# Threhsoldls
#
#######

df = pd.read_csv('outputs/results_HERA_Thresholds_Final.csv')
df = df.groupby(['Model','Patch_Size','Neighbour','RFI_Threshold'],as_index=False).agg({'AUROC_TRUE':['mean','std'],
                                                                                  'AUPRC_TRUE':['mean','std'],
                                                                                  'F1_TRUE':['mean','std'],
                                                                                  'NLN_AUROC_TRUE':['mean','std'],
                                                                                  'NLN_AUPRC_TRUE':['mean','std'],
                                                                                  'NLN_F1_TRUE':['mean','std']})
thresholds = list(pd.unique(df.RFI_Threshold))
ind = np.arange(len(pd.unique(df.RFI_Threshold)))  # the x locations for the groups
width = 0.2  # the width of the bars

fig, axs = plt.subplots(1,3, figsize=(15,5))

for model in list(pd.unique(df.Model)):
    if model == 'DAE_disc':
        _filter = (df.Model == model) & (df.Patch_Size==32) & (df.Neighbour==10)
        _label='NLN'
        _auroc, _auprc, _f1 = 'NLN_AUROC_TRUE', 'NLN_AUPRC_TRUE', 'NLN_F1_TRUE'

        nln_rects0 = axs[0].bar(ind - width, df[_filter][_auroc]['mean'], width, yerr=df[_filter][_auroc]['std'], label=_label)
        nln_rects1 = axs[1].bar(ind - width, df[_filter][_auprc]['mean'], width, yerr=df[_filter][_auprc]['std'], label=_label)
        nln_rects2 = axs[2].bar(ind - width, df[_filter][_f1]['mean'], width, yerr=df[_filter][_f1]['std'], label=_label)

    elif model == 'UNET':
        _filter = (df.Model == model) & (df.Patch_Size==32) 
        _label='UNET'
        _auroc, _auprc, _f1 = 'AUROC_TRUE', 'AUPRC_TRUE', 'F1_TRUE'
        unet_rects0 = axs[0].bar(ind , df[_filter][_auroc]['mean'], width, yerr=df[_filter][_auroc]['std'], label=_label)
        unet_rects1 = axs[1].bar(ind , df[_filter][_auprc]['mean'], width, yerr=df[_filter][_auprc]['std'], label=_label)
        unet_rects2 = axs[2].bar(ind , df[_filter][_f1]['mean'], width, yerr=df[_filter][_f1]['std'], label=_label)

    elif model == 'AOFlagger':
        _filter = (df.Model == model) 
        _label='AOFlagger'
        _auroc, _auprc, _f1 = 'AUROC_TRUE', 'AUPRC_TRUE', 'F1_TRUE'
        aoflagger_rects0 = axs[0].bar(ind + width, df[_filter][_auroc]['mean'], width, yerr=df[_filter][_auroc]['std'], label=_label)
        aoflagger_rects1 = axs[1].bar(ind + width, df[_filter][_auprc]['mean'], width, yerr=df[_filter][_auprc]['std'], label=_label)
        aoflagger_rects2 = axs[2].bar(ind + width, df[_filter][_f1]['mean'], width, yerr=df[_filter][_f1]['std'], label=_label)

axs[0].set_ylabel('AUROC',fontsize=12)
axs[0].set_xlabel('AOFlagger Thresholds',fontsize=12)
axs[0].set_xticks(ind)
axs[0].set_xticklabels(thresholds,rotation=20, ha='right',fontsize=12)
axs[0].grid()

axs[1].set_ylabel('AUPRC',fontsize=12)
axs[1].set_xlabel('AOFlagger Thresholds',fontsize=12)
axs[1].set_xticks(ind)
axs[1].set_xticklabels(thresholds,rotation=20, ha='right',fontsize=12)
axs[1].grid()

axs[2].set_ylabel('F1 Score',fontsize=12)
axs[2].set_xlabel('AOFlagger Thresholds',fontsize=12)
axs[2].set_xticks(ind)
axs[2].set_xticklabels(thresholds,rotation=20, ha='right',fontsize=12)
axs[2].grid()

axs[1].legend(handles = [nln_rects1,unet_rects1,aoflagger_rects1] , labels=['NLN', 'UNET','AOFlagger'], ncol=3)
#loc='upper center',bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False,
plt.tight_layout()

plt.savefig('/tmp/Thresholds',dpi=300)
