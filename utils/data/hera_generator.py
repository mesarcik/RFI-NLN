import tempfile
import time
import pickle
import datetime
from tqdm import tqdm
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
from astropy import units

import hera_sim
from hera_sim import Simulator, DATA_PATH, utils
from uvtools.plot import labeled_waterfall

station_models = ['HERA_H1C_RFI_STATIONS.npy', 
                  'HERA_H2C_RFI_STATIONS.npy']

start_time = 2457458.1738949567 # JD
integration_time = 3.512#4.68
Ntimes = int(30 * units.min.to("s") / integration_time)# 10 minute observation

# Define the frequency parameters.
Nfreqs = 2**9
bandwidth = 88e6#8e7# 100 MHz
start_freq = 107e6# start at 100MHz


array_layout = hera_sim.antpos.hex_array(2, split_core=False, outriggers=0)

sim_params = dict(
    Nfreqs=Nfreqs,
    start_freq=start_freq,
    bandwidth=bandwidth,
    Ntimes=Ntimes,
    start_time=start_time,
    integration_time=integration_time,
    array_layout=array_layout,
)
sim = Simulator(**sim_params)

def waterfall(sim, antpairpol=(0,1,"xx"), figsize=(6,3.5), dpi=200, title=None):
    """Convenient plotting function to show amp/phase."""
    fig, (ax1, ax2) = plt.subplots(
                                   nrows=2,
                                   ncols=1,
                                   figsize=figsize,
                                   dpi=dpi,
                               )
    fig, ax1 = labeled_waterfall(
                                sim.data,
                                antpairpol=antpairpol,
                                mode="log",
                                ax=ax1,
                                set_title=title,
                                )
    ax1.set_xlabel(None)
    ax1.set_xticklabels(['' for tick in ax1.get_xticks()])
    fig, ax2 = labeled_waterfall(
                                 sim.data,
                                 antpairpol=antpairpol,
                                 mode="phs",
                                 ax=ax2,
                                 set_title=False,
                                 )
    plt.savefig('/tmp/temp/{}_{}_{}'.format(*antpairpol),dpi=300)


def simulate(id_rfis):
    """
        Adds parameters to the hera sim simulator class and generates the data

        Parameters
        ----------
        id_rfis (tuple/list)  the in distribution (ID) RFI

        Returns
        -------
        Simulator 

    """
    sim.refresh()
    hera_sim.defaults.set("h1c")

    sim.add(
    "diffuse_foreground",
    component_name="diffuse_foreground",
    seed="once",
    )

    #########################################################################################
    #########################################################################################
    #########################################################################################

    #if np.random.random(1)[0] >0.5:
    if 'rfi_stations' in id_rfis:
        sim.add(
            "rfi_stations",
            stations="/home/mmesarcik/anaconda3/envs/hera/lib/python3.7/site-packages/hera_sim/data/{}".format(station_models[
                                                                                                                np.random.randint(0,2)]),
            component_name="rfi_stations",
            seed="once",
        )

    if 'rfi_dtv' in id_rfis:
        sim.add(
        "rfi_dtv",
        dtv_band=(0.174, 0.214),  
        dtv_channel_width=0.08,
        dtv_chance=0.025,
        dtv_strength=20000,
        dtv_std=200.0,
        component_name="rfi_dtv",
        seed="once",
        )

    if 'rfi_impulse' in id_rfis:
        sim.add(
        "rfi_impulse",
        impulse_chance=0.005,  # A lot of sources
        impulse_strength=20000.00,
        component_name="rfi_impulse",
        seed="once",
        )

    if 'rfi_scatter' in id_rfis:
        sim.add(
        "rfi_scatter",
        scatter_chance=0.0008,  # A lot of sources
        scatter_strength=20000.00,
        scatter_std=200.0,
        component_name="rfi_scatter",
        seed="once",
        )

    #########################################################################################
    #########################################################################################
    #########################################################################################
    sim.add("thermal_noise",
            seed="initial",
            Trx=0,
            component_name="noisy_ant")

    sim.add("whitenoisecrosstalk", amplitude=1.0, seed="once")
    sim.add("bandpass",gain_spread=0.1, dly_rng= (-20, 20))


    #########################################################################################
    #########################################################################################
    #########################################################################################
    return sim


def extract_data(sim,baselines,subset):
    """
        Extracts the visibilities at each randomly sampled baseline

        Parameters
        ----------
        sim (Simulator) the previously instantiated simulator with all effects and features 
        baselines (int) number of baselines to be sampled
        subset (tuple) the list of rfi waveforms to be extracted from the simulator 

        Returns
        -------
        np.array, np.array, np.array

    """
    data,labels, masks  = [],[],[]
    _pairs = sim.get_antpairpols()
    auto_inds = np.array([i for i,p in enumerate(_pairs) if p[0]==p[1]])
    corr_pairs = [p for p in _pairs if p[0]!=p[1]]

    corr_inds = np.random.choice(range(len(corr_pairs,)), baselines, replace=False)# sample random baselines given by "baselines"
    inds = np.concatenate([auto_inds, corr_inds],axis=-1)

    for ind in inds:
        pairs = _pairs[ind]
        data_temp = np.absolute(sim.get_data(pairs)).astype('float16')
        data.append(data_temp)

        mask_temp = np.zeros(data_temp.shape, dtype='bool')
        label_temp = ''
        for rfi in subset:
            try:
                mask_temp = np.logical_or(mask_temp, 
                                          np.absolute(sim.get(rfi,pairs))>0) 
                label_temp = label_temp+'_{}'.format(rfi)
            except Exception  as e:
                continue 
        masks.append(mask_temp) 
        labels.append(label_temp[1:])

    data = np.expand_dims(np.array(data), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)

    return data, masks, labels 

def plot(sim):
    """
        Save waterfall plots of the magnitude and phase of the simulated data 

        Parameters
        ----------
        sim (Simulator) the previously instantiated simulator with all effects and features 

        Returns
        -------
        None

    """
    for i in sim.get_antpairpols():
        fig1 = waterfall(sim, antpairpol=i, title=' '.join(str(e) for e in i))
        plt.close('all')

def main():
    """
        Runs the simulator with different subsets of RFI and saves them as pickles

        Parameters
        ----------
        None

        Returns
        -------
        None
    """
    n =40
    baselines=7
    rfis = ['rfi_stations', 'rfi_dtv', 'rfi_impulse', 'rfi_scatter']
    for L in tqdm([1,3]):# to simulate IID and OOD RFI
        for subset in itertools.combinations(rfis,L): 
            data =  np.empty([2*n*baselines, 2**9, 2**9, 1], dtype='float16')
            masks =  np.empty([2*n*baselines, 2**9, 2**9, 1], dtype='bool')
            labels = np.empty([2*n*baselines],dtype=object)
            st, en = 0, 2*baselines
            for i in range(n):
                sim = simulate(subset)
                _data, _masks, _labels = extract_data(sim, baselines,subset)
                data[st:en,...], masks[st:en,...], labels[st:en]  = _data, _masks, _labels
                st=en
                en+=2*baselines

            f_name = '/home/mmesarcik/data/HERA/HERA_{}_{}.pkl'.format(datetime.datetime.now().strftime("%d-%m-%Y"),'-'.join(subset))
            print('{} saved!'.format(f_name))

            pickle.dump([data,labels,masks],open(f_name, 'wb'), protocol=4)

if __name__ == '__main__':
    main()

