#félix import
import os
import argparse
import torch
import torch.nn as nn
import sys
from tqdm import tqdm

#modan import
import numpy as np 
import h5py
import sys
import os
from pathlib import Path
import argparse
import numpy.lib.recfunctions as rfn
import time
import torch
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import maad

#for scikit maad tfsd
from numpy import sum, log, min, max, abs, mean, median, sqrt, diff, var
from maad.sound import (envelope, smooth, temporal_snr, linear_to_octave, 
                        avg_amplitude_spectro, avg_power_spectro, spectral_snr, 
                        median_equalizer)
from maad.util import (rle, index_bw, amplitude2dB, power2dB, dB2power, mean_dB,
                       skewness, kurtosis, format_features, into_bins, entropy, 
                       linear_scale, plot1d, plot2d, overlay_rois)

class CenseDataset(torch.utils.data.Dataset):
    def __init__(self, data, force_numpy=False):
        self.data = data[0,:,:]
        self.len_data = data.shape[1]
        self.force_numpy = force_numpy
    def __getitem__(self, idx):
        if self.force_numpy:
            input_spec = np.copy(self.data[idx])
        else:
            input_spec = torch.from_numpy(np.copy(self.data[idx]))
        return (input_spec)

    def __len__(self):
        return self.len_data

def inference_tfsd(spectral_data, batch_size=480, fn=np.array([20, 25, 31, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]),
                    ):
    # useCuda = torch.cuda.is_available() and not settings['training']['force_cpu']
    # if useCuda:
    #     print('Using CUDA.')
    #     dtype = torch.cuda.FloatTensor
    #     ltype = torch.cuda.LongTensor
    # else:
    #     print('No CUDA available.')
    #     dtype = torch.FloatTensor
    #     ltype = torch.LongTensor

    # Load datasets
    dataSpec = spectral_data
    mydataset = CenseDataset(dataSpec, force_numpy=True)
    mydataloader = torch.utils.data.DataLoader(mydataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
    tqdm_it=tqdm(mydataloader, 
                desc='EVAL')

    all_scores_voices = np.empty((0))
    all_scores_birds = np.empty((0))

    for x in tqdm_it:
        #x = x.type(dtype)
        #tn = np.array([k*0.125 for k in range(x.shape[0])])
        
        x = x.cpu().detach().numpy()
        #voices
        #score_voices = maad.features.tfsd(x,fn, tn, flim=(500,1500), mode='else') 
        scores_voices = tfsd(x,fn,flim=(500,1500)) 

        #birds
        #score_birds = maad.features.tfsd(x,fn, tn, flim=(1500,6000), mode='else') 
        scores_birds = tfsd(x,fn,flim=(1500,6000)) 

        scores_voices = np.full((x.shape[0]), scores_voices)
        scores_birds = np.full((x.shape[0]), scores_birds)

        all_scores_voices = np.concatenate((all_scores_voices, scores_voices))
        all_scores_birds = np.concatenate((all_scores_birds, scores_birds))

    scores = np.zeros((len(all_scores_voices), 3))
    scores[:, 1] = all_scores_voices
    scores[:, 2] = all_scores_birds

    #print(scores)
    return(scores)

def tfsd(x, fn, flim=(500,1500)):
    xt = x.T

    # back to amplitude spectrogram
    xt = 10**(xt/20)
    if np.any(np.isnan(xt)):
        print('AAAAAAAAAA')
        print(xt)

    # Derivation along the time axis, for each frequency bin
    GRADdt = diff(xt, n=1, axis=1)
    # Derivation of the previously derivated matrix along the frequency axis 
    GRADdf = diff(GRADdt, n=1, axis=0)
    # select the bandwidth
    if flim is not None :
        #idx_select = np.where((fn >= flim[0]) and (fn <= film[1]))
        #idx_select = np.where([flim[1] >= freq >= flim[0] for freq in fn])[0]
        #GRADdf_select = GRADdf[idx_select, :]
        GRADdf_select = GRADdf[index_bw(fn[0:-1],bw=flim),]
    else :
        GRADdf_select = GRADdf    
    # calcul of the tfsdt : sum of the pseudo-gradient in the frequency bandwidth
    # which is normalized by the total sum of the pseudo-gradient
    tfsd =  sum(abs(GRADdf_select))/sum(abs(GRADdf)) 

    return(tfsd)