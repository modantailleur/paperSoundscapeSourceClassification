
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:02:54 2022

@author: user
"""
import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

import numpy as np
import librosa
import numpy as np
import librosa
import torch.utils.data
import torch
from transcoder.transcoders import ThirdOctaveToMelTranscoderPinv, ThirdOctaveToMelTranscoder
from utils.util import sort_labels_by_score
from utils.util import plot_multi_spectro
import utils.bands_transform as bt

if __name__ == '__main__':
    MODEL_PATH = "./reference_models"
    filename = "street_pedestrian-paris-152-4607-a.wav"

    model_name_bce = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
    model_name_bce_slow = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
    model_name_mse = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+step=train+transcoder=cnn_pinv+ts=0_model'

    model_type = 'cnn_pinv'
    dtype=torch.FloatTensor
    spec = True
    fs=32000
    full_filename = "audio/" + filename
    force_cpu = False
    #manage gpu
    useCuda = torch.cuda.is_available() and not force_cpu

    if useCuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
        #MT: add
        device = torch.device("cuda:0")
    else:
        print('No CUDA available.')
        dtype = torch.FloatTensor
        ltype = torch.LongTensor
        #MT: add
        device = torch.device("cpu")

    x_32k = librosa.load(full_filename, sr=fs)
    x_32k = np.array(x_32k[0])
    x_32k = librosa.util.normalize(x_32k)
    x_32k = x_32k[0:32000]

    transcoder_deep_bce = ThirdOctaveToMelTranscoder(model_type, model_name_bce, MODEL_PATH, device)
    transcoder_pinv = ThirdOctaveToMelTranscoderPinv(MODEL_PATH, model_name_bce, device)

    x_mels_deep_bce = transcoder_deep_bce.wave_to_thirdo_to_mels(x_32k)
    x_logit_deep_bce = transcoder_deep_bce.mels_to_logit(x_mels_deep_bce, mean=True)
    x_logit_deep_bce = x_logit_deep_bce.T

    x_mels_pinv = transcoder_pinv.wave_to_thirdo_to_mels(x_32k)
    x_mels_gt, x_logit_gt = transcoder_deep_bce.wave_to_mels_to_logit(x_32k)

    print('\n XXXXXXXXX ORIGINAL PANN CLASSIFIER (MEL INPUT) XXXXXXXXXXX')
    labels = sort_labels_by_score(np.mean(x_logit_gt, axis=0), transcoder_deep_bce.classif_inference.labels_str)[1][:10]
    scores = sort_labels_by_score(np.mean(x_logit_gt, axis=0), transcoder_deep_bce.classif_inference.labels_str)[0][:10]
    for k in range(len(labels)):
        print(f'{labels[k]} : {round(float(scores[k]), 2)}')

    print('\n XXXXXXXXXXXX TRANSCODED PANN CLASSIFIER (THIRD-OCTAVE INPUT, TEACHER-STUDENT APPROACH) XXXXXXXXXXXX')
    labels = sort_labels_by_score(np.mean(x_logit_deep_bce, axis=0), transcoder_deep_bce.classif_inference.labels_str)[1][:10]
    scores = sort_labels_by_score(np.mean(x_logit_deep_bce, axis=0), transcoder_deep_bce.classif_inference.labels_str)[0][:10]
    for k in range(len(labels)):
        print(f'{labels[k]} : {round(float(scores[k]), 2)}')

    tho_tr = bt.ThirdOctaveTransform(sr=32000, flen=4096, hlen=4000)
    thirdo = tho_tr.wave_to_third_octave(x_32k)
    # Calculate the size ratio for rows and columns
    thirdo = thirdo -4

    row_ratio = 80/29
    col_ratio = 103/8

    plot_multi_spectro([thirdo, x_mels_pinv, x_mels_deep_bce, x_mels_gt], 
             32000, 
            #  title=['Fast third-octaves', 'Mels (PINV)', 'Mels (transcoder)', 'Mels (original)'], 
             title=['Coarse', 'Fine (PINV transcoder)', 'Fine (CNN transcoder)', 'Fine (original)'], 
             vmin=-40, 
             vmax=5, 
             ylabel=['Third octave bin', 'Mel bin', 'Mel bin', 'Mel bin'])
