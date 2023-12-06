#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:12:06 2022

@author: user
"""

import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from pathlib import Path
from sklearn import preprocessing
from yamnet.torch_audioset.yamnet.model import yamnet_category_metadata
import torch.nn.functional as F
from pathlib import Path

class TFSDInfo():
    def __init__(self, device=torch.device("cpu")):
        self.labels_str = ['L50', 'TFSD5001s', 'TFSD4000125ms']
        self.n_labels = 3
        
        




