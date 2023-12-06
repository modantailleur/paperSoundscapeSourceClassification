import os 
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

from textwrap import wrap
import numpy as np
# import demo_h5 as demo uncomment to display the data computed using demo_h5.py
from scipy.stats.stats import pearsonr   
import texttable
import latextable
import pandas as pd
import xlsxwriter
import csv
from yamnet.yamnet_mel_inference import YamnetMelInference
from pann.pann_mel_inference import PannMelInference
from tfsd.tfsd_info import TFSDInfo
from scipy.stats import pearsonr
import piso
from ast import literal_eval
from itertools import groupby
import doce
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import utils.util as ut 

mels_type = "tfsd"

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

#same results than corr_coeff above
def corr2_coeff_modan(A, B):
    #A:3,74
    #B:527,74

    corr_mat = np.zeros((A.shape[0], B.shape[0]))
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(B)[0]):
            cor = pearsonr(A[i,:], B[j,:]).statistic
            print(cor)
            print(cor.shape)
            corr_mat[i,j] = np.mean(cor)

    # Finally get corr coeff
    return corr_mat

def merge_overintervals(ser):
    ser.sort(key=lambda x: x[0])
    return [next(i) for _, i in groupby(ser, key=lambda x: x[1])]

def interpol(arr, target):
    original_indices = np.arange(len(arr))
    interp_func = interp1d(original_indices, arr, kind='linear', fill_value='extrapolate')
    interpolated_indices = np.linspace(0, len(arr) - 1, target)
    interpolated_array = interp_func(interpolated_indices)
    return(interpolated_array)


def compute_groundtruth():
    df = pd.read_excel("./AUMILAB_DATASET/annotations_mt.xlsx")
    data = df[['t', 'v', 'b']].to_numpy()

    names = df['filename'].to_numpy()
    names = [name[:-4] for name in names]
    return(data, names)

def compute_metric(setting, detection_dir, data_dir, data, names, threshold=0.2, to_tvb=False):
    classifier = setting.classifier
    deep = setting.deep

    if (classifier in ["PANN","CNN-PINV-PANN"]) & (deep == "False"):
        classifier_evaluator = PannMelInference()
    if (classifier in ["YamNet","CNN-PINV-YamNet"]) & (deep == "False"):
        classifier_evaluator = YamnetMelInference()
    if (classifier == "TFSD") or (classifier == "felix") or (deep=="True"):
        classifier_evaluator = TFSDInfo()

    labels_str = classifier_evaluator.labels_str
    if (classifier in ["PANN", "YamNet", "CNN-PINV-PANN", "CNN-PINV-YamNet"]) & (deep=="False"):
        cat_str = np.array([classifier_evaluator.sub_classes_dict[label] for label in labels_str])

    scores = np.zeros((data.shape[0], 3))

    directory = detection_dir
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if (setting.identifier()[:-len(setting.step)] in file) & ("deep="+setting.deep in file):
                f = os.path.join(subdir, file)
                output = np.load(f)

                to_find = file[len(setting.identifier())+12:-4]
                mask = [name == to_find for name in names]

                ########
                if (classifier in ["PANN", "YamNet", "CNN-PINV-PANN", "CNN-PINV-YamNet"]) & (deep=="False"):
                    if (classifier in ["PANN", "CNN-PINV-PANN"]):
                        if to_tvb:
                            tvb = ut.batch_logit_to_tvb(output)[0]
                            t = tvb[0]
                            v = tvb[1]
                            b = tvb[2]
                        else:
                            temp = np.mean(output, axis=0)
                            t = temp[300]
                            v = temp[0]
                            b = temp[112]
                    else:
                        output[output>classifier_evaluator.threshold] = 1
                        output[output<=classifier_evaluator.threshold] = 0
                        
                        x_t = output[:, cat_str == 't']
                        x_v = output[:, cat_str == 'v']
                        x_b = output[:, cat_str == 'b']
                        
                        t = np.mean(np.max(x_t, axis=1))
                        v = np.mean(np.max(x_v, axis=1))
                        b = np.mean(np.max(x_b, axis=1))

                    scores[mask] = np.array([t,v,b])
                    
                else:
                    output = np.mean(output, axis=0)
                    scores[mask] = output

    if classifier == "TFSD":
        min_value = scores[:, 0].min()
        max_value = scores[:, 0].max()
        scores[:, 0] = (scores[:, 0] - min_value) / (max_value - min_value)

    correlation_table = corr2_coeff(data.T, scores.T)
    return(correlation_table)
