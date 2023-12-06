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
from scipy.stats import pearsonr, spearmanr
import utils.util as ut

mels_type = "yamnet"

def weighted_top_10(arr, idx_t=300, idx_v=0, idx_b=111):
    sorted_indices = np.argsort(arr)[::-1]
    sorted_array = arr[sorted_indices]

    sorted_t = np.where(sorted_indices == idx_t)[0][0]
    sorted_v = np.where(sorted_indices == idx_v)[0][0]
    sorted_b = np.where(sorted_indices == idx_b)[0][0]

    t = 1 - sorted_t/20 if sorted_t < 20 else 0
    v = 1 - sorted_v/20 if sorted_v < 20 else 0
    b = 1 - sorted_b/20 if sorted_b < 20 else 0

    return(t,v,b)

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

def corr2_coeff_spearman(A, B):
    corr_mat = np.zeros((A.shape[0], B.shape[0]))
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(B)[0]):
            cor = spearmanr(A[i,:], B[j,:]).statistic
            corr_mat[i,j] = np.mean(cor)

    # Finally get corr coeff
    return corr_mat

#same results than corr_coeff above
def corr2_coeff_pearson(A, B):
    #A:3,74
    #B:527,74

    corr_mat = np.zeros((A.shape[0], B.shape[0]))
    p_value_mat = np.zeros((A.shape[0], B.shape[0]))

    for i in range(np.shape(A)[0]):
        for j in range(np.shape(B)[0]):
            cor = pearsonr(A[i,:], B[j,:]).statistic
            p_value = pearsonr(A[i,:], B[j,:]).pvalue
            corr_mat[i,j] = np.mean(cor)
            p_value_mat[i,j] = np.mean(p_value)

    # Finally get corr coeff
    return corr_mat, p_value_mat

def compute_groundtruth(setting, detection_dir, data_dir, threshold=0.2):

    file_ptp = "./GRAFIC_DATASET/FinalNotesWithWav.csv"
    arr = np.genfromtxt(file_ptp, delimiter=",", dtype=str)

    col_to_keep = [9,10,12,-1]
    data = arr[1:, :]

    bool_cond = (data[:, col_to_keep] != '"NA"') & (data[:, col_to_keep] != 'NA')

    #a bit weird because of the two inversions, but checks if there is a false in any column and return False if that's the case
    bool_cond = ~np.invert(bool_cond).any(axis=1)
    data = data[bool_cond]
    data = data[:,col_to_keep]

    names = [string[0:-4] for string in data[:,-1]]
    data = data[:, :-1]
    #divide by 10 because the scale is 0-10, and BCE takes only values between 0 and 1
    data = data.astype(float)/10
    return(data, names)

def compute_metric(setting, detection_dir, data_dir, data, names, to_tvb=False):

    classifier = setting.classifier
    deep = setting.deep

    if (classifier in ["PANN",  "CNN-PINV-PANN"]) & (deep == "False"):
        classifier_evaluator = PannMelInference()
    if (classifier in ["YamNet",  "CNN-PINV-YamNet"]) & (deep == "False"):
        classifier_evaluator = YamnetMelInference()
    if (classifier == "TFSD") or (classifier == "felix") or (deep=="True"):
        classifier_evaluator = TFSDInfo()

    labels_str = classifier_evaluator.labels_str
    if (classifier in ["YamNet", "PANN",  "CNN-PINV-PANN",  "CNN-PINV-YamNet"]) & (deep == "False"):
        cat_str = np.array([classifier_evaluator.sub_classes_dict[label] for label in labels_str])
    scores = np.zeros((data.shape[0], 3))

    directory = detection_dir
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if (setting.identifier()[:-len(setting.step)] in file) & ("deep="+setting.deep in file):
                f = os.path.join(subdir, file)
                output = np.load(f)

                #12 because of the word "prediction" and the "_" before and after, -4 because of ".wav"
                to_find = file[len(setting.identifier())+12:-4]
                mask = [name == to_find for name in names]            

                if (classifier in ["PANN", "YamNet",  "CNN-PINV-PANN",  "CNN-PINV-YamNet"]) & (deep=="False"):
                    if (classifier in ["PANN", "CNN-PINV-PANN"]):
                        if to_tvb:
                            # tvb = ut.batch_logit_to_tvb_top(output)[0]
                            tvb = ut.batch_logit_to_tvb(output)[0]
                            t = tvb[0]
                            v = tvb[1]
                            b = tvb[2]
                        else:
                            temp = np.mean(output, axis=0)
                            #t: 300 --> vehicle
                            #v: 0 --> speech
                            #b: 514 --> Environmental Noise
                            #b: 112 --> bird vocalization
                            t = temp[327]
                            v = temp[0]
                            b = temp[112]
                            # t,v,b = weighted_top_10(temp)
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

    # print(data.shape)
    # print(scores.shape)
    # scores[:, 2] = scores[:, 2]*5
    
    if classifier == "TFSD":
        min_value = scores[:, 0].min()
        max_value = scores[:, 0].max()
        scores[:, 0] = (scores[:, 0] - min_value) / (max_value - min_value)

    scores = scores.clip(min=0, max=1)

    correlation_table, p_value = corr2_coeff_pearson(data.T, scores.T)
    correlation_table_spearman = corr2_coeff_spearman(data.T, scores.T)

    # print('P VALUE')
    # print(p_value)

    # print('SPEARMAN')
    # print(correlation_table_spearman)
    
    return(correlation_table)
