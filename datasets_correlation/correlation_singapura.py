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
import librosa 
import doce
from sklearn import preprocessing

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

def compute_groundtruth(setting, experiment, detection_dir, data_dir, threshold=0.2):
    n_to_delete = -4
    file_ptp = "./SINGAPURA_DATASET/labels_public.xlsx"

    # beginning of each t,v,b event. For example, if event_t = '1-', each event
    # that begins with '1-' (for example '1-2') will be considered as a 't' event
    event_t = '1-'
    event_v = '7-'
    event_b = '9-2'
    df = pd.read_excel(file_ptp)

    col_to_keep = [0,1,3,4]

    data = []
    names = []

    #create new column for the interval of presence
    df["interval_presence"] = df[['onset', 'offset']].values.tolist()
    df['interval_presence'] = df['interval_presence'].apply(lambda x: pd.Interval(*x))

    #create new column aggregating events in 3 types: t,v, or b. The way those 3 events are aggregated 
    # are define by event_t, event_v and event_b
    df['event_tvb'] = 'o'
    df.loc[(df['event_label'].str.startswith(event_t)), 'event_tvb'] = 't'
    df.loc[(df['event_label'].str.startswith(event_v)), 'event_tvb'] = 'v'
    df.loc[(df['event_label'].str.startswith(event_b)), 'event_tvb'] = 'b'

    directory = data_dir
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            
            setting_str_level = doce.Setting(experiment.level, [setting.dataset], positional=False).identifier()
            level = np.load(experiment.path.level+setting_str_level+'_level_'+ file[:n_to_delete] + '.npy')
            sr = np.load(experiment.path.level+setting_str_level+'_sr.npy')
            level = np.load(experiment.path.level+setting_str_level+'_level_'+ file[:n_to_delete] + '.npy')
            #level = np.log10(level*10)
            level_len = len(level)
            audio_len = 10
            normalization = 10*level_len/audio_len
            #normalization = 10 # if no weighting by level in dB

            #create new dataframe selecting only rows corresponding to the targeted filename
            df_file = df.loc[df['filename'] == file[:-4]+'.flac'].copy()
            if len(df_file) == 0:
                print('NO FILE FOUND FOR ' + file )

            #select only t,v,b events, which are in the proximity that is set
            df_file = df_file.loc[(df_file['event_tvb'] != 'o') & (df_file['proximity'].isin(['near', 'moving', 'far']))]

            if len(df_file) == 0:
                names.append(file[:-4])
                data.append([0., 0., 0.])
            else:
                
                # group interval of presence by annotator and event using piso to unionize intervals. It unionizes every interval for
                # each annotation of a class for an annotator. For example if a 't' is annotated in interval [0,5] and another
                # 't' is annotated by the same annotator in interval [2,7], the result will be [0,7] 
                df_file = df_file.groupby(['annotator', 'event_tvb'])['interval_presence'] \
                    .apply(pd.arrays.IntervalArray) \
                    .apply(piso.union) \
                    .reset_index()
                
                #in case of weighted presence
                weighted_presence = [np.sum([np.sum(level[int(interv[0]*level_len/audio_len):int(interv[1]*level_len/audio_len)]) for interv in item.to_tuples()]) for item in df_file['interval_presence']]
                df_file["weighted_total_time_of_presence"] = weighted_presence
                
                #in case of none weighted presence
                #creates a new column for the total time of presence by summing the times represented in the interval
                df_file["total_time_of_presence"] = [np.sum([interv[1]-interv[0] for interv in item.to_tuples()]) for item in df_file['interval_presence']]

                #takes the mean of every annotator for each t,v,b event
                df_file = df_file.groupby(['event_tvb'])['weighted_total_time_of_presence'].mean()
                
                #t,v,b are the ratio of presence of each class t,v,b (between 0 and 1) in the whole 10s file (reason why '/10')
                try:
                    t = df_file['t']/normalization
                except KeyError:
                    t = float('NaN')
                
                try:
                    v = df_file['v']/normalization
                except KeyError:
                    v = float('NaN')
                
                try:
                    b = df_file['b']/normalization
                except KeyError:
                    b = float('NaN')
                
                names.append(file[:-4])
                data.append(np.array([t,v,b]))

    data = np.array(data)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    data[np.isnan(data)] = 0
    names = np.array(names)

    return(data, names)

def compute_metric(setting, detection_dir, data_dir, data, names):
    classifier = setting.classifier
    deep = setting.deep

    if (classifier in ["PANN", "CNN-PINV-PANN"]) & (deep == "False"):
        classifier_evaluator = PannMelInference()
    if (classifier in ["YamNet", "CNN-PINV-YamNet"]) & (deep == "False"):
        classifier_evaluator = YamnetMelInference()
    if (classifier == "TFSD") or (classifier == "felix") or (deep=="True"):
        classifier_evaluator = TFSDInfo()

    labels_str = classifier_evaluator.labels_str
    if (classifier in ["PANN", "YamNet",  "CNN-PINV-PANN",  "CNN-PINV-YamNet"]) & (deep=="False"):
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
                if (classifier in ["PANN", "YamNet",  "CNN-PINV-PANN",  "CNN-PINV-YamNet"]) & (deep=="False"):
                    if (classifier in ["PANN", "CNN-PINV-PANN"]):
                        temp = np.mean(output, axis=0)
                        t = temp[300]
                        v = temp[0]
                        b = temp[111]
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

    print('AAAAAAAAAAAAAAA')
    print(np.mean(scores))
    print(scores)
    print(data)

    if classifier == "TFSD":
        min_value = scores[:, 0].min()
        max_value = scores[:, 0].max()
        scores[:, 0] = (scores[:, 0] - min_value) / (max_value - min_value)

    correlation_table = corr2_coeff(data.T, scores.T)

    return(correlation_table)
    
