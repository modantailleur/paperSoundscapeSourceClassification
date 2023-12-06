import sys
import numpy as np
import soundfile as sf
import os
import maad
import librosa

#############################
#############################
###########################
#ACTIVATE ONLY IF R PACKAGES ARE NEEDED

# import rpy2
# import rpy2.robjects.packages as rpackages
# from rpy2.robjects.packages import importr
# from rpy2.robjects.vectors import StrVector
# import rpy2.robjects.numpy2ri
# rpy2.robjects.numpy2ri.activate()
# from rpy2 import robjects as ro
# import wave

# # Selectively install what needs to be install.
# utils = rpackages.importr('utils')
# utils.chooseCRANmirror(ind=1)

# r = rpy2.robjects.r
# r['source']('/home/user/Documents/Thèse/Code/R/indices_modan.R')
# paIndices = rpy2.robjects.globalenv['indices']

# # # Install packages
# # packnames = ('seewave')
# # utils.install_packages(StrVector(packnames))

# swTFSD = importr('seewave').TFSD
# swWav2Flac = importr('seewave').wav2flac
# swreadWave = importr('tuneR').readWave

# importr('pracma')
# importr('soundecology')

#############################
#############################
#############################

class TFSDInference():
    def __init__(self, db_compensation=0):
        self.labels_str = ['L50', 'TFSD5001s', 'TFSD4000125ms']
        self.n_labels = 3
        self.db_compensation = db_compensation
        self.db_compensation_multiplier = 10**(db_compensation/10)

    def inference_from_scratch(self, file_name):
        s, fs = librosa.load(file_name, sr=32000)
        s = s*self.db_compensation_multiplier

        #traffic
        rmse = librosa.feature.rms(y=s, frame_length=fs//8, hop_length=512)
        rmse[rmse <= 0] = 10e-15
        score_traffic = np.median(np.log10(rmse))

        # #voices
        # Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs, nperseg=fs, flims=[31.5, 16000], window='hann')
        # Sxx_power[Sxx_power <= 0] = 10e-15
        # Sxx_power = np.log10(Sxx_power)
        # score_voices = maad.features.tfsd(Sxx_power,fn, tn, flim=(500,1500)) 

        # #birds
        # Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs, nperseg=fs//8, flims=[31.5, 16000], window='hann')
        # Sxx_power[Sxx_power == 0] = 10e-15
        # Sxx_power = np.log10(Sxx_power)
        # score_birds = maad.features.tfsd(Sxx_power,fn, tn, flim=(1500,6000)) 

        #voices
        Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs, nperseg=fs, flims=[31.5, 16000], window='hann', mode='psd')
        Sxx_power[Sxx_power <= 0] = 10e-15
        Sxx_power = 10*np.log10(Sxx_power)
        score_voices = maad.features.tfsd(Sxx_power,fn, tn, log=False, flim=(500,1500)) 

        #birds
        Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs, nperseg=fs//8, flims=[31.5, 16000], window='hann', mode='psd')
        Sxx_power[Sxx_power == 0] = 10e-15
        Sxx_power = 10*np.log10(Sxx_power)
        score_birds = maad.features.tfsd(Sxx_power,fn, tn, log=False, flim=(1500,6000)) 

        #all
        scores = np.array([score_traffic, score_voices, score_birds])
        scores = np.expand_dims(scores, axis=0)

        return(scores)     

def detection(file_name):
    s, fs = librosa.load(file_name, sr=32000)

    #traffic
    rmse = librosa.feature.rms(y=s, frame_length=fs//8, hop_length=512)
    rmse[rmse <= 0] = 10e-15
    score_traffic = np.median(np.log10(rmse))

    # #voices
    # Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs, nperseg=fs, flims=[31.5, 16000], window='hann')
    # Sxx_power[Sxx_power <= 0] = 10e-15
    # Sxx_power = np.log10(Sxx_power)
    # score_voices = maad.features.tfsd(Sxx_power,fn, tn, flim=(500,1500)) 

    # #birds
    # Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs, nperseg=fs//8, flims=[31.5, 16000], window='hann')
    # Sxx_power[Sxx_power == 0] = 10e-15
    # Sxx_power = np.log10(Sxx_power)
    # score_birds = maad.features.tfsd(Sxx_power,fn, tn, flim=(1500,6000)) 

    #voices
    Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs, nperseg=fs, flims=[31.5, 16000], window='hann', mode='psd')
    Sxx_power[Sxx_power <= 0] = 10e-15
    Sxx_power = 10*np.log10(Sxx_power)
    score_voices = maad.features.tfsd(Sxx_power,fn, tn, flim=(500,1500)) 

    #birds
    Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs, nperseg=fs//8, flims=[31.5, 16000], window='hann', mode='psd')
    Sxx_power[Sxx_power == 0] = 10e-15
    Sxx_power = 10*np.log10(Sxx_power)
    score_birds = maad.features.tfsd(Sxx_power,fn, tn, flim=(1500,6000)) 

    #all
    scores = np.array([score_traffic, score_voices, score_birds])
    scores = np.expand_dims(scores, axis=0)

    #print(scores)
    return(scores)

if __name__ == '__main__':
    # GRAFIC
    # directory = "/home/user/Documents/Thèse/Code/3-CorrelationPANNYamNet-Pleasantness/GRAFIC_DATASET/Enregistrements_Mobiles_Paris_4x19pts"
    # save_beg =  "/home/user/Documents/Thèse/Code/3-CorrelationPANNYamNet-Pleasantness/tfsd_outputs/GRAFIC/"
    # n_to_delete = -4

    # SINGA:PURA
    directory = "/home/user/Documents/Thèse/Code/3-CorrelationPANNYamNet-Pleasantness/SINGAPURA_DATASET/labelled/"
    save_beg =  "/home/user/Documents/Thèse/Code/3-CorrelationPANNYamNet-Pleasantness/tfsd_outputs/SINGAPOUR/"
    n_to_delete = -4

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file == "[b827eb3e52b8][2020-08-20T08-07-05Z][manual][---][4edbade2d41d5f80e324ee4f10d401c0][]-146.wav":
                f = os.path.join(subdir, file)
                s, fs = librosa.load(f, sr=32000)

                print('XXXXXXXXXXXXXXXX')
                print(file)
                print(fs)
                # for time of presence of traffic (to do with dB instead of bites)
                rmse = librosa.feature.rms(y=s, frame_length=fs//8, hop_length=512)
                score_traffic = np.median(np.log10(rmse))
                
                ###########################
                #SCIKIT MAAD IMPLEMENTATION
                # for time of presence of voices
                #in paper flims=[31.5, 16000]. In PA code, flims=[50, 10000]
                Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs, nperseg=fs, flims=[31.5, 16000], window='hann')
                Sxx_power = np.log10(Sxx_power)
                score_voices = maad.features.tfsd(Sxx_power,fn, tn, flim=(400,1000)) 

                # for time of presence of birds
                Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs, nperseg=fs//8, flims=[31.5, 16000], window='hann')
                Sxx_power = np.log10(Sxx_power)
                score_birds = maad.features.tfsd(Sxx_power,fn, tn, flim=(1500,5000)) 

                #s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
                # Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs)
                # Sxx_power = np.log10(Sxx_power)
                # score_voices = maad.features.tfsd(Sxx_power,fn, tn, flim=(400,1000)) 
                # #Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs)
                # score_birds = maad.features.tfsd(Sxx_power,fn, tn, flim=(2000,8000), display=False)
                ###########################
                #SEEWAVE 

                # if f[-5:] == '.flac':
                #     print('ok')
                #     swWav2Flac(f, reverse = FALSE, overwrite = TRUE)
                #     f=f[:-5]+'.wav'

                # audio = swreadWave(f)
                # score_voices = swTFSD(wave=audio, channel=1, ovlp=0, flim=np.array([0.4, 1]))
                # score_voices = score_voices[0]
                # # score_birds = swTFSD(wave=audio, channel=1, ovlp=0, flim=np.array([2, 8]))
                # score_birds = swTFSD(wave=audio, channel=1, ovlp=0, flim=np.array([1.5, 5]))
                # score_birds = score_birds[0]

                ##########################
                #PIERRE AUMOND
                # indices = paIndices(f, 0)
                # score_traffic = indices[1]
                # score_voices = indices[2]
                # score_birds = indices[3]

                #seewave: 
                #wn: hamming
                #third octave bands: toctave=c(50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000)
                #wn: automatically 125ms
                scores = np.array([score_traffic, score_voices, score_birds])
                scores = np.expand_dims(scores, axis=0)

                
                np.save(save_beg+"scores_"+file[:n_to_delete], scores)
                print(scores)
                print(scores.shape)
