import librosa
import os
from scipy.io.wavfile import write

def reshape_audio_database(output_dir="./SINGAPURA_DATASET/labelled-reshape/", input_dir="./SINGAPURA_DATASET/labelled/"):
    cpt=0
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            f = os.path.join(subdir, file)
            #take mono from 6.1 input (mean on every channel)
            audio, sr = librosa.load(f, sr=32000, mono=True)
            cpt+=1
            print(cpt)
            write(output_dir+file[:-5]+".wav", sr, audio)

if __name__ == '__main__':
    reshape_audio_database()