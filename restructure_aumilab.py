from textwrap import wrap
import numpy as np
import pandas as pd
import os 
import librosa
from scipy.io.wavfile import write

def reshape_metadata(file_ptp="./AUMILAB/aumilab_test.xlsx", audio_dir="./AUMILAB/audios-reshape/", output_dir="./AUMILAB/aumilab_test_reshape.xlsx", audio_len=10):

    df = pd.read_excel(file_ptp)

    df['filename'] = [f [:-4] for f in df['filename'].to_list()]

    df_output = pd.DataFrame()
    cpt=0
    for subdir, dirs, files in os.walk(audio_dir):
        for file in files:
            f = os.path.join(subdir, file)
            #create new dataframe selecting only rows corresponding to the targeted filename
            df_file = df.loc[df['filename'] == file[:-6]].copy()
            file_idx = int(file[-5:-4])
            time_min = (file_idx-1)*audio_len
            time_max = file_idx*audio_len

            df_file['start'] = [time_min if ((start<time_min)&(end>time_min)) else start for (start, end) in df_file[['start', 'end']].values]
            df_file['end'] = [time_max if ((start<time_max)&(end>time_max)) else end for (start, end) in df_file[['start', 'end']].values]
            df_file['filename'] = [filename+'-'+str(file_idx)+'.wav' for filename in df_file['filename'].to_list()]

            df_file['end'] = [end-time_min for end in df_file['end'].to_list()]
            df_file['start'] = [start-time_min for start in df_file['start'].to_list()]

            df_file['end'] = [end if 0 <= end <= audio_len else 0 for end in df_file['end'].to_list()]
            df_file['start'] = [start if audio_len >= start >= 0 else 0 for start in df_file['start'].to_list()]

            if len(df_output) != 0:
                data = [df_output, df_file]
                df_output = pd.concat(data)
            else:
                df_output = df_file
            
            cpt+=1
            print(cpt)

    df_output['duration'] = 10
    writer = pd.ExcelWriter(output_dir, engine='xlsxwriter')
    #df_cat = pd.read_excel(file_ptp, sheet_name="aumilab_categories")
    df_output.to_excel(writer, sindex=False)
    #df_cat.to_excel(writer, sheet_name="aumilab_categories", index=False)
    writer.save()

def reshape_audio_database(output_dir="./AUMILAB/audios-reshape/", input_dir="./AUMILAB/audios/", audio_len=10):
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            print(file)
            f = os.path.join(subdir, file)

            audio, sr = librosa.load(f, sr=32000)
            length_sec = len(audio)/sr
            print(length_sec)

            if length_sec >= audio_len:
                chunk_size = int(audio_len*sr)
                n_chunks = int(length_sec/audio_len)
                all_samples = np.array_split(audio, n_chunks)
                for idx, sample in enumerate(all_samples):
                    if len(sample) == chunk_size:
                        str_to_write = file[:-4]+"-"+str(idx+1)+".wav"
                        if not str_to_write in ["62f7caf5ff16b865defa0e1c-6.wav", "62e400157a2acbde063ed41d-3.wav", "62e215817a2acbde063ec756-3.wav", "62e215817a2acbde063ec756-2.wav"]:
                            write(output_dir+file[:-4]+"-"+str(idx+1)+".wav", sr, sample)
                        else:
                            print(f("file {str_to_write} is empty and is'nt informative, skipping calculations"))


if __name__ == '__main__':
    reshape_metadata()

