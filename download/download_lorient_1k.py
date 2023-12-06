
import subprocess

dir = './'
doi = '10.5281/zenodo.5153616'

output_dir = dir + 'LORIENT-1K_DATASET/'
#for reordering folders
source_folder = dir + 'URBAN-SOUND-8K/UrbanSound8K/'

#############
# GET ZIP FILES FROM ZENODO

command = ['zenodo_get', doi, '-o', output_dir]
subprocess.run(command, check=True)


    # ############
    # # EXTRACT ZIP FILES

    # # Get a list of all .tar.gz files in the directory
    # tar_files = glob.glob(output_dir + '/*.tar.gz')

    # # Extract each .tar.gz file
    # for tar_file in tar_files:
    #     with tarfile.open(tar_file, 'r:gz') as tar:
    #         tar.extractall(output_dir)

    # # Move the child "doce" folder to the same level as the parent folder and rename it to "doce_temp"
    # shutil.move(source_folder, "temp")

    # # Delete the empty source folder
    # shutil.rmtree(output_dir)

    # # Rename the "doce_temp" folder to "doce"
    # os.rename("temp", output_dir)

    # ####################
    # #REMOVE EVERY "FOLD" FOLDER (every information in on the metadata/UrbanSound8k.csv file)

    # # get the path of the audio subfolder
    # audio_path = output_dir + "audio/"

    # # loop through all directories in audio and move their contents to audio root
    # for root, dirs, files in os.walk(audio_path):
    #     for file in files:
    #         # get the full path of the file
    #         file_path = os.path.join(root, file)
    #         # move the file to the audio root
    #         if not file.startswith('.'):
    #             shutil.move(file_path, audio_path)
            
    # # loop through all directories in audio and remove them
    # for root, dirs, files in os.walk(audio_path):
    #     for dir in dirs:
    #         # get the full path of the directory
    #         dir_path = os.path.join(root, dir)
    #         # remove the directory and all its contents
    #         shutil.rmtree(dir_path)

    # ##########################
    # # RECALCULATE DURATIONS (durations are badly calculated in the original csv file)

    # # Set file paths
    # csv_path = output_dir + 'metadata/UrbanSound8K.csv'
    # audio_dir = output_dir + 'audio'

    # # Load CSV into Pandas DataFrame
    # df = pd.read_csv(csv_path)

    # # Add column for duration
    # df['duration'] = 0

    # n_file = 0
    # len_df = len(df)
    # # Loop through each row of the DataFrame
    # for index, row in df.iterrows():
    #     # Get the file path for the audio file
    #     audio_path = os.path.join(audio_dir, row['slice_file_name'])
        
    #     # Load the audio file with librosa
    #     audio, sr = librosa.load(audio_path, sr=None)
        
    #     # Calculate the duration in seconds
    #     duration = librosa.get_duration(y=audio, sr=sr)
        
    #     # Update the duration column in the DataFrame
    #     df.at[index, 'duration'] = duration

    #     n_file += 1
    #     print('\r' + f'{n_file} / {len_df} files have been processed in dataset',end=' ')

    # # save the csv with recalculated duration column
    # new_csv_path = output_dir + 'metadata/UrbanSound8K_recalculated.csv'
    # df.to_csv(new_csv_path, index=False)

    # print("Extraction complete.")

