# paperSoundscapeSourceClassification

## SETUP

First, install requirements.txt using the following command in a new Python 3.9.15. environment:

```
pip install -r requirements.txt
```

The experiment plan is developped with [doce](https://doce.readthedocs.io/en/latest/). 
No need to download doce as a doce folder is already provided in the repository.

Additionally, download the pretrained model (PANN ResNet38) by executing the following command:

```
python3 download/download_pretrained_models.py
```

## REPLICATION: CORRELATION WITH GRAFIC AND LORIENT1K

### DOWNLOAD DATASETS

Lorient-1k is already downloaded in the current repository in LORIENT1K_DATASET. It is available at https://zenodo.org/records/5153616, but the current LORIENT1K_DATASET folder
also contains the audios which are not available on the zenodo repository.
Grafic has not been published yet, please send me a message at modan.tailleur@gmail.com if you want to get this dataset.

### RUN EXPERIMENT


To replicate the correlation calculation, you can launch, in this order:

```
python3 main_doce.py -s groundtruth/ -c
python3 main_doce.py -s normalization/ -c
python3 main_doce.py -s reference/ -c
```

If you don't have access to GRAFIC yet, please launch

```
python3 main_doce.py -s groundtruth/dataset=Lorient1k -c
python3 main_doce.py -s normalization/dataset=Lorient1k -c
python3 main_doce.py -s reference/dataset=Lorient1k -c
```

Then you can launch the following code to generate the result tables. It will automatically add an std evaluation by generating noise in the humanly made annotations (gaussion noise with an std of 0.1), and 
generating like this 100 different correlation tables from which we can have a correlation std estimation:

```
python3 utils_plot/plot_result_table.py --dataset Lorient1k
python3 utils_plot/plot_result_table.py --dataset Grafic
```

Launch the following code to generate the figures in the paper's correlation tables:

```
python3 utils_plot/plot_relation_annotation_prediction.py
```

## REPLICATION: CENSE LORIENT FIGURES

### DOWNLOAD DATASETS

Use the following command to download the Cense Lorient dataset. This dataset being heavy (1.5T), make sure you download it on a device with free space.
```
python3 download/download_cense_lorient.py
```
Or in a custom "mydevice/myfolder" folder:
```
python3 download/download_cense_lorient.py --output "mydevice/myfolder/CenseData/"
```

### RUN EXPERIMENT

Check the multiplication factor to put in front of traffic, voices and birds predictions using the following code. Traffic should be 2.88, voices 1.80, and birds 2.65.
Add the argument --h5_path to add "mydevice/myfolder" as the path where the cense Lorient dataset is stored (e.g. python3 cense_pick_random_samples.py --desc winter2020 --h5_path "mydevice/myfolder"). 

```
python3 utils_plot/plot_tvb_mf_jasa2023.py
```

Launch the following code to generate the subsets for each figure:

```
python3 cense_pick_random_samples.py --desc winter2020
python3 cense_pick_random_samples.py --desc winter2020-3s
python3 cense_pick_random_samples.py --desc music_festival
python3 cense_pick_random_samples.py --desc no_music_festival
python3 cense_pick_random_samples.py --desc church_functional
python3 cense_pick_random_samples.py --desc church_not_functional
```

Then, run the following code to compute PANN-1/3oct on the subsets. The db_offset is set to -88, calculated on winter2020 dataset, in order 
to get dBFS from the cense sensors. Please check if this db_offset is close to -88 (printed in terminal) when running the code for winter2020 before going further (2 or 3 dB of difference is fine). 
```
python3 cense_compute_classifier.py --desc winter2020
python3 cense_compute_classifier.py --desc winter2020-3s
python3 cense_compute_classifier.py --desc music_festival
python3 cense_compute_classifier.py --desc no_music_festival
python3 cense_compute_classifier.py --desc church_functional
python3 cense_compute_classifier.py --desc church_not_functional
```

Run the following code to generate the figures in the './figures/' folder:

```
python3 cense_create_jasa_figures.py --desc winter2020
python3 cense_create_jasa_figures.py --desc winter2020-3s
python3 cense_create_jasa_figures.py --desc music
python3 cense_create_jasa_figures.py --desc church

```

## DIVERSE

To generate the spectrogram:

```
python3 utils_plot/plot_spectro.py
```

To generate the Top 10 of PANN predictions and PANN-1/3oct predictions on a given audio, put your audio "myaudio.wav" in the folder "audio" and run:
```
python3 prediction.py myaudio.wav
```

To generate sound from the generated Mel spectrogram:
```
python3 generate_audio.py myaudio.wav
```

