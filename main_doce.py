import numpy as np
import doce
from pathlib import Path
import time
import torch
from transcoder.transcoder_inference_dataset import YamNetInference as ysed
from transcoder.transcoder_inference_dataset import PANNInference as psed
from transcoder.transcoder_inference_dataset import TrYamNetInference as tysed
from transcoder.transcoder_inference_dataset import TrPANNInference as tpsed
from transcoder.transcoder_inference_dataset import TrPANNInferenceSlow as tpsed_slow
from tfsd.tfsd_inference_dataset import TFSDInference as tsed
from censeModels.felix_inference_dataset import FelixInference as fsed
from censeModels.thirdoctave_inference_dataset import ThirdOctaveInference as osed
import os
import datasets_correlation.correlation_grafic as cg
import datasets_correlation.correlation_singapura as cs
import datasets_correlation.correlation_aumilab as ca
import datasets_correlation.correlation_aumilab_mt as camt
import datasets_correlation.correlation_lorient1k as cl
from trainers import DLModelsTrainer
from models import FC
import pickle 
import copy
import librosa
from torchaudio.transforms import MelSpectrogram

torch.manual_seed(0)

# define the experiment
experiment = doce.Experiment(
  name = 'paper_soundscape_evaluation',
  purpose = 'calculates the correlation in time of presence of tvb for the outputs of PANN, CNN-train-synth and other models',
  author = 'Modan Tailleur',
  address = 'modan.tailleur@ls2n.fr',
)

########## ACCESS PATH ##################

#general
exp_path = './doce_outputs/'

#to ssd
#exp_path = '/media/user/MT-SSD-NEW/0-PROJETS_INFO/Thèse/correlations_pann_yamnet_outputs/doce_outputs/'

#########################################


experiment.set_path('output',  exp_path+experiment.name+'/', force=True)
experiment.set_path('detection', exp_path+experiment.name+'/detection/', force=True)
experiment.set_path('detection_mean', exp_path+experiment.name+'/detection_mean/', force=True)
experiment.set_path('correlation', exp_path+experiment.name+'/correlation/', force=True)
experiment.set_path('time', exp_path+experiment.name+'/time/', force=True)
experiment.set_path('groundtruth', exp_path+experiment.name+'/groudtruth/', force=True)
experiment.set_path('model', exp_path+experiment.name+'/model/', force=True)
experiment.set_path('loss', exp_path+experiment.name+'/loss/', force=True)
experiment.set_path('level', exp_path+experiment.name+'/level/', force=True)
experiment.set_path('normalization', exp_path+experiment.name+'/normalization/', force=True)

experiment.add_plan('normalization',
# more datasets are available if necessary (Aumilab and Singapura are datasets with onsets and offsets, not suitable for time of presence)                
#   dataset = ['Grafic', 'Lorient1k', 'SingaPura', 'Aumilab'],
  dataset = ['Grafic', 'Lorient1k'],
)

# the level plan is only used for SingaPura and Aumilab (weightened annotations depending on the level)
experiment.add_plan('level',
  dataset = ['SingaPura', 'Aumilab'],
)

experiment.add_plan('groundtruth',
# more datasets are available if necessary (Aumilab and Singapura are datasets with onsets and offsets, not suitable for time of presence)                
#   dataset = ['Grafic', 'SingaPura', 'Aumilab', 'Lorient1k', 'Aumilab-MT'],
  dataset = ['Grafic', 'Lorient1k'],
)

experiment.add_plan('reference',
  deep = ['False'],
  step = ['compute', 'metric'],
# more classifiers are available if necessary
#   classifier = ['thirdoctave', 'felix', 'TFSD', 'YamNet', 'PANN', 'CNN-PINV-YamNet', 'CNN-PINV-PANN', 'CNN-PINV-PANN-Slow'],
  classifier = ['felix', 'PANN', 'CNN-PINV-PANN'],
# more datasets are available if necessary (Aumilab and Singapura are datasets with onsets and offsets, not suitable for time of presence)
#   dataset = ['Grafic', 'SingaPura', 'Aumilab', 'Lorient1k', 'Aumilab-MT'],
  dataset = ['Grafic', 'Lorient1k'],
)

# The deep plan is not part of the paper replication. It just trains an MLP to go from 527 classes (for PANN) to 3 traffic voices and birds classes
# adapted to SingaPura and Aumilab outputs. It lead to disappointing results, as explained in the paper, and was thus discarded from the paper results.
experiment.add_plan('deep',
  deep = ['True'],
  step = ['train', 'compute', 'metric'],
  classifier = ['thirdoctave', 'TFSD','YamNet','PANN', 'CNN-PINV-YamNet', 'CNN-PINV-PANN', 'CNN-PINV-PANN-slow'],
  dataset = ['Grafic', 'SingaPura', 'Aumilab', 'Lorient1k'],
)

#traffic correlation
experiment.set_metric(
  name = 'corr_t',
  path = 'correlation',
#   significance = True,
  percent=True,
#   higher_the_better=True,
  precision=2
  )

#voices correlation
experiment.set_metric(
  name = 'corr_v',
  path = 'correlation',
#   significance = True,
  percent=True,
#   higher_the_better=True,
  precision=2
  )

#birds correlation
experiment.set_metric(
  name = 'corr_b',
  path = 'correlation',
#   significance = True,
  percent=True,
#   higher_the_better=True,
  precision=2
  )

#global correlation
experiment.set_metric(
  name = 'corr_global',
  path = 'correlation',
#   significance = True,
  percent=True,
#   higher_the_better=True,
  precision=2
  )

def step(setting, experiment):
    
    print('XXXXXXXX ONGOING EXPERIMENT XXXXXXXX')
    print(setting.identifier())

    # in case we want a prediction every 10ms
    # pann_type = 'CNN14'
    # mean = False
    # normalize = True

    # in case we want a global prediction every 10s, averaged on the entirety of the file (mean=True)
    pann_type = 'ResNet38'
    mean = True
    normalize = False

    n_to_delete = -4
    plan_name = experiment.get_current_plan().get_name()
    #WARNING: comment following line if you do not want to keep existing data
    keep_existing = False
    start_time = time.time()

    # choose the correct audio directory data_dir 
    if setting.dataset == 'SingaPura':
        data_dir = "./SINGAPURA_DATASET/labelled-reshape/"
        constant_10s_audio = True

    if setting.dataset == 'Grafic':
        data_dir = './GRAFIC_DATASET/Enregistrements_Mobiles_Paris_4x19pts/'
        constant_10s_audio = False

    if setting.dataset == 'Aumilab':
        data_dir = './AUMILAB_DATASET/audios-reshape/'
        constant_10s_audio = True

    if setting.dataset == 'Lorient1k':
        data_dir = './LORIENT1K_DATASET/audio/'
        constant_10s_audio = False

    if setting.dataset == 'Aumilab-MT':
        data_dir = './AUMILAB_DATASET/audios-mt-annotated/'
        constant_10s_audio = False

    ############### LEVEL NORMALIZATION ###########################
    #generates the normalization reference for dB compensation.
    if plan_name == 'normalization':

        setting_str_groundtruth = doce.Setting(experiment.groundtruth, [setting.dataset], positional=False).identifier()
        fname_name = setting_str_groundtruth+'_fname.npy'
        annot_files = np.load(experiment.path.groundtruth+fname_name)

        count = 1
        count_frames = 0
        level_db_list = []

        for subdir, dirs, files in os.walk(data_dir):
            for file in files:        
                    if not any((file.endswith('.mp3'), file.endswith('.wav'))):
                        print(f'WARNING: ONLY MP3 AND WAV FORMAT ARE SUPPORTED, file {file} will be skipped')
                        continue

                    if not file[:-4] in annot_files:
                        print(f'WARNING: file {file} had no annotations')
                        continue

                    f = os.path.join(subdir, file)
                    audio = librosa.load(f, sr=32000)[0]
                    spectro = np.abs(librosa.stft(audio, n_fft=4096, hop_length=4000))
                    #idx3 --> ~24Hz, idx1600 --> ~12500Hz (approximated frequency range of third-octave spectrograms)
                    spectro = spectro[3:1600, :]
                    level_db = np.mean(20*np.log10(spectro+10e-10), axis=0)
                    level_db_list = np.concatenate((level_db_list,level_db))
                    count+=1
                    count_frames += len(level_db)

        # Calculate the threshold value for the 99th percentile
        threshold = np.percentile(level_db_list, 99)

        print(f'dB normalization: {threshold}')
        np.save(experiment.path.normalization+setting.identifier()+'_normalization.npy', threshold)
    else:
        if plan_name != "groundtruth":
            # load the normalization coefficient
            setting_str_normalization = doce.Setting(experiment.normalization, [setting.dataset], positional=False).identifier()
            normalization_name = setting_str_normalization+'_normalization.npy'
            db_offset = - np.load(experiment.path.normalization+normalization_name)

    ############### LEVEL CALCULATION ###########################
    #the level plan generates the level in dB of each audio file. The reference for the dB calculation is set to the minimum value
    #in the dataset (this way the level is always greater than 0)
    if plan_name == 'level':
            db_offset_multiplier = 10**(db_offset/10)
            melspec_layer = MelSpectrogram(
                            n_mels=128,
                            sample_rate=32000,
                            n_fft=1024,
                            win_length=1024,
                            hop_length=256,
                            f_min=0.0,
                            f_max=(32000 / 2.0),
                            center=True,
                            power=2.0,
                            mel_scale="slaney",
                            norm="slaney",
                            normalized=True,
                            pad_mode="constant",
                        )
            cpt = 0
            #calculation of the level of each file
            for subdir, dirs, files in os.walk(data_dir):
                for file in files:
                    f = os.path.join(subdir, file)
                    wav, sr = librosa.load(f, sr=32000)
                    wav = wav * db_offset_multiplier
                    torchwav = torch.Tensor(wav).unsqueeze(0)
                    melspec = melspec_layer(torchwav)
                    melspec = 10 * torch.log10(melspec + 1e-10)
                    melspec = torch.clamp((melspec + 100) / 100, min=0.0, max=1.0)
                    level = torch.mean(melspec, dim=(0, 1))
                    level = level.detach().cpu().numpy()
                    np.save(experiment.path.level+setting.identifier()+'_level_'+ file[:n_to_delete] + '.npy', level)
                    cpt+=1
                    print(cpt)

            np.save(experiment.path.level+setting.identifier()+'_sr.npy', sr)

    ############### GROUNDTRUTH CALCULATION ###########################
    # the groundtruth of each dataset is calculated in this section.
    # Note that the level plan must be computed before the groundtruth plan, because the 
    # levels are used for weighting the annotations.
    if plan_name == 'groundtruth':
            
            if setting.dataset == 'Grafic':
                groundtruth, fname = cg.compute_groundtruth(setting, experiment.path.detection, data_dir)
            if setting.dataset == 'Aumilab':
                groundtruth, fname = ca.compute_groundtruth(setting, experiment, experiment.path.detection, data_dir)
            if setting.dataset == 'SingaPura':
                groundtruth, fname = cs.compute_groundtruth(setting, experiment, experiment.path.detection, data_dir)
            if setting.dataset == 'Lorient1k':
                groundtruth, fname = cl.compute_groundtruth()
            if setting.dataset == 'Aumilab-MT':
                groundtruth, fname = camt.compute_groundtruth()

            np.save(experiment.path.groundtruth+setting.identifier()+'_groundtruth.npy', groundtruth)
            np.save(experiment.path.groundtruth+setting.identifier()+'_fname.npy', fname)

    ############### CORRELATIONS CALCULATION ###########################
    if plan_name in ['reference', 'deep', 'merged'] :
        #choose a classifier, and the its related informations
        if setting.classifier == 'YamNet':
            classif_model = ysed(normalize=normalize, db_offset=db_offset)
        
        if setting.classifier == 'PANN':
            classif_model = psed(constant_10s_audio=constant_10s_audio, normalize=normalize, db_offset=db_offset, verbose=True, pann_type=pann_type)

        if setting.classifier == 'CNN-PINV-YamNet':
            classif_model = tysed(normalize=normalize)
        
        if setting.classifier == 'CNN-PINV-PANN':
            classif_model = tpsed(constant_10s_audio=constant_10s_audio, normalize=normalize, verbose=True, db_offset=db_offset, pann_type=pann_type)

        if setting.classifier == 'CNN-PINV-PANN-Slow':
            classif_model = tpsed_slow(constant_10s_audio=constant_10s_audio, normalize=normalize, verbose=False, db_offset=db_offset)

        if setting.classifier == 'TFSD':
            classif_model = tsed(db_offset=db_offset)

        if setting.classifier == 'felix':
            classif_model = fsed(dataset=setting.dataset)

        if setting.classifier == 'thirdoctave':
            classif_model = osed()

        # training mode for the models enhanced by deep learning in the deep plan
        if setting.step == 'train':

                #open groundtruth for training
                setting_str_groundtruth = doce.Setting(experiment.groundtruth, [setting.dataset], positional=False).identifier()
                groundtruth_name = setting_str_groundtruth+'_groundtruth.npy'
                fname_name = setting_str_groundtruth+'_fname.npy'

                groundtruth = np.load(experiment.path.groundtruth+groundtruth_name)
                fname =  np.load(experiment.path.groundtruth+fname_name)

                #open classifiers scores used for training
                setting_str_detection_mean = doce.Setting(experiment.reference, ['False', 'compute', setting.classifier, setting.dataset], positional=False).identifier()
                scores_name = setting_str_detection_mean + '_detection_mean.npy'
                fname_scores_name = setting_str_detection_mean + '_fname.npy'

                scores = np.load(experiment.path.detection_mean+scores_name)
                fname_scores = np.load(experiment.path.detection_mean+fname_scores_name)
                
                #only keep common scores and groundtruth
                diff_files = np.setdiff1d(fname_scores,fname)
                if len(fname) != len(fname_scores):
                    print(f'WARNING: some groundtruth files are not present in the classifier scores. Here are the files that are causing trouble: {diff_files}')
                    print('The training will not be computed on this file')

                fcommon_groundtruth = np.nonzero(np.in1d(fname, fname_scores))[0]
                fcommon_scores = np.nonzero(np.in1d(fname_scores, fname))[0]

                fname = fname[fcommon_groundtruth]
                fname_scores = fname[fcommon_groundtruth]
                groundtruth = groundtruth[fcommon_groundtruth]
                scores = scores[fcommon_scores]
                
                if not all(fname == fname_scores):
                    raise Exception("Your groundtruth and your classifier scores have different files, please check before running training mode") 

                np.save(experiment.path.model+setting.identifier()+'_fname.npy', fname)

                if not setting.dataset in ["Grafic", "Lorient1k"]:
                    trainer = DLModelsTrainer(experiment.path.model, scores, groundtruth, fname, classif_model.n_labels)
                    losses_train_fold, losses_eval_fold = trainer.train()
                    np.save(experiment.path.loss+setting.identifier()+'_loss_train.npy', losses_train_fold)
                    np.save(experiment.path.loss+setting.identifier()+'_loss_eval.npy', losses_eval_fold)

                    eval_fold = trainer.eval_fold
                    max_size = np.max([len(eval) for eval in eval_fold])
                    #add -1 for folds loss list where length is not enough: allows to store and load as an array, and to use np.where afterwards
                    eval_fold = [np.pad(eval, (0,max_size-len(eval)), mode='constant', constant_values=-1) if len(eval) != max_size else np.array(eval) for eval in eval_fold]
                    
                    train_fold = trainer.train_fold
                    max_size = np.max([len(train) for train in train_fold])
                    #add -1 for folds loss list where length is not enough: allows to store and load as an array, and to use np.where afterwards
                    train_fold = [np.pad(train, (0,max_size-len(train)), mode='constant', constant_values=-1) if len(train) != max_size else np.array(train) for train in train_fold]

                    np.save(experiment.path.model+setting.identifier()+'_train_fold.npy', train_fold)
                    np.save(experiment.path.model+setting.identifier()+'_eval_fold.npy', eval_fold)

                    with open((experiment.path.model+setting.identifier()+'_model_fold'), 'wb') as f:
                        pickle.dump(trainer.model_fold, f)

        #compute the classifier on the dataset
        if setting.step == 'compute':

            #NO DEEP 
            if setting.deep == "False":
                f_scores_mean = []
                f_scores_name = []
                cpt = sum([len(files) for r, d, files in os.walk(data_dir)])
                idx_cur_file = 1

                #MT: test, to remove
                # LL = []
                for subdir, dirs, files in os.walk(data_dir):
                    for file in files:        
                            if not any((file.endswith('.mp3'), file.endswith('.wav'))):
                                print(f'WARNING: ONLY MP3 AND WAV FORMAT ARE SUPPORTED, file {file} will be skipped')
                                continue
                            f_test = os.path.join(experiment.path.detection, setting.identifier()+'_detection_'+ file[:n_to_delete]+'.npy')
                            if (os.path.exists(f_test)) & (keep_existing):
                                print(f'file {f_test} already exists, passing calculation')
                                idx_cur_file += 1
                                scores = np.load(experiment.path.detection+setting.identifier()+'_detection_'+ file[:n_to_delete] + '.npy')
                                f_scores_mean.append(np.mean(scores, axis=0))
                                f_scores_name.append(file[:n_to_delete])
                                continue

                            f = os.path.join(subdir, file)
                            # print('FILE CURRENTLY CALCULATED')
                            # print(f)

                            if 'PANN' in setting.classifier:
                                scores = classif_model.inference_from_scratch(f, mean=mean, to_tvb=False)
                            else:
                                scores = classif_model.inference_from_scratch(f)

                            np.save(experiment.path.detection+setting.identifier()+'_detection_'+ file[:n_to_delete] + '.npy', scores)
                            f_scores_mean.append(np.mean(scores, axis=0))
                            f_scores_name.append(file[:n_to_delete])
                            print(f'\rCOMPUTED: {idx_cur_file} / {cpt}')
                            idx_cur_file += 1

                print('LENGTH OF NAMES')
                print(len(f_scores_name))
                np.save(experiment.path.detection_mean+setting.identifier()+'_detection_mean.npy', f_scores_mean)
                np.save(experiment.path.detection_mean+setting.identifier()+'_fname.npy', f_scores_name)

            #DEEP: trains a small model to fit traffic, voices and birds predictions. Only works on large datasets such as SingaPura or Aumilab.
            if setting.deep == "True":
                f_scores_mean = []
                f_scores_name = []
                if setting.dataset in ["Grafic", "Lorient1k"]:
                    str_model = doce.Setting(experiment.deep, [setting.deep, 'train', setting.classifier, 'Aumilab'], positional=False).identifier()
                    model_fold_name = str_model+'_model_fold'
                    train_fold_name = str_model+'_train_fold.npy'
                    eval_fold_name = str_model+'_eval_fold.npy'
                    model = FC(classif_model.n_labels)
                    print('model fold')
                    print(experiment.path.model+model_fold_name)
                    with open(experiment.path.model+model_fold_name, 'rb') as f:
                        model_fold = pickle.load(f)

                    loaded_model_fold = [copy.deepcopy(model) for k in range(len(model_fold))]
                    for idx, state_dict in enumerate(model_fold):
                        loaded_model_fold[idx].load_state_dict(state_dict)
                else:
                    str_model = doce.Setting(experiment.deep, [setting.deep, 'train', setting.classifier, setting.dataset], positional=False).identifier()
                    model_fold_name = str_model+'_model_fold'
                    train_fold_name = str_model+'_train_fold.npy'
                    eval_fold_name = str_model+'_eval_fold.npy'
                    model = FC(classif_model.n_labels)
                    print('model fold')
                    print(experiment.path.model+model_fold_name)
                    with open(experiment.path.model+model_fold_name, 'rb') as f:
                        model_fold = pickle.load(f)

                    loaded_model_fold = [copy.deepcopy(model) for k in range(len(model_fold))]
                    for idx, state_dict in enumerate(model_fold):
                        loaded_model_fold[idx].load_state_dict(state_dict)

                if setting.dataset not in ['Grafic', 'Lorient1k']:
                    train_fold = np.load(experiment.path.model+train_fold_name)
                    eval_fold = np.load(experiment.path.model+eval_fold_name)
                    fname_name = str_model+'_fname.npy'
                    fname = np.load(experiment.path.model+fname_name)
                else:
                    str_model = doce.Setting(experiment.deep, [setting.deep, 'train', setting.classifier, setting.dataset], positional=False).identifier()
                    fname_name = str_model+'_fname.npy'
                    fname = np.load(experiment.path.model+fname_name)

                cpt = 0
                for subdir, dirs, files in os.walk(experiment.path.detection):
                    cpt = sum([len(files) for r, d, files in os.walk(data_dir)])
                    idx_cur_file = 1
                    for file in files:
                        str_check = doce.Setting(experiment.reference, ['False', 'compute', setting.classifier, setting.dataset], positional=False).identifier()
                        if file.startswith(str_check):
                            f = os.path.join(subdir, file)
                            wav_file = file.replace(str_check, "")[11:n_to_delete]
                            cond = np.where(fname == wav_file)
                            if len(cond[0]) == 0:
                                print(f"File {file} isn't found in the list: precessing with first model")
                                model = loaded_model_fold[0]
                            elif setting.dataset in ['Grafic', 'Lorient1k']:
                                model = loaded_model_fold[0]
                            else:
                                idx_model = np.where(cond == eval_fold)[0][0]
                                model = loaded_model_fold[idx_model]

                            x = np.load(f).astype(np.float32)
                            x = torch.from_numpy(x)

                            new_scores = model(x).mean(axis=0).cpu().detach().numpy()
                            np.set_printoptions(precision=3)

                            new_scores = np.expand_dims(new_scores, axis=0)
                            np.save(experiment.path.detection+setting.identifier()+'_detection_'+ wav_file + '.npy', new_scores)
                            print(f'\rCOMPUTED (DEEP): {idx_cur_file} / {cpt}')
                            idx_cur_file += 1
                            f_scores_mean.append(np.mean(new_scores, axis=0))
                            f_scores_name.append(file[:n_to_delete])

                print('LENGTH OF NAMES')
                print(len(f_scores_name))
                np.save(experiment.path.detection_mean+setting.identifier()+'_detection_mean.npy', f_scores_mean)
                np.save(experiment.path.detection_mean+setting.identifier()+'_fname.npy', f_scores_name)
            #################################
            #################################

        #calculates the correlations between the annotated groundtruth t,v,b and the predictions of the model
        if setting.step == 'metric':
                if (setting.deep == "False") & (setting.classifier == "thirdoctave"):
                    print('NO METRIC AVAILABLE FOR THIRD OCTAVE IN REFERENCE PLAN, PASSING CALCULATION')
                else:
                    setting_str_groundtruth = doce.Setting(experiment.groundtruth, [setting.dataset], positional=False).identifier()
                    setting_str_groundtruth = doce.Setting(experiment.groundtruth, [setting.dataset], positional=False).identifier()
                    groundtruth_name = setting_str_groundtruth+'_groundtruth.npy'
                    fname_name = setting_str_groundtruth+'_fname.npy'

                    groundtruth = np.load(experiment.path.groundtruth+groundtruth_name)
                    fname = np.load(experiment.path.groundtruth+fname_name)

                    if setting.dataset == 'Grafic':
                        correlation_table = cg.compute_metric(setting, experiment.path.detection, data_dir, groundtruth, fname, to_tvb=not mean)
                    if setting.dataset == 'Aumilab':
                        correlation_table = ca.compute_metric(setting, experiment.path.detection, data_dir, groundtruth, fname, to_tvb=not mean)
                    if setting.dataset == 'SingaPura':
                        correlation_table = cs.compute_metric(setting, experiment.path.detection, data_dir, groundtruth, fname, to_tvb=not mean)
                    if setting.dataset == 'Lorient1k':
                        correlation_table = cl.compute_metric(setting, experiment.path.detection, data_dir, groundtruth, fname, to_tvb=not mean)
                    if setting.dataset == 'Aumilab-MT':
                        correlation_table = camt.compute_metric(setting, experiment.path.detection, data_dir, groundtruth, fname, to_tvb=not mean)

                    print("CORRELATIONS")
                    print(correlation_table)
                    corr_t = correlation_table[0,0]
                    corr_v = correlation_table[1,1]
                    corr_b = correlation_table[2,2]
                    corr_global = np.mean([corr_t, corr_v, corr_b])

                    np.save(experiment.path.correlation+setting.identifier()+'_corr_t.npy', corr_t)
                    np.save(experiment.path.correlation+setting.identifier()+'_corr_v.npy', corr_v)
                    np.save(experiment.path.correlation+setting.identifier()+'_corr_b.npy', corr_b)
                    np.save(experiment.path.correlation+setting.identifier()+'_corr_global.npy', corr_global)



    duration = time.time() - start_time
    np.save(experiment.path.time+setting.identifier()+'_duration.npy', duration)
    print("--- %s seconds ---" % (duration))
    
        
# invoke the command line management of the doce package
if __name__ == "__main__":
  doce.cli.main(experiment = experiment, func=step)