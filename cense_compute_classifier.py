import pandas as pd
import torch
from tqdm import tqdm
from transcoder.transcoders import ThirdOctaveToMelTranscoderPinv, ThirdOctaveToMelTranscoder
import pickle
import numpy as np
import maad
from tfsd.cense_inference_tfsd import tfsd
from censeModels.felix_inference_dataset import FelixInference
import argparse

class DatasetGenerator(object):
    def __init__(self, cense_data_path):
        self.cense_data_path = cense_data_path
        with open(cense_data_path, 'rb') as pickle_file:
            self.data_dict = pickle.load(pickle_file)
        self.spectral_data = self.data_dict['spectral_data']
        self.laeq = self.data_dict['laeq']
        data_dict_without_spectral = {key: value for key, value in self.data_dict.items() if key != 'spectral_data'}
        self.df = pd.DataFrame(data_dict_without_spectral)
        self.len_dataset = len(self.spectral_data)

    def __getitem__(self, idx):
        spectral_data = self.spectral_data[idx]
        laeq = self.laeq[idx]
        return spectral_data, laeq

    def __len__(self):
        return self.len_dataset

#for CNN + PINV
class TranscoderPANNEvaluater:
    def __init__(self, transcoder, eval_dataset, dtype=torch.FloatTensor, db_compensation=-94):
        self.dtype = dtype
        self.transcoder = transcoder
        self.eval_dataset = eval_dataset
        self.db_compensation = db_compensation

    def evaluate(self, batch_size=32, device=torch.device("cpu")):
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        eval_outputs = np.array([])

        for (spectral_data, _) in tqdm_it:
            spectral_data = spectral_data.type(self.dtype)
            spectral_data = spectral_data.to(device)
            
            spectral_data = spectral_data + self.db_compensation
            _ , presence = self.transcoder.thirdo_to_mels_to_logit(spectral_data, frame_duration=10)

            presence = torch.mean(presence, axis=-1)
            if len(eval_outputs) != 0:
                eval_outputs = torch.cat((eval_outputs, presence), dim=0)
            else:
                eval_outputs = presence
        eval_outputs = eval_outputs.detach().cpu().numpy()
        return(eval_outputs)

#for acoustic indicators (laeq, )
class AcousticEvaluater:
    def __init__(self, eval_dataset, dtype=torch.FloatTensor, db_compensation=-94):
        self.dtype = dtype
        self.eval_dataset = eval_dataset
        self.db_compensation = db_compensation
        self.fn=np.array([20, 25, 31, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500])

    def evaluate(self, batch_size=1, device=torch.device("cpu")):
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        eval_outputs = np.array([])

        for (spectral_data, _) in tqdm_it:
            spectral_data = spectral_data.type(self.dtype)
            spectral_data = spectral_data.to(device)
            
            spectral_data = spectral_data + self.db_compensation

            spectral_data = spectral_data.detach().cpu().numpy()

            scores_voices = tfsd(spectral_data[0],self.fn,flim=(500,1500)) 
            scores_birds = tfsd(spectral_data[0],self.fn,flim=(1500,6000)) 

            presence = torch.Tensor([scores_voices, scores_birds])
            presence = presence.unsqueeze(dim=0)
            # presence = inference_tfsd(spectral_data=spectral_data, batch_size=480)
            # presence[:, 0] = laeq
            # presence = presence.mean(axis=0)
            
            if len(eval_outputs) != 0:
                eval_outputs = torch.cat((eval_outputs, presence), dim=0)
            else:
                eval_outputs = presence
        eval_outputs = eval_outputs.detach().cpu().numpy()
        return(eval_outputs)

#for acoustic indicators (laeq, )
class FelixEvaluater:
    def __init__(self, felixinf, eval_dataset, dtype=torch.FloatTensor):
        self.dtype = dtype
        self.felixinf = felixinf
        self.eval_dataset = eval_dataset
        self.fn=np.array([20, 25, 31, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500])

    def evaluate(self, batch_size=1, device=torch.device("cpu")):
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        eval_outputs = np.array([])

        for (spectral_data, _) in tqdm_it:
            spectral_data = spectral_data.type(self.dtype)
            spectral_data = spectral_data.to(device)
            
            presence = self.felixinf.inference_from_thirdo(spectral_data)
            
            presence = np.mean(presence, axis=0)
            presence = np.expand_dims(presence, axis=0)
            if len(eval_outputs) != 0:
                eval_outputs = np.concatenate((eval_outputs, presence), axis=0)
            else:
                eval_outputs = presence
        return(eval_outputs)

class LevelEvaluater:
    def __init__(self, eval_dataset, dtype=torch.FloatTensor, db_compensation=-94):
        self.dtype = dtype
        self.eval_dataset = eval_dataset
        self.db_compensation = db_compensation
        self.fn=np.array([20, 25, 31, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500])

    def evaluate(self, batch_size=1, device=torch.device("cpu")):
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        eval_outputs = np.array([])

        for (spectral_data, _) in tqdm_it:
            spectral_data = spectral_data.type(self.dtype)
            spectral_data = spectral_data.to(device)
            
            spectral_data = torch.mean(spectral_data, axis=-1)

            # presence = inference_tfsd(spectral_data=spectral_data, batch_size=480)
            # presence[:, 0] = laeq
            # presence = presence.mean(axis=0)
            
            if len(eval_outputs) != 0:
                eval_outputs = torch.cat((eval_outputs, spectral_data), dim=0)
            else:
                eval_outputs = spectral_data
                        
        eval_outputs = eval_outputs.detach().cpu().numpy()
        return(eval_outputs)

def main(config):

    # The +26dB linked to the microphone sensitivity is already taken into account by the transcoder model (this is actually a bug, to be fixed in futur versions)
    # +32dB is the result of the db compensation calculation on winter2020 subset (see paper for more details).
    db_compensation = -94 + 32
    if config.desc == 'test':
        # data used for the traffic, voices and birds map
        start_date = "202011"
        end_date = "202012"
        sensors=["p0720", "p0310", 'p0640']
        n_file = 6
        cet_date = False

    if config.desc == 'winter2020':
        # data used for the traffic, voices and birds map
        start_date = "202011"
        end_date = "202031"
        sensors="all"
        n_file = 33443
        cet_date = False

    if config.desc == 'winter2020-3s':
        # data used for the traffic, voices and birds clock graph (only 3 sensors)
        start_date = "202011"
        end_date = "202031"
        sensors=["p0720", "p0310", 'p0640']
        n_file = 32312
        cet_date = False

    if config.desc == 'music_festival':
        # this corresponds to a Sunday of the Interceltique de Lorient 2021 festival
        start_date = "202188"
        end_date = "202189"
        sensors="all"
        n_file = 6675
        cet_date = False
    
    if config.desc == 'no_music_festival':
        start_date = "202171"
        end_date = "202181"
        sensors="all"
        n_file = 195968
        cet_date = False
    
    if config.desc == 'church_functional':
        # time period where the church bells were functional. p0480 is a sensor close to them.
        start_date = "202011"
        end_date = "202021"
        sensors=["p0480"]
        n_file = 36195        
        cet_date = False

    if config.desc == 'church_not_functional':
        # time period where the church bells were not functional:
        # https://www.ouest-france.fr/bretagne/lorient-56100/lorient-muettes-les-cloches-de-saint-louis-ont-le-bourdon-7050817
        start_date = "2020101"
        end_date = "2020111"
        sensors=["p0480"]
        n_file = 16456
        cet_date = False
    
    compute_predictions(classifier='level', sensors=sensors, db_compensation=db_compensation, start_date=start_date, end_date=end_date, n_file=n_file, output_path=config.output_path, spectral_path=config.spectral_path)
    compute_predictions(classifier='transcoder', sensors=sensors, db_compensation=db_compensation, start_date=start_date, end_date=end_date, n_file=n_file, output_path=config.output_path, spectral_path=config.spectral_path)
    # compute_predictions(classifier='acoustic', sensors=sensors, db_compensation=db_compensation, start_date=start_date, end_date=end_date, n_file=n_file, output_path=config.output_path, spectral_path=config.spectral_path)

def compute_predictions(classifier, sensors, db_compensation, start_date, end_date, n_file, output_path, spectral_path):

    #transcoder setup
    MODEL_PATH = "./reference_models"
    cnn_logits_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
    transcoder = 'cnn_pinv'
    dtype=torch.FloatTensor
    fs=32000
    force_cpu = False
    #manage gpu
    useCuda = torch.cuda.is_available() and not force_cpu

    if useCuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
        #MT: add
        device = torch.device("cuda:0")
    else:
        print('No CUDA available.')
        dtype = torch.FloatTensor
        ltype = torch.LongTensor
        #MT: add
        device = torch.device("cpu")

    batch_size = 1

    if sensors == 'all':
        cense_data_path = spectral_path + 'cense_lorient_spectral_data_with_'+str(n_file)+'_files_all_sensors_start_'+start_date+'_end_'+end_date
    else:
        sensors_str = '_'.join(sensors)
        cense_data_path = spectral_path + 'cense_lorient_spectral_data_with_'+str(n_file)+'_files__' + sensors_str + '__' + \
                                                'start_' + start_date + '_end_' + end_date

    dataset = DatasetGenerator(cense_data_path=cense_data_path)

    if classifier == 'transcoder':
        transcoder_cnn_logits_pann = ThirdOctaveToMelTranscoder(transcoder, cnn_logits_name, MODEL_PATH, device=device)
        evaluater = TranscoderPANNEvaluater(transcoder=transcoder_cnn_logits_pann, eval_dataset=dataset, db_compensation=db_compensation)

    if classifier == 'acoustic':
        evaluater = AcousticEvaluater(eval_dataset=dataset, db_compensation=0)
        batch_size = 1

    if classifier == 'felix':
        felixinf = FelixInference(dataset='CenseLorient')
        evaluater = FelixEvaluater(felixinf=felixinf, eval_dataset=dataset, dtype=dtype)
        batch_size = 1

    if classifier == 'level':
        evaluater = LevelEvaluater(eval_dataset=dataset, db_compensation=-94+26)
        batch_size = 1

    eval_outputs = evaluater.evaluate(batch_size=batch_size, device=device)

    if classifier != 'level':

        df_to_save = dataset.df.copy()

        if classifier == 'transcoder':
            classes_names = ['C_' + str(k) for k in range(eval_outputs.shape[1])]
        if classifier == 'acoustic':
            classes_names = ['tfsd_mid', 'tfsd_high']
        if classifier == 'felix':
            classes_names = ['t', 'v', 'b']
        classes_df = pd.DataFrame(eval_outputs, columns=classes_names)
        df_to_save = pd.concat([df_to_save, classes_df], axis=1)

        if sensors == 'all':
            df_to_save.to_pickle(output_path + 'cense_lorient_'+classifier+'_with_'+str(n_file)+'_files_'+'dbcompensation_'+str(db_compensation)+'_all_sensors_start_'+ start_date + '_end_' + end_date)
        else:
            sensors_str = '_'.join(sensors)
            df_to_save.to_pickle(output_path + 'cense_lorient_'+classifier+'_with_'+str(n_file)+'_files_'+'dbcompensation_'+str(db_compensation)+'__'+ sensors_str + '__' + 'start_'+ start_date + '_end_' + end_date)

    else:
        eval_outputs = eval_outputs.reshape(-1)
        threshold = np.percentile(eval_outputs, 99)

        print('THRESHOLD')
        print(threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 1s Mels and Third-Octave spectrograms')
    parser.add_argument('--spectral_path', type=str, default="./cense_exp/spectral_data/",
                        help='The path where the spectral data files of Cense Lorient are stored')
    parser.add_argument('--output_path', type=str, default="./cense_exp/predictions/",
                        help='The path where to store the predictions')
    parser.add_argument('--desc', type=str, default="test",
                        help='The type of plot for which the data is retrieved ("winter2020", "winter2020-3s", "music_festival", "no_music_festival", "church_functional", "church_not_functional")')
    config = parser.parse_args()
    main(config)