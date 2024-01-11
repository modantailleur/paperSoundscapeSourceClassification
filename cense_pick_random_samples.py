import h5py
import datetime
import os
import numpy as np
import pickle 
import pandas as pd
import random
import datetime as dt
from datetime import datetime, timedelta, date
import argparse

random.seed(0)

def which_semester(date):
    if 6 >= date.month >= 1:
        semester = 1
    else:
        semester = 2
    return(semester)

def which_season(date):
    year = date.year
    date_no_hour = datetime(date.year, date.month, date.day)
    seasons = [('winter', (datetime(year, 1, 1), datetime(year, 3, 20))),
               ('spring', (datetime(year, 3, 21), datetime(year, 6, 20))),
               ('summer', (datetime(year, 6, 21), datetime(year, 9, 22))),
               ('fall', (datetime(year, 9, 23), datetime(year, 12, 20))),
               ('winter', (datetime(year, 12, 21), datetime(year, 12, 31)))]
    found_season = None
    for season, (start, end) in seasons:
        if start <= date_no_hour <= end:
            found_season = season
    return(found_season)

def is_lockdown(date):
    # Set the start and end dates of the lockdown period
    lockdown_start = datetime(2020, 3, 17)
    lockdown_end = datetime(2020, 5, 3)
    if lockdown_start <= date <= lockdown_end:
        check = True
    else:
        check = False
    return(check)

def is_weekend(date):
    day_of_week = date.strftime("%A")
    if day_of_week in ['Saturday', 'Sunday']:
        check = True
    else:
        check = False
    return(check)

def is_just_before_sunrise(date, sunrise_from_hist):

    sunrise_24h = datetime.strptime(sunrise_from_hist, "%I:%M%p")
    sunrise_24h = datetime.combine(dt.date(date.year, date.month, date.day), 
                          dt.time(sunrise_24h.hour, sunrise_24h.minute))
    
    max_before_sunrise = sunrise_24h - timedelta(minutes=30)
    min_before_sunrise = sunrise_24h - timedelta(minutes=90)

    if min_before_sunrise <= date <= max_before_sunrise:
        check = True
    else:
        check = False
    return(check)

def is_night(date):

    if 6 < date.hour < 22:
        check=False
    else:
        check = True

    return(check) 

def is_raining(date, precip_from_hist):

    if precip_from_hist >= 1:
        check = True
    else:
        check=False
    return(check)

def from_id_to_h5_file(h5_path, id):
    for subdir, dirs, files in os.walk(h5_path):
        for file in files:
            with h5py.File(h5_path+file, "r") as f:
                try:
                    if f.attrs["map_id_sensor"] == id:
                        return(file)
                except KeyError:
                    pass
    return(None)

def from_h5_file_to_id(h5_path, h5_file):
            with h5py.File(h5_path+h5_file, "r") as f:
                try:
                    id = f.attrs["map_id_sensor"]
                    return(id)
                except KeyError:
                    return(None)

class RandomMinuteInfoFromDate():
    def __init__(self, weather_hist, date):
        """
        This creates an instance with info from a random day of the year, within the time delta_date
        Weather hist is a pandas dataframe with the weather.
        
        """
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)

        self.precise_date = date.replace(hour=random_hours, minute=random_minutes)
        #date attributes
        self.hour = random_hours
        self.minute = random_minutes
        self.hour = self.precise_date.hour
        self.month = self.precise_date.month
        self.semester = which_semester(self.precise_date)
        self.season = which_season(self.precise_date)
        self.day_of_week = date.strftime("%A")
        self.lockdown = is_lockdown(self.precise_date)
        self.weekend = is_weekend(self.precise_date)
        self.night = is_night(self.precise_date)

        # get weather information of the date
        self.date_hist = weather_hist.loc[weather_hist["date_time"]==self.precise_date.strftime('%F')]
        self.sunrise_from_hist = self.date_hist["sunrise"].values[0].replace(" ", "")
        self.sunset_from_hist = self.date_hist["sunset"].values[0].replace(" ", "")
        self.precip_from_hist = self.date_hist["precipMM"].values[0]
        self.cloudcover_from_hist = self.date_hist["cloudcover"].values[0]

        #date weather attributes
        self.just_before_sunrise = is_just_before_sunrise(self.precise_date, self.sunrise_from_hist)
        self.raining = is_raining(self.precise_date, self.precip_from_hist)

class InfoFromDate():
    def __init__(self, weather_hist, date, cur_hour, cur_min):
        """
        This creates an instance with info from a random day of the year, within the time delta_date
        Weather hist is a pandas dataframe with the weather.
        
        """

        self.precise_date = date.replace(hour=cur_hour, minute=cur_min)
        #date attributes
        self.hour = cur_hour
        self.minute = cur_min
        self.hour = self.precise_date.hour
        self.month = self.precise_date.month
        self.semester = which_semester(self.precise_date)
        self.season = which_season(self.precise_date)
        self.day_of_week = date.strftime("%A")
        self.lockdown = is_lockdown(self.precise_date)
        self.weekend = is_weekend(self.precise_date)
        self.night = is_night(self.precise_date)

        # get weather information of the date
        self.date_hist = weather_hist.loc[weather_hist["date_time"]==self.precise_date.strftime('%F')]
        self.sunrise_from_hist = self.date_hist["sunrise"].values[0].replace(" ", "")
        self.sunset_from_hist = self.date_hist["sunset"].values[0].replace(" ", "")
        self.precip_from_hist = self.date_hist["precipMM"].values[0]
        self.cloudcover_from_hist = self.date_hist["cloudcover"].values[0]

        #date weather attributes
        self.just_before_sunrise = is_just_before_sunrise(self.precise_date, self.sunrise_from_hist)
        self.raining = is_raining(self.precise_date, self.precip_from_hist)

def pick_random_samples(start_date, end_date, sensors=["p0720", "p0310", 'p0640'], files_per_day=None, cet_date=False, h5_path='./cense_data/', output_path='./cense_exp/spectral_data/'):
    """
    sensors: "all", ["sensor1", "sensor2" etc...]
    """
    idx_spec_to_save = 0
    weather_hist = pd.read_csv('lorient_weather.csv')  

    if sensors is not None:
        if sensors == "all":
            h5_files = [f for f in os.listdir(h5_path) if f.endswith('.h5')]
        else:
            h5_files = [from_id_to_h5_file(h5_path, sensor) for sensor in sensors]

    # Choose 30 unique pairs of h5 files and epoch times
    couples = []
    hour_list = []
    month_list = []
    semester_list = []
    season_list = []
    day_of_week_list = []
    lockdown_list = []
    weekend_list = []
    night_list = []
    laeq_list = []
    leq_list = []
    just_before_sunrise_list = []
    raining_list = []
    h5_list = []
    id_list = []
    date_list = []
    spectral_data_list = []

    censor_count = 0

    for h5_file in h5_files:
        censor_count += 1
        id_sensor = from_h5_file_to_id(h5_path, h5_file)
        print('NEW SENSOR')
        print(f'sensor {id_sensor} number {censor_count}')
        with h5py.File(h5_path+h5_file, "r") as f:
            try:
                latitude = f.attrs['lat']
                longitude = f.attrs['long']
                position = (latitude, longitude)
            except KeyError:
                print("NO LATITUDE AND LONGITUDE FOR THIS FILE, SKIPPING PROCESS")
                continue
            
            presence = None
            for key_layer_1 in f:
                for key_layer_2 in f[key_layer_1]:
                    year, month = map(int, key_layer_1.split('_'))
                    day = int(key_layer_2)
                    date = datetime(year, month, day)
                    if start_date <= date < end_date:
                        try:
                            data = f[key_layer_1][key_layer_2]['fast_125ms']
                        except KeyError:
                            print(f'ERROR: couldnt find fast125ms in file, passing calculation for file {h5_file}')
                            continue
                        
                        print('DATE')
                        print((year, month, day))

                        data_np = data[()]

                        df = pd.DataFrame(data_np, columns = data_np.dtype.names)

                        df['epoch'] = df['epoch'].apply(lambda x: int(x/1000))

                        winter_time = datetime(year, 10, 29)
                        summer_time = datetime(year, 3, 28)
                        if (summer_time > date > winter_time) or (cet_date):
                            print('WINTER')
                            df['datetime'] = pd.to_datetime(df['epoch'], unit='s').dt.tz_localize('UTC').dt.tz_convert('CET')
                        else:
                            print('SUMMER')
                            df['datetime'] = pd.to_datetime(df['epoch'], unit='s').dt.tz_localize('UTC').dt.tz_convert('EET')
                        failures = 0
                        found = 0
                        cur_date_list = []
                        
                        if files_per_day is not None:
                            while (failures + found < files_per_day):
                                info = RandomMinuteInfoFromDate(weather_hist, date)
                                while info.precise_date in cur_date_list:
                                    info = RandomMinuteInfoFromDate(weather_hist, date)
                                
                                df_date = df[(df['datetime'].dt.hour == info.hour) & (df['datetime'].dt.minute==info.minute)]

                                spectral_data = df_date.iloc[: , 1:-3].to_numpy()
                                laeq = df_date.iloc[: , -2:-1].to_numpy()[:,0]
                                leq = df_date.iloc[: , -3:-2].to_numpy()[:,0]

                                if spectral_data.shape[0] != 480:
                                    print('FAIL: too few spectral data, continuing')
                                    failures += 1
                                    continue
                                
                                couples.append((h5_file, info.precise_date))
                                h5_list.append(h5_file)
                                id_list.append(id_sensor)
                                date_list.append(info.precise_date)
                                hour_list.append(info.hour)
                                month_list.append(info.month)
                                semester_list.append(info.semester)
                                season_list.append(info.season)
                                day_of_week_list.append(info.day_of_week)
                                lockdown_list.append(info.lockdown)
                                weekend_list.append(info.weekend)
                                night_list.append(info.night)
                                just_before_sunrise_list.append(info.just_before_sunrise)
                                raining_list.append(info.raining)
                                spectral_data_list.append(spectral_data)
                                laeq_list.append(np.mean(laeq))
                                leq_list.append(np.mean(leq))

                                idx_spec_to_save += 1

                                print(f'file idx: {idx_spec_to_save}. failures: {failures}')

                                found += 1
                                cur_date_list.append(info.precise_date)
                        else:
                            for cur_hour in range(24):
                                for cur_min in range(60):
                                    info = InfoFromDate(weather_hist, date, cur_hour, cur_min)
                                    df_date = df[(df['datetime'].dt.hour == info.hour) & (df['datetime'].dt.minute==info.minute)]

                                    spectral_data = df_date.iloc[: , 1:-3].to_numpy()
                                    laeq = df_date.iloc[: , -2:-1].to_numpy()[:,0]
                                    leq = df_date.iloc[: , -3:-2].to_numpy()[:,0]

                                    if spectral_data.shape[0] != 480:
                                        print('FAIL: too few spectral data, continuing')
                                        failures += 1
                                        continue
                                    
                                    couples.append((h5_file, info.precise_date))
                                    h5_list.append(h5_file)
                                    id_list.append(id_sensor)
                                    date_list.append(info.precise_date)
                                    hour_list.append(info.hour)
                                    month_list.append(info.month)
                                    semester_list.append(info.semester)
                                    season_list.append(info.season)
                                    day_of_week_list.append(info.day_of_week)
                                    lockdown_list.append(info.lockdown)
                                    weekend_list.append(info.weekend)
                                    night_list.append(info.night)
                                    just_before_sunrise_list.append(info.just_before_sunrise)
                                    raining_list.append(info.raining)
                                    spectral_data_list.append(spectral_data)
                                    laeq_list.append(np.mean(laeq))
                                    leq_list.append(np.mean(leq))

                                    idx_spec_to_save += 1

                                    print(f'file idx: {idx_spec_to_save}. failures: {failures}')

                                    found += 1
                                    cur_date_list.append(info.precise_date)

    data_dict = {
        'h5': h5_list,
        'id_sensor': id_list,
        'date': date_list,
        'hour': hour_list,
        'month': month_list,
        'semester': semester_list,
        'season': season_list,
        'day_of_week': day_of_week_list,
        'lockdown': lockdown_list,
        'weekend': weekend_list,
        'night': night_list,
        'just_before_sunrise': just_before_sunrise_list,
        'raining': raining_list,
        'laeq': laeq_list,
        'leq': leq_list,
        'spectral_data': np.array(spectral_data_list)
    }

    if not os.path.exists(output_path):
        # Create the directory recursively
        os.makedirs(output_path)

    if sensors == "all":
        dict_file_path = output_path + 'cense_lorient_spectral_data_with_' + str(idx_spec_to_save) + '_files_all_sensors_start_' + \
                            str(start_date.year) + str(start_date.month) + str(start_date.day) + '_end_' + \
                            str(end_date.year) + str(end_date.month) + str(end_date.day) 
    else:
        sensors_str = '_'.join(sensors)
        dict_file_path = output_path + 'cense_lorient_spectral_data_with_' + str(idx_spec_to_save) + '_files__' + \
            sensors_str + '__' \
                                        + 'start_' + str(start_date.year) + str(start_date.month) + str(start_date.day) + '_end_' + \
                            str(end_date.year) + str(end_date.month) + str(end_date.day) 

    
    with open(dict_file_path, 'wb') as file:
        pickle.dump(data_dict, file)

def main(config):

    if config.desc == 'test':
        # data used for the traffic, voices and birds map
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 2)
        sensors=["p0720", "p0310", 'p0640']
        files_per_day = 3
        cet_date = False

    if config.desc == 'winter2020':
        # data used for the traffic, voices and birds map
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 3, 1)
        sensors="all"
        files_per_day = 10
        cet_date = False

    if config.desc == 'winter2020-3s':
        # data used for the traffic, voices and birds clock graph (only 3 sensors)
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 3, 1)
        sensors=["p0720", "p0310", 'p0640']
        files_per_day = 200
        cet_date = False

    if config.desc == 'music_festival':
        # this corresponds to a Sunday of the Interceltique de Lorient 2021 festival
        start_date = datetime(2021, 8, 8)
        end_date = datetime(2021, 8, 9)
        sensors="all"
        files_per_day = None
        cet_date = False
    
    if config.desc == 'no_music_festival':
        start_date = datetime(2021, 7, 1)
        end_date = datetime(2021, 8, 1)
        sensors="all"
        files_per_day = None
        cet_date = False
    
    if config.desc == 'church_functional':
        # time period where the church bells were functional. p0480 is a sensor close to them.
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 2, 1)
        sensors=["p0480"]
        files_per_day = None
        cet_date = False

    if config.desc == 'church_not_functional':
        # time period where the church bells were not functional:
        # https://www.ouest-france.fr/bretagne/lorient-56100/lorient-muettes-les-cloches-de-saint-louis-ont-le-bourdon-7050817
        start_date = datetime(2020, 10, 1)
        end_date = datetime(2020, 11, 1)
        sensors=["p0480"]
        files_per_day = None
        cet_date = False

    pick_random_samples(start_date=start_date, end_date=end_date, sensors=sensors, files_per_day=files_per_day, cet_date=cet_date, h5_path=config.h5_path, output_path=config.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 1s Mels and Third-Octave spectrograms')

    parser.add_argument('--h5_path', type=str, default="./cense_data/",
                        help='The path where the h5 files of Cense Lorient are stored')
    parser.add_argument('--output_path', type=str, default="./cense_exp/spectral_data/",
                        help='The path where to store the spectral data')
    parser.add_argument('--desc', type=str, default="test",
                        help='The type of plot for which the data is retrieved ("winter2020", "winter2020-3s", "music_festival", "no_music_festival", "church_functional", "church_not_functional")')
    config = parser.parse_args()
    main(config)