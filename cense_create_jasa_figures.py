import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import openpyxl
import folium
# from selenium.webdriver.firefox.options import Options
from selenium import webdriver
from PIL import Image
import os 
import time 
from selenium.webdriver.chrome.options import Options
from PIL import Image
from datetime import timedelta
import argparse

DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS_OF_YEAR = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

SHORT_DAYS_OF_WEEK = ['Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.', 'Sat.', 'Sun.']
SHORT_MONTHS_OF_YEAR = ["Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]

def encode_weekday(x):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return(encode_parameter(x, weekdays))

def encode_hour(x):
    hour = [k for k in range(0, 24)]
    return(encode_parameter(x, hour))

def encode_month(x):
    month = [k for k in range(1, 13)]
    return(encode_parameter(x, month))

def encode_parameter(x, parameter):
    parameter_enc = np.linspace(0, 2*np.pi, len(parameter)+1)
    parameter_enc = parameter_enc[:-1]
    parameter_encodings = {day: encoding for day, encoding in zip(parameter, parameter_enc)}
    return(parameter_encodings[x])

def preprocess_df(df, t_class, v_class, b_class, t_coef, v_coef, b_coef, keep_only_tvb=True):
    """
    Preprocessing of the pandas dataframe, to:
    - remove useless classes (3 classes are kept, the others are droped)
    - apply coefficient of the slope, from the correlation between the predictions and the annotations on Lorient1k
    - encode sensor_id with integer values, seasons, day of week, hour and month on circular values (cosine and sine), in order
       to use them in clustering. Applies a scaling factor on them, to manage the weight that it will have for the clustering.
    """
    if keep_only_tvb:
        #drop classes that are not the t_class, v_class or b_class
        columns_to_keep = [t_class, v_class, b_class]
        columns_to_drop = [col for col in df.columns if col.startswith('C_') and col not in columns_to_keep]
        df.drop(columns=columns_to_drop, inplace=True)

    #apply coefficient of the slope of Lorient1k on the predictions
    df[t_class] = df[t_class] * t_coef
    df[v_class] = df[v_class] * v_coef
    df[b_class] = df[b_class] * b_coef

    #clip prediction between 0 and 1
    df[t_class] = df[t_class].clip(0, 1)
    df[v_class] = df[v_class].clip(0, 1)
    df[b_class] = df[b_class].clip(0, 1)

    #create an encoder for each parameter that is not numerical, and store it in a new column with the name "enc+parameter"
    le_id_sensor = preprocessing.LabelEncoder()
    le_id_sensor.fit(np.unique(df["id_sensor"].to_numpy()))

    le_season = preprocessing.LabelEncoder()
    le_season.fit(np.unique(df["season"].to_numpy()))

    le_day_of_week = preprocessing.LabelEncoder()
    le_day_of_week.fit(np.unique(df["day_of_week"].to_numpy()))

    #create encoded one hot versions of each string parameter
    df["enc_id_sensor"] = le_id_sensor.transform(df["id_sensor"].to_numpy())
    df["enc_season"] = le_season.transform(df["season"].to_numpy())
    df["enc_day_of_week"] = le_day_of_week.transform(df["day_of_week"].to_numpy())

    #drop every NA
    df.dropna(inplace=True)

    #normalize the columns of the 3 indicators that we want to cluster
    # scaler = MinMaxScaler()
    # df["laeq"] = scaler.fit_transform(df[["laeq"]])
    # df["tfsd_mid"] = scaler.fit_transform(df[["tfsd_mid"]])
    # df["tfsd_high"] = scaler.fit_transform(df[["tfsd_high"]])
    #no normalization, just using the coefficient of the slope calculated on grafic:

    #encode weekday: if we want to cluster by weekday, weekday is organized on a circle (Monday to Sunday then Monday again). I've created 
    # a function encode_weekday that maps each day of the week to a value between 0 and 2pi and I calculate the cos and sin of that value 
    # so that each day of the week is mapped on a circle. I can then cluster by cos_weekday and sin_weekday. 
    scale = 1
    df['circle_day_of_week'] = df['day_of_week'].map(encode_weekday)
    df['cos_day_of_week'] = df['circle_day_of_week'].map(np.cos) * scale
    df['sin_day_of_week'] = df['circle_day_of_week'].map(np.sin) * scale

    scale = 1
    df['circle_hour'] = df['hour'].map(encode_hour)
    df['cos_hour'] = df['circle_hour'].map(np.cos) * scale
    df['sin_hour'] = df['circle_hour'].map(np.sin) * scale

    scale = 1
    df['circle_month'] = df['month'].map(encode_month)
    df['cos_month'] = df['circle_month'].map(np.cos) * scale
    df['sin_month'] = df['circle_month'].map(np.sin) * scale

    return(df)


def get_df(n_files, time_start, time_end, db_offset, all_sensors, tvb_classes=['C_300', 'C_0', 'C_112'], sensors=["p0640", "p0310", "p0720"], pred_path = "./cense_exp/", keep_only_tvb=True):
    if all_sensors:
        df_transcoder = pd.read_pickle(pred_path+'cense_lorient_transcoder_with_'+str(n_files)+'_files_dbcompensation_'+str(db_offset)+'_all_sensors_start_'+time_start+'_end_'+time_end)
        acoustic_path = pred_path+'cense_lorient_acoustic_with_'+str(n_files)+'_files_dbcompensation_'+str(db_offset)+'_all_sensors_start_'+time_start+'_end_'+time_end
        if os.path.exists(acoustic_path):
            df_acoustic = pd.read_pickle(acoustic_path)
        else:
            df_acoustic = None
        df_felix = None
        if sensors is not None:
            df_transcoder = df_transcoder[df_transcoder['id_sensor'].isin(sensors)]
            if df_acoustic is not None:
                df_acoustic = df_acoustic[df_acoustic['id_sensor'].isin(sensors)]
    else:
        sensors_str = '_'.join(sensors)
        df_transcoder = pd.read_pickle(pred_path+'cense_lorient_transcoder_with_'+str(n_files)+'_files_dbcompensation_'+str(db_offset)+'__' + sensors_str +'__'+'start_'+time_start+'_end_'+time_end)
        acoustic_path = pred_path+'cense_lorient_acoustic_with_'+str(n_files)+'_files_dbcompensation_'+str(db_offset)+'__' + sensors_str +'__'+'start_'+time_start+'_end_'+time_end
        if os.path.exists(acoustic_path):
            df_acoustic = pd.read_pickle()
        else:
            df_acoustic = None       
        felix_path = pred_path+'cense_lorient_felix_with_'+str(n_files)+'_files_dbcompensation_'+str(db_offset)+'__' + sensors_str +'__'+'start_'+time_start+'_end_'+time_end
        if os.path.exists(felix_path):
            df_felix = pd.read_pickle(felix_path)
        else:
            df_felix = None         

    if df_acoustic is not None:
        df_a = preprocess_df(df=df_acoustic,
                    t_class='laeq',
                    v_class='tfsd_mid',
                    b_class='tfsd_high',
                    t_coef=1/(94),
                    #I have a doubt on this one, wasn't it inversely correlated ?
                    v_coef=2.045388,
                    b_coef=0.8101505,
                    keep_only_tvb=keep_only_tvb)
    else:
        df_a = None
        
    if df_felix is None:
        df_f = None
    else:
        df_f = preprocess_df(df=df_felix,
                t_class='t',
                v_class='v',
                b_class='b',
                t_coef=1,
                v_coef=1,
                b_coef=1,
                keep_only_tvb=keep_only_tvb)
    
    if db_offset == -100:
        df = preprocess_df(df=df_transcoder,
                    t_class=tvb_classes[0],
                    v_class=tvb_classes[1],
                    b_class=tvb_classes[2],
                    t_coef=3.65089984,
                    v_coef=1.7112849,
                    b_coef=9.21451585,
                    keep_only_tvb=keep_only_tvb)

    if db_offset == -88:
        df = preprocess_df(df=df_transcoder,
                    t_class=tvb_classes[0],
                    v_class=tvb_classes[1],
                    b_class=tvb_classes[2],
                    # WITH LORIENT 1-K
                    # t_coef=3.65089984,
                    # v_coef=1.7112849,
                    # b_coef=9.21451585,
                    # WITH GRAFIC
                    # t_coef=2.73412322,
                    # v_coef=1.87321914,
                    # b_coef=5.14428502,
                    # WITH BOTH
                    # t_coef = 2.87730735,
                    # v_coef = 1.83711622,
                    # b_coef = 5.94695718,
                    # WITH LOGICAL CLASSES
                    t_coef = 10.36571881,
                    v_coef = 1.83711622,
                    b_coef = 5.94695718,
                    keep_only_tvb=keep_only_tvb)
    
    return(df_a, df_f, df)

class DFMethods:
    def __init__(self, tvb_classes):
        if tvb_classes is not None:
            self.t_class = tvb_classes[0]
            self.v_class = tvb_classes[1]
            self.b_class = tvb_classes[2]
        # Load the Excel workbook
        workbook = openpyxl.load_workbook('cense_lorient_coordinates_ID.xlsx')
        sheet = workbook.active
        self.data_dict = dict((key, tuple(map(float, value.split(',')))) for key, value in sheet.iter_rows(min_row=2, values_only=True))

    def determine_highest_class(self, row):
        if row[self.t_class] > row[self.v_class] and row[self.t_class] > row[self.b_class]:
            return 0
        elif row[self.v_class] > row[self.t_class] and row[self.v_class] > row[self.b_class]:
            return 1
        else:
            return 2

    # Define a function to map 'id_sensor' to coordinates
    def map_id_to_coordinates(self, id_sensor):
        out = self.data_dict.get(id_sensor, (None, None))
        return out  # Returns (None, None) if id_sensor not found


def get_df_for_map(df, tvb_classes, temporality=None):

    dfmet = DFMethods(tvb_classes)
    if temporality is None:
        df_map = df.copy()
    elif temporality == "early morning":
        df_map = df[(df['hour'] >= 5) & (df['hour'] <= 8)]
    elif temporality == "rush hour":
        df_map = df[(df['hour'] >= 17) & (df['hour'] <= 19)]
    elif temporality == "night life":
        df_map = df[(df['hour'] == 0) | (df['hour'] == 1) | (df['hour'] == 2) | (df['hour'] >= 22)]
    elif temporality == "night":
        df_map = df[(df['hour'] >= 3) & (df['hour'] <= 6)]
    
    df_map = df_map.groupby('id_sensor')[tvb_classes].mean().reset_index()
    df_map['highest_class'] = df_map.apply(dfmet.determine_highest_class, axis=1)
    df_map[['latitude', 'longitude']] = df_map['id_sensor'].map(dfmet.map_id_to_coordinates).apply(pd.Series)

    return(df_map)

def create_3source_map(df, column_names, title='blank', save_as_png=True):
    # Create a map centered at a specific location (you can change the coordinates)
    # tiles = 'CartoDB dark_matter'
    # tiles = 'Stamen Toner'
    tiles = 'cartodbpositron'
    m = folium.Map(location=[47.751809, -3.362845], zoom_start=15, tiles=tiles)

    # Get the minimum and maximum values of the selected column for scaling opacity

    column_names = [x for x in reversed(column_names)]
    colors = ['#8e8e8e', '#ae3620', 'green']
    # had to invert colors list for the plot
    colors = [x for x in reversed(colors)]

    # Iterate through the DataFrame and add markers to the map
    for index, row in df.iterrows():
        for idx, (column_name, color) in enumerate(zip(column_names, colors)):
            if idx == 2:
                radius = row[column_names[idx]]
                outer_color = color
            if idx == 1:
                radius = np.sqrt(row[column_names[idx]]**2 + row[column_names[idx+1]]**2)
                outer_color = color
            if idx == 0:
                radius = np.sqrt(row[column_names[idx]]**2 + row[column_names[idx+1]]**2 + row[column_names[idx+2]]**2)
                outer_color = 'black'

            radius = radius/3

            radius=radius*40

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=outer_color,
                fill=True,
                fill_color=color,
                fill_opacity=1,
                popup=f"{column_name}: {row[column_name]}, Highest Class: {row['highest_class']}",
                weight=0,
                overlay=True,
                z_index=idx
            ).add_to(m)

    pathtitle='./figures/' + title
    mapFname = pathtitle+'.html'
    m.save(mapFname)

    #saving as png by doing a screenshot
    if save_as_png:
        mapUrl = 'file://{0}/{1}'.format(os.getcwd(), mapFname)

        # Create a Chrome WebDriver instance
        options = Options()
        options.binary_location = '/usr/bin/google-chrome'  # Specify the path to your Chrome binary
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)
        driver.get(mapUrl)
        time.sleep(1)
        driver.save_screenshot(pathtitle + '.png')
        driver.quit()
        # Crop the image
        image = Image.open(pathtitle + '.png')
        left = 200  # Replace with your desired coordinates
        top = 40  # Replace with your desired coordinates
        right = 600 # Replace with your desired coordinates
        bottom = 550 # Replace with your desired coordinates
        image = image.crop((left, top, right, bottom))

        print('FIG: ' + pathtitle + '.png' + ' saved')
        # Save the cropped image
        image.save(pathtitle + '.png')

def clock_plot_multi(dfs, pann_classes=['C_300'], sensors=None, descs=['music festival', '2 weeks before music festival'], save=True, plot_desc='', timezone_adjustment=0, colors=[[0.7, 0.7, 0.2]], time='hour'):

    if time == 'hour':
        ticklabels = [f'{i}h' if i in [0, 6, 12, 18] else '' for i in range(24) ]
    if time == 'strong_minute':
        ticklabels = [f'{int(i/60)}h' if i in [60*k for k in range(24)] else '' for i in range(1440) ]
    if time == 'minute':
        ticklabels = [f'{i}min' if i in [10*k for k in range(6)] else '' for i in range(60) ]

    lim=(0, 1)
    fontname = 'Times New Roman'
    fontsize = 16
    # Create a grid of subplots
    fig, axes = plt.subplots(1, len(dfs), figsize=(len(dfs)*4, 4), subplot_kw={'projection': 'polar'})
    fig.subplots_adjust(wspace=0.3)  # Adjust vertical spacing between subplots

    for idx1, (df, desc) in enumerate(zip(dfs, descs)):
        cur_df = df.copy()
        df_mean = hourly_means(cur_df, sensor=sensors, classes=pann_classes, time=time, timezone_adjustment=timezone_adjustment)
        title =  desc

        len_df_mean = len(df_mean.index)
        angles = df_mean.index / (len_df_mean/(2*np.pi))
        val = df_mean.to_numpy()

        ax = axes[idx1]

        # Duplicate value at 23h and add it at the beginning
        val = np.concatenate((val[-1:], val))

        # Add an additional angle for 23h + full circle
        angles = np.concatenate((angles[-1:] + 2*np.pi, angles))
        
        if len(val.shape) == 2:
            n_val = val.shape[1]
        else:
            n_val = 1

        for k in range(n_val):
            x = angles
            y = val[:, k]

            ax.plot(np.append(x, x[0]), np.append(y, y[0]), linestyle='-', color=colors[k], alpha=0.5)
            ax.fill_between(x, y, color=colors[k], alpha=0.1)

        # delete the radial labels
        plt.setp(ax.get_yticklabels(), visible=False)

        # set the circumference labels
        if time == 'strong_minute':
            ax.set_xticks(np.linspace(0, 2 * np.pi, len_df_mean, endpoint=False)[::60])
            ax.set_xticklabels(ticklabels[::60], fontname=fontname, fontsize=int(fontsize / 1.3))
        else:
            ax.set_xticks(np.linspace(0, 2*pi, len_df_mean, endpoint=False))
            ax.set_xticklabels(ticklabels, fontname=fontname, fontsize=int(fontsize/1.3))

        # make the labels go clockwise
        ax.set_theta_direction(-1)

        # place 0 at the top
        ax.set_theta_offset(np.pi/2.0)    

        #set title
        ax.set_title(title, fontname=fontname, fontsize=fontsize, y=1.15)
        ax.set_ylim(lim)
        # put the points on the circumference

    plt.ylim(lim[0],lim[1])
    fig_to_save = plt.gcf()

    if save:
        print('FIG: ' + './figures/'+plot_desc+'_clock_graphs.png' + ' saved')
        fig_to_save.savefig('./figures/'+plot_desc+'_clock_graphs.pdf')
        fig_to_save.savefig('./figures/'+plot_desc+'_clock_graphs.png')
    else:
        plt.show()


def hourly_means(df, sensor=None, classes=['C_300', 'C_0', 'C_112'], time="hour", timezone_adjustment=0):
    df_copy = df.copy()
    df_copy['date'] = df_copy['date'] + timedelta(hours=timezone_adjustment)  # Add 1 hour to convert from UTC+1 to UTC+2
    df_copy['hour'] = df_copy['date'].dt.hour
    if time == 'strong_minute':
        df_copy['minute'] = df_copy['date'].dt.minute
        df_copy['strong_minute'] = df_copy['hour']*60 + df_copy['minute']
    if sensor is None:
        if time is not None:
            hourly_means = df_copy.groupby(time)[classes].mean()
        else:
            hourly_means = df_copy[classes]
    else:
        if time is not None:
            hourly_means = df_copy[df_copy["id_sensor"] == sensor].groupby(time)[classes].mean()
        else:
            hourly_means = df_copy[df_copy["id_sensor"] == sensor][classes]

    if time == 'strong_minute':
        index = list(range(0, 1440))  # Create a range from 0 to 1439

        # Find missing indices
        missing_indices = set(range(0, 1440)).difference(df.index)
        # Create a new DataFrame with missing indices filled with 0
        hourly_means = hourly_means.reindex(index=range(0, 1440), fill_value=0)

    return(hourly_means)


def main(config):
    if config.desc == None:
        return None
    if config.desc == "winter2020-3s":
        pd.options.display.max_colwidth = 200

        t_class='C_327'
        # t_class='C_300'
        v_class='C_0'
        # b_class='C_111'
        b_class='C_112'

        #--> only 3 sensors between january and february 2020, with 200 1min samples per day
        n_files = 32312
        time_start = '202011'
        time_end = '202031'
        # db_offset = -94+40
        db_offset = -88
        all_sensors = False

        # sensors ids
        # "p0720" : in north east, residential 
        # "p0450" : in pedestrian street
        # "p0310" : in pedestrian street
        # "p0640" : in huge boulevard
        # "p0160" : in huge boulevard
        sensor_res = "p0720"
        sensor_ped = "p0310"
        sensor_traf = "p0160"

        df_a_3s, df_f_3s, df_3s = get_df(n_files, time_start, time_end, db_offset, all_sensors, tvb_classes=[t_class, v_class, b_class], sensors=[sensor_res, sensor_ped, sensor_traf], pred_path=config.pred_path)

        df_3s_traf = df_3s[df_3s['id_sensor'] == sensor_traf]
        df_3s_ped = df_3s[df_3s['id_sensor'] == sensor_ped]
        df_3s_res = df_3s[df_3s['id_sensor'] == sensor_res]

        clock_plot_multi([df_3s_traf, df_3s_ped, df_3s_res], pann_classes=[t_class, v_class, b_class], sensors=None, descs=['Traffic street', 'Pedestrian street', 'Residential area'], save=True, plot_desc=config.desc, timezone_adjustment=0, colors=[[0.24, 0.16, 0.12], [0.60, 0.28, 0.10], [0.36, 0.73, 0.33]])

    if config.desc == "winter2020":
        pd.options.display.max_colwidth = 200

        t_class='C_327'
        # t_class='C_300'
        v_class='C_0'
        # b_class='C_111'
        b_class='C_112'

        #--> all available sensors between january and february 2020, with 10 1min samples per day
        n_files = 33443
        time_start = '202011'
        time_end = '202031'
        # db_offset = -94+40
        # db_offset = -94+20
        db_offset = -88
        all_sensors = True

        df_a, df_f, df = get_df(n_files, time_start, time_end, db_offset, all_sensors, tvb_classes=[t_class, v_class, b_class], sensors=None, pred_path=config.pred_path)

        df_map_general = get_df_for_map(df=df, tvb_classes=[t_class, v_class, b_class])
        df_map_nl = get_df_for_map(df=df, tvb_classes=[t_class, v_class, b_class], temporality="night life")
        df_map_early = get_df_for_map(df=df, tvb_classes=[t_class, v_class, b_class], temporality="early morning")
        df_map_rush = get_df_for_map(df=df, tvb_classes=[t_class, v_class, b_class], temporality="rush hour")

        create_3source_map(df_map_general, column_names=[t_class, v_class, b_class], title='map_PANN_general')
        create_3source_map(df_map_nl, column_names=[t_class, v_class, b_class], title='map_PANN_night_life')
        create_3source_map(df_map_early, column_names=[t_class, v_class, b_class], title='map_PANN_early_morning')
        create_3source_map(df_map_rush, column_names=[t_class, v_class, b_class], title='map_PANN_rush_hour')

    if config.desc == "music":
        pd.options.display.max_colwidth = 200

        t_class='C_300'
        v_class='C_0'
        # b_class = 'C_137'
        # b_class = 'C_326'
        b_class='C_112'

        pann_class = 'C_137'

        # sensors ids
        sensor_res = "p0720"
        sensor_ped = "p0310"
        sensor_traf = "p0160"

        # #--> only available sensors between january and february 2020, with 700 1min samples per day
        # n_files = 8234
        # time_start = '2021725'
        # time_end = '2021726'
        # # db_offset = -94+40
        # db_offset = -88
        # all_sensors = True

        #--> only available sensors between january and february 2020, with 700 1min samples per day
        # n_files = 29199
        # time_start = '202186'
        # time_end = '2021815'
        # n_files = 22660
        # time_start = '202071'
        # time_end = '202091'
        # # db_offset = -94+40
        # db_offset = -88
        # all_sensors = True

        ##############################
        #### july before interceltic festival
        n_files = 195968
        time_start = "202171"
        time_end = "202181"
        db_offset = -88
        all_sensors = True

        ####################
        ### fete de la musique - 21st of june
        # n_files = 4860
        # time_start = '2021621'
        # time_end = '2021622'
        # # db_offset = -94+40
        # db_offset = -88
        # all_sensors = True

        # _, _, df_bf = get_df(n_files, time_start, time_end, db_offset, all_sensors, tvb_classes=[t_class, v_class, b_class], sensors=None, keep_only_tvb=False)
        _, _, df_bf = get_df(n_files, time_start, time_end, db_offset, all_sensors, tvb_classes=[t_class, v_class, b_class], sensors=["p0220", "p0240", "p0200", "p0010", "p0100"], keep_only_tvb=False, pred_path=config.pred_path)
        df_bf = df_bf[df_bf["day_of_week"]=="Sunday"]
        df_bf[pann_class] = df_bf[pann_class] * 5

        #--> only available sensors between january and february 2020, with 700 1min samples per day
        # n_files = 29199
        # time_start = '202186'
        # time_end = '2021815'
        n_files = 6675
        time_start = '202188'
        time_end = '202189'
        # db_offset = -94+40
        db_offset = -88
        all_sensors = True

        _, _, df_f = get_df(n_files, time_start, time_end, db_offset, all_sensors, tvb_classes=[t_class, v_class, b_class], sensors=["p0220", "p0240", "p0200", "p0010", "p0100"], keep_only_tvb=False, pred_path=config.pred_path)
        df_f = df_f[df_f["day_of_week"]=="Sunday"]
        df_f[pann_class] = df_f[pann_class] * 5

        clock_plot_multi([df_bf, df_f], pann_classes=[pann_class], sensors=None, descs=['Sunday - Regular Summer Day', 'Sunday - Music Festival'], save=True, plot_desc='music', timezone_adjustment=0, colors=[[0.75, 0.75, 0.0]])
        # clock_plot_1s_multi([df_f, df_bf], pann_classes=[pann_class, 'C_0'], sensors=None, descs=['Summer - music festival', 'Sunday - regular day'], save=True, plot_desc='music', timezone_adjustment="UTC+1", colors=[[0.75, 0.75, 0.0], [0.5, 0.1, 0.1]])

    if config.desc == "church":
        pd.options.display.max_colwidth = 200

        t_class='C_300'
        v_class='C_0'
        # b_class = 'C_137'
        # b_class = 'C_326'
        b_class='C_112'

        #church bell
        pann_class = 'C_201'
        # flute
        # pann_class = 'C_196'
        # music
        # pann_class = 'C_137'

        # sensors ids
        sensor_res = "p0720"
        sensor_ped = "p0310"
        sensor_traf = "p0160"

        # n_files = 33443
        # time_start = '202011'
        # time_end = '202031'
        # # db_offset = -94+40
        # db_offset = -88
        # all_sensors = True

        #--> only 3 sensors between january and february 2020, with 100 1min samples per day
        # n_files = 1432
        # time_start = '20191225'
        # time_end = '20191226'
        # # db_offset = -94+40
        # db_offset = -88
        # all_sensors = True

        #on a time period where it is used
        n_files = 36195
        time_start = '202011'
        time_end = '202021'
        # db_offset = -94+40
        db_offset = -88
        all_sensors = False

        df_a, df_f, df = get_df(n_files, time_start, time_end, db_offset, all_sensors, tvb_classes=[t_class, v_class, b_class], sensors=["p0480"], keep_only_tvb=False, pred_path=config.pred_path)

        df = df[df['id_sensor'] == 'p0480']
        df[pann_class] = df[pann_class] * 75

        #on a time period where it is not used (octobre to november 2020 --> https://www.ouest-france.fr/bretagne/lorient-56100/lorient-muettes-les-cloches-de-saint-louis-ont-le-bourdon-7050817)
        n_files = 16456
        time_start = '2020101'
        time_end = '2020111'
        # db_offset = -94+40
        db_offset = -88
        all_sensors = False

        _, _, df_nu = get_df(n_files, time_start, time_end, db_offset, all_sensors, tvb_classes=[t_class, v_class, b_class], sensors=["p0480"], keep_only_tvb=False, pred_path=config.pred_path)

        df_nu = df_nu[df_nu['id_sensor'] == 'p0480']
        df_nu[pann_class] = df_nu[pann_class] * 75

        clock_plot_multi([df_nu, df], pann_classes=[pann_class], sensors=None, descs=['Church bells non-operational', 'Church bells operational'], save=True, plot_desc=config.desc, timezone_adjustment=0, colors=[[0.65, 0., 0.82]], time='strong_minute')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate 1s Mels and Third-Octave spectrograms')

    parser.add_argument('--pred_path', type=str, default="./cense_exp/predictions/",
                        help='The path where the h5 files of Cense Lorient are stored')
    parser.add_argument('--desc', type=str, default="winter2020",
                        help='The type of plot for which the data is retrieved ("winter2020", "winter2020-3s", "music_festival", "no_music_festival", "church_functional", "church_not_functional")')
    config = parser.parse_args()

    main(config)
