# K, this will be loading the data for the Generator paper

from Losses_and_Metrics import __calculatate_skillscore_baseline
from sqlalchemy import create_engine
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from bokeh.plotting import figure, output_file, show
import bokeh
import pandas as pd
import shutil
import time
import datetime
from pandas.core.tools.datetimes import _guess_datetime_format_for_array
import numpy as np
import tensorflow as tf
import os
#-----------------------------------------------------------------------------------------------------------------------
# create list of profile names
def list_of_profile_names(engine):
    find_profile_names = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
    """
    prof_table = pd.read_sql_query(find_profile_names, engine)
    prof_list_of_list = prof_table.values.tolist()
    prof_list = []
    for elem in prof_list_of_list:
        prof_list.extend(elem)
    return prof_list
#-----------------------------------------------------------------------------------------------------------------------
# read database

    # merged_df = data_pd.merge(tstamp_pd, data_pd, left_index=True, right_index=True)
    #
    # return merged_df
#-----------------------------------------------------------------------------------------------------------------------
# We'll need access to the DB to read a specific profile's solar+ and grid
def extract_data_ts_from_db(profile_name, data_type, engine):
    data_query = " SELECT " + data_type + " FROM " + profile_name
    data_pd = pd.read_sql_query(data_query, engine)
    if 'column' in data_pd.columns[0]:
        data_pd.rename(columns={data_pd.columns[0]: 'grid'}, inplace=True)

    tstamp_query = " SELECT " + 'tstamp' + " FROM " + profile_name
    tstamp_pd = pd.read_sql_query(tstamp_query, engine)
    return data_pd.join(tstamp_pd)


def find_all_files_of_type_in_folder(folder_path, datatype):
    file_list = []
    for file in os.listdir(folder_path):
        if file.endswith(datatype):
            file_list.append(os.path.join(folder_path, file))
    return file_list

def extract_datetime_format(datetime_string):
    # its prolly '%Y-%m-%d %H:%M'
    from pandas.core.tools.datetimes import _guess_datetime_format_for_array
    datetime_np = np.array([datetime_string])
    datetime_format = _guess_datetime_format_for_array(datetime_np)
    return datetime_format

# We'll need to load the weather data
def assemble_weather_info(profile_name, location):
    # we wanna read the folder to see all the csvs for the data
    # then we wanna start assemling for every csv
    nwp_folder = os.path.join(os.getcwd(), 'NWP_data')
    nwp_folder = os.path.join(nwp_folder, location)
    if profile_name in os.listdir(nwp_folder):
        profile_folder_path = os.path.join(nwp_folder, profile_name)
        nwp_csv_paths = find_all_files_of_type_in_folder(profile_folder_path, '.csv')
        print('found nwp data')
        nwp_pd = []
        for nwp_csv_path in nwp_csv_paths:
            single_csv = read_and_process_single_csv(nwp_csv_path)
            nwp_pd.append(single_csv)
        nwp_pd = pd.concat(nwp_pd,  ignore_index=True)
                # merge the two together

        return nwp_pd
    else:
        print('didnt find the profile you wanted to use in the NWP_data folder')

def read_and_process_single_csv(file_path):
    nwp_pd = pd.read_csv(file_path)

    drop_indices = nwp_pd[nwp_pd['Time']=='Time'].index #catch if Nigel fucked up merging still
    nwp_pd = nwp_pd.drop(drop_indices)
    nwp_pd = nwp_pd.reset_index()

    for index in range(nwp_pd.shape[0]): #for some reason its doing dumb instable shit without the -1 ...
        time_string = nwp_pd['Time'][index]
        # optionally guess datetime_format here once
        time_tuple = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M').timetuple()
        time_stamp = time.mktime(time_tuple)
        nwp_pd.at[index, 'Time'] = int(time_stamp)

    return nwp_pd

# We'll need to synchronize this shit with the datetime arrays
def crop_dataframes_accordingly(nwp_df, profile_df):
    # Find the inner start index for both arrays and crop
    common_start_ts = max(nwp_df.iloc[0]['Time'], profile_df.iloc[0]['tstamp'])
    profile_df = profile_df[profile_df['tstamp'] >= common_start_ts]
    profile_df.reset_index()

    nwp_df = nwp_df[nwp_df['Time'] >= common_start_ts]
    nwp_df.reset_index()
    print('Cropped to the same starting timestamp at: ', common_start_ts)
    print('Nwp dataframe shape is ', nwp_df.shape, ' starting at ts', nwp_df.iloc[0]['Time'], 'going to ',
          nwp_df.iloc[-1]['Time'])
    print('profile dataframe shape is ', profile_df.shape, ' starting at ts', profile_df.iloc[0]['tstamp'], 'going to ',
          profile_df.iloc[-1]['tstamp'])
    # find the last inner index for both, but remember that we need to adjust their time resolutions ...
    # basically we expect the NWP to be sth like this:                   *   *   *   *   *   *
    # So the historical shit will have to be cropped like this:          ************************
    # so the last profile point has to be 1 profile_sample point before the potential next NWP sample point in time
    nwp_last_ts = nwp_df.iloc[-1]['Time']
    profile_last_ts = profile_df.iloc[-1]['tstamp']

    profile_delta_ts = profile_df.iloc[1]['tstamp'] - profile_df.iloc[0]['tstamp']
    nwp_delta_ts = nwp_df.iloc[1]['Time'] - nwp_df.iloc[0]['Time']
    necessary_offset = nwp_delta_ts - profile_delta_ts

    # this assumes that the resolution and timespots of shots are synchronized!
    # ToDo: maybe fix at some point
    if profile_last_ts - necessary_offset >= nwp_last_ts:  # we have more historical data than NWP data
        # basically we expect the NWP to be sth like this:                   *   *   *   *   *   ^
        # So the historical shit will have to be cropped like this:          ************************^*******************
        profile_crop_ts = nwp_last_ts + necessary_offset
        nwp_crop_ts = nwp_last_ts
    else:
        # last_profile_ts - necessary_offset < last_nwp_ts:
        # basically we expect the NWP to be sth like this:                   *   *   ^   *   *   *
        # So the historical shit will have to be cropped like this:          ***********^**
        nwp_crop_steps = (nwp_last_ts - (profile_last_ts - necessary_offset)) / nwp_delta_ts
        nwp_crop_steps = np.ceil(nwp_crop_steps)
        nwp_crop_ts = nwp_last_ts - nwp_crop_steps * nwp_delta_ts
        profile_crop_ts = nwp_crop_ts + necessary_offset

    profile_df = profile_df[profile_df['tstamp'] <= profile_crop_ts]
    profile_df.reset_index()
    nwp_df = nwp_df[nwp_df['Time'] <= nwp_crop_ts]
    nwp_df.reset_index()
    print('Cropped to the adequate end timestamps')
    print('Nwp dataframe shape is ', nwp_df.shape, ' starting at ts', nwp_df.iloc[0]['Time'], 'going to ',
          nwp_df.iloc[-1]['Time'])
    print('profile dataframe shape is ', profile_df.shape, ' starting at ts', profile_df.iloc[0]['tstamp'], 'going to ',
          profile_df.iloc[-1]['tstamp'])
    return nwp_df, profile_df

def scale_dataframe(df, target_columns=None):
    scaler = StandardScaler()

    if target_columns is None:
        data = df[df.columns]
        if len(data.shape) == 1:
            data = [data]
            data = scaler.fit_transform(data)
            df[df.columns] = data[0]

        else:
            df[df.columns] = scaler.fit_transform(data)
    else:
        data = df[target_columns]
        if len(data.shape) == 1:
            data = [data]
            data = scaler.fit_transform(data)
            df[target_columns] = data[0]

        else:
            df[target_columns] = scaler.fit_transform(data)

    return df

# We'll split it into datasets
#ToDo: Maybe split less than 1y for test and val
def split_into_sets(profile_df, nwp_df):
    if max(profile_df['tstamp']) - min(profile_df['tstamp']) < 5.0:
        print('NOT FORSEEN< WE HAVE LESS THAN 5Y data, ABOPOOORT')
    else:
        train_profile_df = profile_df[profile_df['tstamp'] >= min(profile_df['tstamp']) + 2 * 365 * 24 * 60 * 60]
        test_val_profile_df = profile_df[profile_df['tstamp'] < min(profile_df['tstamp']) + 2 * 365 * 24 * 60 * 60]
        test_profile_df = test_val_profile_df.iloc[0:int(test_val_profile_df.shape[0] / 2)]
        val_profile_df = test_val_profile_df.iloc[int(test_val_profile_df.shape[0] / 2):]

        train_nwp_df = nwp_df[nwp_df['Time'] >= min(nwp_df['Time']) + 2 * 365 * 24 * 60 * 60]
        test_val_nwp_df = nwp_df[nwp_df['Time'] < min(nwp_df['Time']) + 2 * 365 * 24 * 60 * 60]
        test_nwp_df = test_val_nwp_df.iloc[0:int(test_val_nwp_df.shape[0] / 2)]
        val_nwp_df = test_val_nwp_df.iloc[int(test_val_nwp_df.shape[0] / 2):]

    return train_profile_df, val_profile_df, test_profile_df, train_nwp_df, val_nwp_df, test_nwp_df

# We'll chunk it up in samples, do a data blabla
def split_into_and_save_samples(profile_df, nwp_df, sliding_window=7*24*60*60, bins=50, set_name='TrialSet', subset_type='train', intersample_factor=None):
    # for each timestep in the nwp_df, crop the sliding window
    # find the corresponding start in the profile_df +24*60*60 ahead and crop the corresponding window

    # make the dataset directory
    current_wd = os.getcwd()
    if not os.path.isdir(set_name):
        os.mkdir(set_name)
        print('creating folder for', set_name)
    else:
        print('found preexisting folder for', set_name)
    dataset_folder_path = current_wd + '/' + set_name
    os.chdir(dataset_folder_path)

    if os.path.isdir(subset_type):
        print('deleting previous dataset', subset_type)
        shutil.rmtree(subset_type, ignore_errors=True)
    os.mkdir(subset_type)  # create a folder, move to the folder
    os.chdir((dataset_folder_path + '/' + subset_type))

    nwp_df = downsample_df(nwp_df, factor=2)  # ToDo: for now because Nigel's fault
    if intersample_factor is not None:
        nwp_df = interpolate_weather(nwp_df, factor=intersample_factor)

    valid_timestamps = nwp_df[nwp_df['Time']< (max(nwp_df['Time'])-sliding_window)]
    valid_timestamps = valid_timestamps['Time'].to_list()
    progress_norm = len(valid_timestamps)
    counter = 0
    targets = []
    persistence = []

    prev_modulo = np.inf
    baseline_dict = {}
    #ToDo: maybe do not take every 5 min sample of the nwp but instead every 15min sample of the nwp?
    for start_tstamp in valid_timestamps:

        #This query takes from a dataframe: all the entries with a timestamp larger than the start timestamp
        # and then discards all the entries with a timestamp larger than start+sw
        nwp_sample_df = nwp_df[nwp_df['Time']>=start_tstamp]
        nwp_sample_df = nwp_sample_df[start_tstamp+sliding_window > nwp_sample_df['Time']]
        # downsample back to 15 min
        if intersample_factor is not None:
            nwp_sample_df = downsample_df(nwp_sample_df, factor=intersample_factor) #ToDo: For now because Nigel

        # ToDo: those lenghts should be a function fo the sliding window size if we wanna be fancy
        if nwp_sample_df.shape[0] == sliding_window/(2*15*60):
            nwp_sample_df = nwp_sample_df.drop('Time', axis=1)

            profile_sample_df = profile_df[profile_df['tstamp']>=start_tstamp]
            profile_sample_df = profile_sample_df[profile_sample_df['tstamp'] < start_tstamp + sliding_window]
            profile_sample_df = profile_sample_df.drop('tstamp', axis=1)

            # ToDo: those lenghts should be a function fo the sliding window size if we wanna be fancy
            if profile_sample_df.shape[0] == sliding_window/60:
                raw_profile_sample_np = profile_sample_df.to_numpy()
                profile_sample_pdf_np = process_history_sample(raw_profile_sample_np, bins=bins)

                targets.append(profile_sample_pdf_np[-24:,:].tolist())
                persistence.append(profile_sample_pdf_np[-48:-24,:].tolist())

                # calculates persistence
                if len(targets) >= 128:
                    batch_dict = __calculatate_skillscore_baseline(targets, persistence)
                    for key in batch_dict:
                        if key not in baseline_dict:
                            baseline_dict[key] = [batch_dict[key]]
                        else:
                            baseline_dict[key].append(batch_dict[key])
                    targets = []
                    persistence = []

                with tf.io.TFRecordWriter(str(start_tstamp) + '.tfrecord') as writer:
                    example = __convert_to_tf_example(nwp_sample_df.to_numpy(), profile_sample_pdf_np, raw_profile_sample_np)
                    writer.write(example.SerializeToString())

        if (10*counter)%progress_norm < prev_modulo:
            print('Progress at ', counter/progress_norm)
        prev_modulo = (10*counter)%progress_norm
        counter = counter + 1

    # averages persistences
    for key in baseline_dict:
        baseline_dict[key] = np.mean(baseline_dict[key])
    #mnake sure we do not fuck thing up, so revert to previous wkdir
    print(subset_type, 'baselines', baseline_dict)
    os.chdir(current_wd)
    return baseline_dict, {'support_shape': nwp_sample_df.to_numpy().shape,
                           'pdf_history_shape': profile_sample_pdf_np.shape,
                           'raw_history_shape': raw_profile_sample_np.shape}

def calculate_baseline_scores(targets, persistence, split_size=512):
    if len(targets) > split_size:
        start_index = 0
        while start_index < len(targets):
            if start_index == 0:
                baseline_dict = __calculatate_skillscore_baseline(targets[start_index:(start_index+split_size)],
                                                                  persistence[start_index:(start_index+split_size)])
            else:
                batch_dict = __calculatate_skillscore_baseline(targets[start_index:(start_index+split_size)],
                                                                  persistence[start_index:(start_index+split_size)])

                for key in batch_dict:
                    baseline_dict[key].append(batch_dict[key])

            start_index += split_size

        for key in baseline_dict:
            baseline_dict[key] = np.mean(baseline_dict[key])

    else:
        baseline_dict = __calculatate_skillscore_baseline(targets, persistence)

    return baseline_dict
# We'll split the historical sample into a binned pdf
# ASSUMES HISTORY IS NORMALIZED BETWEEN 0 and 1 and everything is given as dataframe with a tstamp information
def process_history_sample(profile_sample, bins):
    profile_sample = np.floor(bins*profile_sample)  #this gives the indices, except where the sample was 1, we need to filter those

    pdf = np.zeros(shape=[int(profile_sample.shape[0] / 60), bins])
    for hour in range(int(profile_sample.shape[0] / 60)):
        hourly_slice = profile_sample[hour*60:(60*hour+60)]

        for minute in range(hourly_slice.shape[0]):
            index = hourly_slice[minute]
            pdf[hour, int(min(index, bins-1))] += 1/60
    return pdf

def __convert_to_tf_example(nwp_input, historical_pdf, history):

    features = {'support': __convert_to_feature(nwp_input),
                'raw_history': __convert_to_feature(history),
                'pdf_history': __convert_to_feature(historical_pdf)
                }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

def __convert_to_feature(np_data):
    list_flat = np_data.flatten().tolist()
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_flat))

def interpolate_weather(nwp_df, factor):
    nwp_shape = nwp_df.shape
    new_indices = np.arange(nwp_df.shape[0]*factor)
    nwp_indices = np.arange(0, nwp_df.shape[0]*factor, int(factor))
    columns = nwp_df.keys()
    nwp_interpolated = pd.DataFrame(columns=columns) 

    # interpolate
    for feature_axis in columns:
        features = np.asarray(nwp_df[feature_axis], dtype=np.float64)
        insert = np.interp(new_indices, nwp_indices, features)
        nwp_interpolated[feature_axis] = insert.tolist()
    del nwp_shape

    return nwp_interpolated

def add_sine(time_array, interval):
    # calculate sine and cosine waves for a given time array
    time_array = time_array%interval
    time_array = (time_array*2*np.pi)/interval
    time_array = time_array.to_numpy()
    time_array = time_array.astype(float)
    ar_sin = np.sin(time_array)
    ar_cos = np.cos(time_array)
    return ar_sin, ar_cos

def manage_features(nwp_df, target_data_type='solar+', location='Seattle'):

    #ToDO: remove once Nigel has new NWP with fixed timezone
    if location is 'Seattle':
        nwp_df['Time'] = nwp_df['Time'] - 7*60*60
    elif location is 'Edmonton':
        nwp_df['Time'] = nwp_df['Time'] - 8 * 60 * 60
    # general
    if "index" in nwp_df.columns:
        nwp_df = nwp_df.drop(columns=['index'])
    if 'Latitude' in nwp_df.columns:
        nwp_df = nwp_df.drop(columns=['Latitude'])
    if 'Longitude' in nwp_df.columns:
        nwp_df = nwp_df.drop(columns=['Longitude'])
    if 'Short_Wave_Flux_Up [W/m2]' in nwp_df.columns:
        nwp_df = nwp_df.drop(columns=['Short_Wave_Flux_Up [W/m2]'])
    if 'Long_Wave_Flux_Up [W/m2]' in nwp_df.columns:
        nwp_df = nwp_df.drop(columns=['Long_Wave_Flux_Up [W/m2]'])

    if 'solar' in target_data_type:
        year_sin, year_cos = add_sine(nwp_df['Time'], interval=365.25*24*60*60)
        nwp_df['Season_sin'] = year_sin

    elif 'grid' in target_data_type:

        year_sin, year_cos = add_sine(nwp_df['Time'], interval=365.25*24*60*60)
        nwp_df['Season_sin'] = year_sin

        week_sin, week_cos = add_sine(nwp_df['Time'], interval=7*24*60*60)
        nwp_df['Week_sin'] = week_sin
        nwp_df['Week_cos'] = week_cos

        if location == 'Seattle':
            holidays_list = add_holidays(nwp_df['Time'])
            nwp_df['Holidays'] = holidays_list

        if 'Surface_Pressure [Pa]' in nwp_df.columns:
            nwp_df = nwp_df.drop(columns=['Surface_Pressure [Pa]'])
        if 'X_Wind [m/s]' in nwp_df.columns:
            nwp_df = nwp_df.drop(columns=['X_Wind [m/s]'])
        if 'Y_Wind [m/s]' in nwp_df.columns:
            nwp_df = nwp_df.drop(columns=['Y_Wind [m/s]'])
        if 'Humidity [kg/kg]' in nwp_df.columns:
            nwp_df = nwp_df.drop(columns=['Humidity [kg/kg]'])
        if 'Short_Wave_Flux_Down [W/m2]' in nwp_df.columns:
            nwp_df = nwp_df.drop(columns=['Short_Wave_Flux_Down [W/m2]'])
        if 'Long_Wave_Flux_Down [W/m2]' in nwp_df.columns:
            nwp_df = nwp_df.drop(columns=['Long_Wave_Flux_Down [W/m2]'])
            #ToDo: add wind and others, fix the rest :-(

    return nwp_df

def downsample_df(_df, factor):
    columns = _df.keys()
    _df_downsampled = pd.DataFrame(columns=columns)
    for column in columns:
        col_array = np.asarray(_df[column])
        col_array = col_array[::factor]
        _df_downsampled[column] = col_array.tolist()
    return _df_downsampled

def add_holidays(time_array):
    import holidays
    #ToDo: DIsclaimer: only works for 15min resolutions!!

    # get the years from the dataset
    years = range(time.localtime(time_array[0]).tm_year, time.localtime(time_array[len(time_array)-1]).tm_year+1)
    # create new column 
    holidays_list = np.zeros([len(time_array),1], )
    interval = 24*4
    # go through the holidays in Washington state and add a ramp for each
    for date, name in sorted(holidays.US(state='WA', years=years).items()):
        holiday_tstamp = int(time.mktime(date.timetuple()))
        # check if the holiday date is in the dataframe 
        if time_array.isin([holiday_tstamp]).any().any():
            ind = time_array[time_array==holiday_tstamp].index[0]
            # assign ones to the holiday date
            holidays_list[ind:ind+interval] = np.ones([interval,1])
            # ramp up and down around the holiday 
            ramp_up = np.expand_dims(np.arange(0,1,(1/interval)), axis=-1)

            ramp_up = np.sin(ramp_up*np.pi/2)
            ramp_down = np.flip(ramp_up)

            if ind > 0:
                holidays_list[max(ind-interval, 0):ind] = ramp_up[-min(interval, ind):]
            if ind < len(time_array)-interval:
                holidays_list[ind+interval:ind+2*interval] = ramp_down
    return holidays_list

def __fix_history(profile_df, target_data_type, target_profile_name):


    if target_profile_name == 'egauge4183':
        target_data_array = profile_df[target_data_type].to_numpy()
        max_pos = np.where(target_data_array == np.amax(target_data_array))[0]
        lower_values = max_pos - (len(max_pos) + 1)
        print(lower_values)
        upper_values = max_pos + (len(max_pos) + 1)
        target_data_array[max_pos] = (target_data_array[lower_values] + target_data_array[upper_values])/2
        profile_df[target_data_type] = target_data_array

    plt.plot(profile_df[target_data_type])
    plt.show()
    return profile_df

# ToDo: wrap this into a function, so we can generate load/pv datasets for specific houses
def generate_dataset(target_data_type, target_profile_name, location='Seattle'):
    engine = create_engine('postgresql://postgres:postgres@stargate/profiles')
    prof_list = list_of_profile_names(engine)

    if target_profile_name not in prof_list:
        print('UhOh ... could not find the profile specified from the available profiles in the database')
    else:
        profile_df = extract_data_ts_from_db(target_profile_name, target_data_type, engine)
        target_data_type = profile_df.columns[0]

    nwp_df = assemble_weather_info(target_profile_name, location) #reads from csv and assembles in one df with reset indices

    nwp_df = manage_features(nwp_df, target_data_type=target_data_type, location=location)

    nwp_df, profile_df = crop_dataframes_accordingly(nwp_df, profile_df) #crops and resets indixes so shit doesnt get bad

    # downsampled_profile = downsample_df(profile_df, 15)
    # downsampled_profile = downsampled_profile['solar+'].to_numpy()
    # nwp_solar = nwp_df['Short_Wave_Flux_Down [W/m2]'].to_numpy()
    # min_len = min(nwp_solar.shape[0], downsampled_profile.shape[0])
    # corcoeffs = np.corrcoef(nwp_solar[:min_len], downsampled_profile[:min_len])
    # print(corcoeffs)

    profile_df = __fix_history(profile_df, target_data_type, target_profile_name)


    nwp_df = scale_dataframe(nwp_df, target_columns=nwp_df.columns[1:]) # We do not want to normalize the time

    profile_df[target_data_type] = profile_df[target_data_type] - min(profile_df[target_data_type]) #this will be normalized from 0 to 1, since we will only need the pdf anyways
    profile_df[target_data_type] = profile_df[target_data_type]/max(profile_df[target_data_type])

    train_profile_df, val_profile_df, test_profile_df, train_nwp_df, val_nwp_df, test_nwp_df = split_into_sets(profile_df, nwp_df)

    dataset_info = {}
    dataset_info['fc_tiles'] = 50
    dataset_info['fc_steps'] = 24
    set_name = target_profile_name + target_data_type
    dataset_info['test_baseline'], shape_dict = split_into_and_save_samples(profile_df=test_profile_df,
                                                                            nwp_df=test_nwp_df,
                                                                            sliding_window=7*24*60*60,
                                                                            bins=dataset_info['fc_tiles'],
                                                                            set_name=set_name,
                                                                            subset_type='test',
                                                                            intersample_factor=None)
    print('test shapes', shape_dict)
    dataset_info['val_baseline'], shape_dict= split_into_and_save_samples(profile_df=val_profile_df,
                                                                            nwp_df=val_nwp_df,
                                                                            sliding_window=7*24*60*60,
                                                                            bins=dataset_info['fc_tiles'],
                                                                            set_name=set_name,
                                                                            subset_type='validation',
                                                                            intersample_factor=None)
    print('val shapes', shape_dict)

    dataset_info['train_baseline'], shape_dict = split_into_and_save_samples(profile_df=train_profile_df,
                                                                            nwp_df=train_nwp_df,
                                                                            sliding_window=7*24*60*60,
                                                                            bins=dataset_info['fc_tiles'],
                                                                            set_name=set_name,
                                                                            subset_type='train',
                                                                            intersample_factor=5)
    print('train shapes', shape_dict)
    for key in shape_dict:
        dataset_info[key] = shape_dict[key]
    os.chdir(os.getcwd() + '/' + set_name)
    with open('dataset_info' + '.pickle', 'wb') as handle:
        pickle.dump(dataset_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Generate and save dataset for load/PV and house.
col_list = ['"solar+"','grid + "solar+"']
generate_dataset(target_data_type=col_list[1], target_profile_name = 'egauge2474')
#houses that are interesting:
# egauge4183, but not grid
# egauge22785
# egauge2474



