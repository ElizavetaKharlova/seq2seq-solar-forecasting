# K, this will be loading the data for the Generator paper

from Losses_and_Metrics import __calculatate_skillscore_baseline
from sqlalchemy import create_engine
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
def assemble_weather_info(profile_name):
    # we wanna read the folder to see all the csvs for the data
    # then we wanna start assemling for every csv
    nwp_folder = os.path.join(os.getcwd(), 'NWP_data')
    if profile_name in os.listdir(nwp_folder):
        profile_folder_path = os.path.join(nwp_folder, profile_name)
        nwp_csv_paths = find_all_files_of_type_in_folder(profile_folder_path, '.csv')

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

    # we wanna read a single csv into a pandas dataframe,
    # and then change the time value to the corresponding timestamp
    if "index" in nwp_pd.columns:
        nwp_pd = nwp_pd.drop(columns=['index'])
    if 'Latitude' in nwp_pd.columns:
        nwp_pd = nwp_pd.drop(columns=['Latitude'])
    if 'Longitude' in nwp_pd.columns:
        nwp_pd = nwp_pd.drop(columns=['Longitude'])
    return nwp_pd

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
def split_into_sets(nwp_df):

    train_nwp_df = nwp_df[nwp_df['Time'] >= min(nwp_df['Time']) + 2 * 365 * 24 * 60 * 60]
    test_val_nwp_df = nwp_df[nwp_df['Time'] < min(nwp_df['Time']) + 2 * 365 * 24 * 60 * 60]
    test_nwp_df = test_val_nwp_df.iloc[0:int(test_val_nwp_df.shape[0] / 2)]
    val_nwp_df = test_val_nwp_df.iloc[int(test_val_nwp_df.shape[0] / 2):]

    return train_nwp_df, val_nwp_df, test_nwp_df

# We'll chunk it up in samples, do a data blabla
def split_into_and_save_samples(nwp_df, sliding_window=7*24*60*60, bins=50, set_name='TrialSet', subset_type='train'):
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

    valid_timestamps = nwp_df[nwp_df['Time']< (max(nwp_df['Time'])-sliding_window)]
    valid_timestamps = valid_timestamps['Time'].to_list()
    progress_norm = len(valid_timestamps)
    counter = 0
    targets = []
    persistence = []

    prev_modulo = np.inf
    baseline_dict = {}
    for start_tstamp in valid_timestamps:
        #we know this tstamp exists here, so we should be able to just index, no?
        nwp_sample_df = nwp_df[nwp_df['Time']>=start_tstamp]
        nwp_sample_df = nwp_sample_df[start_tstamp+sliding_window > nwp_sample_df['Time']]

        # ToDo: this is a dumb hotfix that makes sure that the nwp slices have the proper length, it should not be necessary I think
        if nwp_sample_df.shape[0] == 672:
            nwp_sample_df = nwp_sample_df.drop('Time', axis=1)


            # profile_sample_df = profile_df[profile_df['tstamp']>=start_tstamp]
            # profile_sample_df = profile_sample_df[profile_sample_df['tstamp'] < start_tstamp + sliding_window]
            # profile_sample_df = profile_sample_df.drop('tstamp', axis=1)
            #
            # raw_profile_sample_np = profile_sample_df.to_numpy()
            # profile_sample_pdf_np = process_history_sample(raw_profile_sample_np, bins=bins)
            #
            # targets.append(profile_sample_pdf_np[-24:,:].tolist())
            # persistence.append(profile_sample_pdf_np[-48:-24,:].tolist())
            #
            # # ToDo: why do we get 0 error scores now again?!?
            # if len(targets) >= 256:
            #     batch_dict = __calculatate_skillscore_baseline(targets, persistence)
            #     for key in batch_dict:
            #         if key not in baseline_dict:
            #             baseline_dict[key] = [batch_dict[key]]
            #         else:
            #             baseline_dict[key].append(batch_dict[key])
            #     targets = []
            #     persistence = []
            #
            # with tf.io.TFRecordWriter(str(start_tstamp) + '.tfrecord') as writer:
            #     example = __convert_to_tf_example(nwp_sample_df.to_numpy(), profile_sample_pdf_np, raw_profile_sample_np)
            #     writer.write(example.SerializeToString())

        if (10*counter)%progress_norm < prev_modulo:
            print('Progress at ', counter/progress_norm)
        prev_modulo = (10*counter)%progress_norm
        counter = counter + 1
    # for key in baseline_dict:
    #     baseline_dict[key] = np.mean(baseline_dict[key])
    # #mnake sure we do not fuck thing up, so revert to previous wkdir
    # print(subset_type, 'baselines', baseline_dict)
    os.chdir(current_wd)
    return baseline_dict, {'support_shape': nwp_sample_df.to_numpy().shape,
                           #'pdf_history_shape': profile_sample_pdf_np.shape,
                           #'raw_history_shape': raw_profile_sample_np.shape
                            }

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
                # 'raw_history': __convert_to_feature(history),
                'pdf_history': __convert_to_feature(historical_pdf)
                }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

def __convert_to_feature(np_data):
    list_flat = np_data.flatten().tolist()
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_flat))

def save_sample_into_tf_ds(sample, set_name='trialset', subset_type='train'):
    pass


target_profile_name = 'egauge2474'

nwp_df = assemble_weather_info(target_profile_name) #reads from csv and assembles in one df with reset indices

nwp_df = scale_dataframe(nwp_df, target_columns=nwp_df.columns[1:]) # We do not want to normalize the time


train_nwp_df, val_nwp_df, test_nwp_df = split_into_sets(nwp_df)
del  nwp_df

dataset_info = {}
dataset_info['fc_tiles'] = 50
dataset_info['fc_steps'] = 24
set_name = 'PVHouse1'
dataset_info['test_baseline'], shape_dict = split_into_and_save_samples(
                                             nwp_df=test_nwp_df,
                                             sliding_window=7*24*60*60,
                                             bins=dataset_info['fc_tiles'],
                                             set_name=set_name,
                                             subset_type='test')
# print('test shapes', shape_dict)
# dataset_info['val_baseline'], shape_dict= split_into_and_save_samples(
#                                              nwp_df=val_nwp_df,
#                                              sliding_window=7*24*60*60,
#                                              bins=dataset_info['fc_tiles'],
#                                              set_name=set_name,
#                                              subset_type='vaidation')
# print('val shapes', shape_dict)
#
# dataset_info['train_baseline'], shape_dict = split_into_and_save_samples(
#                                              nwp_df=train_nwp_df,
#                                              sliding_window=7*24*60*60,
#                                              bins=dataset_info['fc_tiles'],
#                                              set_name=set_name,
#                                              subset_type='train')
# print('train shapes', shape_dict)
# for key in shape_dict:
#     dataset_info[key] = shape_dict[key]
# os.chdir(os.getcwd() + '/' + set_name)
# with open('dataset_info' + '.pickle', 'wb') as handle:
#     pickle.dump(dataset_info, handle, protocol=pickle.HIGHEST_PROTOCOL)




