import os, datetime, re
import pandas as pd
import numpy as np
from utils import import_profile


def load_dataset(weather_data_folder='\_data\Weather_2016', pv_data_path="/_data/"):

    print('fetching NWP...')

    weather_mvts, start_end_date =_get_weather_data(weather_data_folder)
    print('fetched NWP ... now fetching PV Data')

    gen = _get_PV(pv_data_path, start_end_date)
    print('fetched PV ... now concatenating')

    generation = __stack_ts(gen, int(gen.shape[0] / weather_mvts.shape[0]))

    full_dataset = np.append(generation, weather_mvts, axis=-1)
    full_dataset = np.delete(full_dataset, np.s_[5], axis=-1)
    return full_dataset

def _get_weather_data(weather_data_folder):
    cwd = os.getcwd()

    path_to_dir = cwd + weather_data_folder

    files_csv = __find_files(path_to_dir, suffix=".csv")
    files_csv = __sorted_nicely(files_csv)

    read_csv = []
    for file_nbr in range(len(files_csv)):
        file = cwd + weather_data_folder + "/" + files_csv[file_nbr]
        buffer = pd.read_csv(file)
        read_csv.append(buffer)

    'making sure you have the right indices, even if simulations are not the same length and do not have the same sample sizes'
    steps_per_sim = np.zeros(len(read_csv))
    samples_to_discard = np.zeros(len(read_csv))

    buffer = read_csv[0].values
    start_nwp = buffer[0, 0]
    print('Simulations start at: ', buffer[0, 0])
    deltaT = datetime.datetime.strptime(buffer[1, 0], '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(buffer[0, 0], '%Y-%m-%d %H:%M:%S')
    print('Sample time of Sims is: ', deltaT)
    spinup = datetime.timedelta(hours=6)
    for sim in range(len(read_csv)):
        samples_to_discard[sim] = spinup/ deltaT
        steps_per_sim[sim] = len(read_csv[sim]) - int(samples_to_discard[sim])

    'printing some Infor'
    print('Sample time of simulation:', deltaT)

    if np.allclose(steps_per_sim, steps_per_sim[0]) != True:
        print("It seems like some simulations diverge in length, consider rechecking the data?")

    if np.allclose(samples_to_discard, samples_to_discard[0]) != True:
        print("Seems that some simulations have a diverging sample time...!!")

    'Saving it all to a numpy array'
    buffer = read_csv[0].values
    weather_mvts = buffer[int(samples_to_discard[0]):, :]
    start_nwp = weather_mvts[0, 0]
    weather_mvts[:,0] = [datetime.datetime.strptime(weather_mvts[i,0], '%Y-%m-%d %H:%M:%S') for i in range(len(weather_mvts[:,0]))]
    flags = 0*weather_mvts[:,-1]
    weather_mvts = np.append(weather_mvts, np.expand_dims(flags, axis=-1), axis=-1)
    end_prev_sim = weather_mvts[-1,0]

    for sim in range(1, len(read_csv)):
        buffer = read_csv[sim].values
        end_nwp = buffer[-1, 0]
        buffer[:, 0] = [datetime.datetime.strptime(buffer[i, 0],  '%Y-%m-%d %H:%M:%S') for i in range(len(buffer[:, 0]))]

        if (end_prev_sim + deltaT) not in buffer[:,0]:
            start_this_sim = buffer[0, 0]
            missing_samples = start_this_sim - end_prev_sim - deltaT
            missing_samples = missing_samples / deltaT
            missing_samples = int(missing_samples)
            print('missing ', missing_samples, ' samples from ', end_prev_sim, 'to ', start_this_sim)
            print('Setting flags for those datapoints to 1')
            insert = weather_mvts[-missing_samples:,:]

            for index in range(len(insert[:,0])):
                insert[index,0] = end_prev_sim + (index+1)*deltaT

            insert[:, 1:-1] = 0
            insert[:, -1] = 1

            print(buffer.shape, insert.shape)
            buffer = np.append(buffer, np.expand_dims(0*buffer[:,-1], axis=-1), axis=-1)
            buffer = np.append(insert, buffer, axis=0)


        else:

            buffer = np.append(buffer, np.expand_dims(0*buffer[:,-1], axis=-1), axis=-1)

        first_appropriate_index = np.where(buffer[:, 0] == (end_prev_sim + deltaT))
        start_this_sim = int(first_appropriate_index[0])
        weather_mvts = np.append(weather_mvts, buffer[(start_this_sim):, :],
                                            axis=0)
        end_prev_sim = weather_mvts[-1, 0]
        end_prev_sim = weather_mvts[-1,0]

    print('replacing the fauly values with the mean of the array')
    faults = weather_mvts[:,-1]
    for axis in range(1, weather_mvts.shape[-1] - 1): #for all but the last data array
        mean = np.mean(weather_mvts[:,axis])

        weather_mvts[:, axis] = np.where(faults == 1, mean, weather_mvts[:, axis])
    return weather_mvts, [start_nwp, end_nwp]

def _get_PV(path, start_end_date): #getting the PV data
    start_nwp = start_end_date[0]
    end_nwp = start_end_date[1]
    cwd = os.getcwd()
    profile_name = __find_files(cwd + path, suffix=".zp")
    profile_path = cwd + path + profile_name[0][:-3]

    shift = datetime.timedelta(hours=-7) #lolol, turns out Steven uses UTC while Thomas something else
    deltaT = datetime.timedelta(minutes=5)
    forecast_distance = datetime.timedelta(days=0)
    start = datetime.datetime.strptime(start_nwp,  '%Y-%m-%d %H:%M:%S') + forecast_distance + shift
    stop = datetime.datetime.strptime(end_nwp,  '%Y-%m-%d %H:%M:%S') + deltaT + forecast_distance + shift


    print('Loading generation data from ', start.strftime('%Y-%m-%d%H:%M:%S'), ' to ', stop.strftime('%Y-%m-%d%H:%M:%S'))
    profile = import_profile(profile_path, time_start=start.strftime('%Y-%m-%d%H:%M:%S'),
                             time_end=stop.strftime('%Y-%m-%d%H:%M:%S'))

    generation = profile['gen']['profile']
    generation = np.squeeze(generation)

    return generation

def __find_files(path_to_dir, suffix=".pickle"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def __sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def __stack_ts(timeseries, downsampling_factor):
    downsampling_factor = int(downsampling_factor)
    length = int(len(timeseries)/downsampling_factor)
    stacked_ts = np.zeros([length, downsampling_factor])

    for i in range(stacked_ts.shape[0]):
        for row in range(downsampling_factor):
            # print(timeseries[downsampling_factor*i+row])
            stacked_ts[i,row] = timeseries[downsampling_factor*i+row]
    return stacked_ts


