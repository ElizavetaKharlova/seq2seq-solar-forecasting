def get_Daniels_data(target_data='Pv',
                     input_len_samples=int(1 * 24 * (60 / 5)),
                     fc_len_samples=int(1 * 24 * (60 / 5)),
                     fc_steps=24,
                     fc_tiles=33):
    # Daniel loads his data / If we want different data we do that here
    # For a encoder decoder model we will need 3 inputs:
    # encoder input, decoder support, blend factor (for how much we want to do teacher forcing)
    raw_data = load_dataset()

    if target_data == 'PV' or target_data == 'pv' or target_data == 'Pv':
        target_dims = [0, 1, 2, 3, 4]
    elif target_data == 'Load' or target_data == 'load' or target_data == 'LOAD':
        target_dims = None
        print('Not Implemented YET, except funky behavior')
        # ToDo: make sure you can add the load data here!

    inp, ev_target, ev_support, pdf_target, pdf_support = datasets_from_data(raw_data,
                                                          sw_len_samples=input_len_samples,
                                                          fc_len_samples=fc_len_samples,
                                                          fc_steps=fc_steps,
                                                          fc_tiles=fc_tiles,
                                                          target_dims=target_dims,
                                                          plot=True,
                                                          steps_to_new_sample=15)
    import numpy as np
    ev_target = np.expand_dims(ev_target, axis=-1)
    ev_support = np.expand_dims(ev_support, axis=-1)

    inp_train = inp[:int(0.8 * inp.shape[0]), :, :]
    inp_test = inp[int(0.8 * inp.shape[0]):, :, :]

    print(pdf_target)
    pdf_train = pdf_target[:int(0.8 * inp.shape[0]), :, :]
    pdf_test = pdf_target[int(0.8 * inp.shape[0]):, :, :]
    pdf_teacher_train = pdf_support[:int(0.8 * inp.shape[0]), :, :]
    pdf_teacher_test = pdf_support[int(0.8 * inp.shape[0]):, :, :]

    ev_target_train = ev_target[:int(0.8 * inp.shape[0]), :, :]
    ev_target_test = ev_target[int(0.8 * inp.shape[0]):, :, :]
    ev_teacher_train = ev_support[:int(0.8 * inp.shape[0]), :, :]
    ev_teacher_test = ev_support[int(0.8 * inp.shape[0]):, :, :]

    blend_factor = np.expand_dims(np.ones(inp_train.shape[0]), axis=-1)
    print(blend_factor.shape)

    print('The training set has an input data shape of ',
          inp_train.shape,
          'to expected value targets of ',
          ev_target_train.shape,
          'or alternatively pdf_targets of ',
          pdf_train.shape)
    print('-----------------------------------------------')
    print('The testing set has an input data shape of ',
          inp_test.shape,
          'to expected value targets of ',
          pdf_test.shape,
          'or alternatively pdf_targets of ',
          ev_target_train.shape)

    return inp_train, inp_test, ev_target_train, ev_target_test, ev_teacher_train, ev_teacher_test, blend_factor, pdf_train, pdf_test, pdf_teacher_train, pdf_teacher_test


def get_Lizas_data():
    import gzip, pickle
    data_path = './data/Lizas_data.pklz'
    f = gzip.open(data_path, 'rb')
    NetworkData = pickle.load(f)
    f.close()

    mvts_array = NetworkData['dataset']
    specs = NetworkData['specs']
    print('Specs:', specs)

    inp, ev_target, ev_support, pdf_target, pdf_support = datasets_from_data(data=mvts_array,
                                      sw_len_samples=specs['sw_steps'],
                                      fc_len_samples=specs['num_steps'],
                                      fc_steps=specs['forecast_steps'],
                                      fc_tiles=specs['num_tiles'],
                                      target_dims=specs['target'],
                                      plot=False,
                                      steps_to_new_sample=1)

    # ---------------------------------------------------------------------------

    import numpy as np
    ev_target = np.expand_dims(ev_target, axis=-1)
    ev_support = np.expand_dims(ev_support, axis=-1)

    inp_train = inp[:int(0.8 * inp.shape[0]), :, :]
    inp_test = inp[int(0.8 * inp.shape[0]):, :, :]

    print(pdf_target)
    pdf_train = pdf_target[:int(0.8 * inp.shape[0]), :, :]
    pdf_test = pdf_target[int(0.8 * inp.shape[0]):, :, :]
    pdf_teacher_train = pdf_support[:int(0.8 * inp.shape[0]), :, :]
    pdf_teacher_test = pdf_support[int(0.8 * inp.shape[0]):, :, :]

    ev_target_train = ev_target[:int(0.8 * inp.shape[0]), :, :]
    ev_target_test = ev_target[int(0.8 * inp.shape[0]):, :, :]
    ev_teacher_train = ev_support[:int(0.8 * inp.shape[0]), :, :]
    ev_teacher_test = ev_support[int(0.8 * inp.shape[0]):, :, :]

    blend_factor = np.expand_dims(np.ones(inp_train.shape[0]), axis=-1)
    print(blend_factor.shape)

    print('The training set has an input data shape of ',
          inp_train.shape,
          'to expected value targets of ',
          ev_target_train.shape,
          'or alternatively pdf_targets of ',
          pdf_train.shape)
    print('-----------------------------------------------')
    print('The testing set has an input data shape of ',
          inp_test.shape,
          'to expected value targets of ',
          pdf_test.shape,
          'or alternatively pdf_targets of ',
          ev_target_train.shape)

    return inp_train, inp_test, ev_target_train, ev_target_test, ev_teacher_train, ev_teacher_test, blend_factor, pdf_train, pdf_test, pdf_teacher_train, pdf_teacher_test

import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler
import matplotlib.pyplot as plt


def datasets_from_data(data, sw_len_samples, fc_len_samples, fc_steps, fc_tiles, target_dims, plot=False, steps_to_new_sample=1):
    # -------------------------------------------------------------------------------------------------------------------
    # data is the whole data array (samples, variables)
    # the last dimension of data is a 'isfaulty?' array, True if a fault is at this timestep, False if not
    #-------------------------------------------------------------------------------------------------------------------
    # specs will include:
        # sw_len_samples, int   - The sliding input window length in samples
        # fc_len_samples, int   - Forecast length in samples
        # fc_steps, int         - Forecast steps
        # fc_tiles, int         - number of tiles for pdf
        #
        # target_dims, int      - dimensions of the target dimensions
        # the code will assume that the target dimensions are the same variable, just downsampled and will restructure it.
    # -------------------------------------------------------------------------------------------------------------------
    scaler = MinMaxScaler()
    faults = data[:,-1]

    scaler.fit(data)
    data_to_be_scaled = scaler.transform(data)



    data = scale(data, axis=0, with_mean=True, with_std=True, copy=True) #scale the dataset
    data[:, -1] = faults

    if plot: #plot if we want to
        __plot_data_array(data[:300,:], label='Scaled Data Set')

    target_variable = data[:, target_dims] #extract target vars


    if len(target_dims) > 1: #get min_max
        target_min_max = __get_min_max(__reconstruct(target_variable)) # reconstruct into one array, get min_max
    else:
        target_min_max = __get_min_max(target_variable)

    # create the list for the arrays
    pdf_targets = []    # pdf forecast
    pdf_supports = []
    ev_targets = []     # expected value forecast
    ev_supports = []
    sw_inputs = []      # inputs

    for timestep in range(0, data.shape[0] - fc_len_samples - sw_len_samples, steps_to_new_sample):
        sliced_sample = data[timestep:(timestep + fc_len_samples + sw_len_samples), :]
        sliced_input = sliced_sample[:-fc_len_samples, :]
        sliced_target = sliced_sample[-fc_len_samples:, target_dims]
        sliced_support = sliced_sample[-(fc_len_samples+1):1, target_dims]

        if len(target_dims) > 1:
            sliced_target = __reconstruct(sliced_target)
            sliced_support = __reconstruct(sliced_support)

        if np.sum(sliced_input[:, -1]) == 0:  # see if we are free of faults

            sw_inputs.append(
                sliced_input[:, :-1])  # drop the last dimension, since we do not need the fault indicator anymore

            pdf_target = __convert_to_pdf(sliced_target,
                                          num_steps=fc_steps, num_tiles=fc_tiles,
                                        min_max=target_min_max)  # pdf for probability distribution thingie
            pdf_support = __convert_to_pdf(sliced_support,
                                          num_steps=fc_steps, num_tiles=fc_tiles,
                                        min_max=target_min_max)  # pdf for probability distribution thingie
            tolerance = 1e-7
            if 1.0-tolerance <= np.mean(np.sum(pdf_target, axis=-1)) >= 1.0 + tolerance:
                print('ooh, u fucked up boi, ', np.mean(np.sum(pdf_target, axis=-1)))
            pdf_targets.append(pdf_target)
            pdf_supports.append(pdf_support)

            ev_target = __convert_to_mv(sliced_target, num_steps=fc_steps)  # ev for expected value
            ev_targets.append(ev_target)
            ev_support = __convert_to_mv(sliced_support, num_steps=fc_steps)  # ev for expected value
            ev_supports.append(ev_support)


        if (int(timestep/steps_to_new_sample) % int(int(data.shape[0] / steps_to_new_sample)/10)) == 0:
            print(int(100*timestep/data.shape[0]), 'percent converted')
    sw_inputs = np.array(sw_inputs)
    ev_targets = np.array(ev_targets)
    ev_supports = np.array(ev_supports)
    pdf_targets = np.array(pdf_targets)
    pdf_supports = np.array(pdf_supports)
    return sw_inputs, ev_targets, ev_supports, pdf_targets, pdf_supports

def __plot_data_array(data_arrays, label): # plots any provided 2D data arrat

    fig, ax = plt.subplots(data_arrays.shape[1] , sharex=True, figsize=(20, 15))

    for array in range(data_arrays.shape[1] ):
        ax[array].plot(data_arrays[:, int(array)], c=[1, 0, 0], label=label)

        if array == 0:
            ax[array].legend()

    ax[data_arrays.shape[1] - 1].set_xlabel("time in samples")
    plt.show()

def __get_min_max(array): #selfexplanatory
    return [np.amin(array), np.amax(array)]

def __reconstruct(mvts_array):  # needed to reconstruct the original array
    array = []

    for ts in range(mvts_array.shape[0]):
        for dim in range(mvts_array.shape[1]):
            array.append(mvts_array[ts, dim])

    return np.array(array)

def __convert_to_pdf(target, num_tiles, num_steps, min_max):  # takes the target timeseries and converts it into a pdf
    # the target array has a certain length
    # num_tiles is the granulatiry of the pdf
    # num_steps is the target dowsample rate

    pdf_target = np.zeros([num_steps, num_tiles])

    target = target - min_max[0]
    min_max[1] = min_max[1] - min_max[0]  # rescale the maximum with regards to the minimum

    samples_in_step = int(target.shape[0] / num_steps)  # see how many steps we need to bunch into one pdf-step

    target = target / min_max[1]  # rescale
    target = np.where(target == 1, target - 1e-7, target)  # make sure we dont get overflow
    target = np.floor(num_tiles * target)  # convert to indices

    for step in range(num_steps):
        for _ in range(samples_in_step):
            pdf_target[step, int(target[step * samples_in_step + _])] += 1.0 / samples_in_step

    return pdf_target

def __convert_to_mv(target, num_steps):  # creates expected value targets

    mv_target = np.zeros([num_steps])

    samples_in_step = int(target.shape[0] / num_steps)

    for step in range(num_steps):
        for _ in range(samples_in_step):
            mv_target[step] += target[step * samples_in_step + _] / samples_in_step

    return mv_target






# Help functions for Daniel to load his Data

import os, datetime, re
import pandas as pd
import numpy as np


def load_dataset(weather_data_folder='\_data\Weather_2016', pv_data_path="/_data/"):

    print('fetching NWP...')

    weather_mvts, start_end_date =_get_weather_data(weather_data_folder)
    print('fetched NWP ... now fetching PV Data')

    gen = _get_PV(pv_data_path, start_end_date)
    print('fetched PV ... now concatenating')

    generation = __stack_ts(gen, int(gen.shape[0] / weather_mvts.shape[0]))

    full_dataset = np.append(generation, weather_mvts, axis=-1)
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
        end_prev_sim = weather_mvts[-1,0]





    print('replacing the fauly values with the mean of the array')
    faults = weather_mvts[:, -1]
    weather_mvts = np.delete(weather_mvts, -1, axis=-1)

    print('converting wind to polar coordinates')
    wind = weather_mvts[:,-1]
    weather_mvts = np.delete(weather_mvts, -1, axis=-1)
    wind_in_radians = wind * np.pi / 180
    wind_in_radians = np.array(wind_in_radians, dtype=np.float)
    wind_in_cos = np.expand_dims(np.cos(wind_in_radians), axis=-1)
    wind_in_sin = np.expand_dims(np.sin(wind_in_radians), axis=-1)
    weather_mvts = np.concatenate((weather_mvts, wind_in_cos, wind_in_sin), axis=-1)


    datetime_array = weather_mvts[:,0].tolist()
    weather_mvts = np.delete(weather_mvts, 0, axis=-1)
    datetime_array_daytime = [datetime_entry.hour * 60 + datetime_entry.minute for datetime_entry in datetime_array]
    datetime_array_daytime_normed_to_1 = datetime_array_daytime / np.amax(datetime_array_daytime)
    daytime_in_radians = datetime_array_daytime_normed_to_1 * 2.0 * np.pi
    daytime_sin = np.expand_dims(np.sin(daytime_in_radians),axis=-1)
    daytime_cos = np.expand_dims(np.cos(daytime_in_radians), axis=-1)

    datetime_array_yeartime = [datetime_entry.month * (365/12) * 24 * 60 + datetime_entry.day*24*60 + datetime_entry.hour * 60 + datetime_entry.minute for datetime_entry in datetime_array]
    datetime_array_yeartime_normed_to_1 = datetime_array_yeartime / np.amax(datetime_array_yeartime)
    yeartime_in_radians = datetime_array_yeartime_normed_to_1 * 2.0 * np.pi
    yeartime_sin = np.expand_dims(np.sin(yeartime_in_radians),axis=-1)
    yeartime_cos = np.expand_dims(np.cos(yeartime_in_radians), axis=-1)


    for axis in range(1, weather_mvts.shape[-1] - 1): #for all but the last data array
        mean = np.mean(weather_mvts[:,axis])

        weather_mvts[:, axis] = np.where(faults == 1, mean, weather_mvts[:, axis])
    faults = np.expand_dims(faults, axis=-1)
    weather_mvts = np.concatenate((weather_mvts, daytime_cos, daytime_sin, yeartime_cos, yeartime_sin, faults), axis=-1)
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
    profile = __import_profile(profile_path, time_start=start.strftime('%Y-%m-%d%H:%M:%S'),
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


def import_zp(filename):
    import gzip, pickle
    f = gzip.open(filename + '.zp', 'rb')
    p_obj = pickle.load(f)
    f.close()
    return p_obj

def __import_profile(pickle_path_name, time_start=None, time_end=None):
    from datetime import datetime
    offline_profile = import_zp(pickle_path_name)

    if 'load' not in offline_profile:
        return None

    time_start_dt = None
    time_start_index = 0
    time_end_index = len(offline_profile['load']['profile'])-1

    if time_start is not None:
        time_start_dt = datetime.strptime(time_start, '%Y-%m-%d%H:%M:%S')
        time_start_index = int((time_start_dt - offline_profile['load']['time_start']).total_seconds() / 60)

    if time_end is not None:
        time_end_dt = datetime.strptime(time_end, '%Y-%m-%d%H:%M:%S')
        time_end_index = int((time_end_dt - offline_profile['load']['time_start']).total_seconds() / 60)

    offline_profile['load']['time_start'] = time_start_dt
    offline_profile['load']['profile'] = offline_profile['load']['profile'][time_start_index:time_end_index]

    if 'gen' in offline_profile:
        offline_profile['gen']['time_start'] = time_start_dt
        offline_profile['gen']['profile'] = offline_profile['gen']['profile'][time_start_index:time_end_index]

    return offline_profile
