from sklearn.preprocessing import scale, StandardScaler
import matplotlib.pyplot as plt
import os, datetime, re
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

def get_Daniels_data(target_data='Pv',
                     input_len_samples=int(3 * 24 * (60 / 5)),
                     fc_len_samples=int(1 * 24 * (60 / 5)),
                     fc_steps=24,
                     fc_tiles=50):
    # Daniel loads his data / If we want different data we do that here
    # For a encoder decoder model we will need 3 inputs:
    # encoder input, decoder support, blend factor (for how much we want to do teacher forcing)
    raw_data, sample_rate_weather = load_dataset()

    if target_data == 'PV' or target_data == 'pv' or target_data == 'Pv':
        target_dims = [0, 1, 2, 3, 4]
    elif target_data == 'Load' or target_data == 'load' or target_data == 'LOAD':
        target_dims = None
        print('Not Implemented YET, except funky behavior')
        # ToDo: make sure you can add the load data here!

    steps_to_new_sample = 3
    inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher = datasets_from_data(raw_data,
                                                          sw_len_samples=input_len_samples,
                                                          fc_len_samples=fc_len_samples,
                                                          fc_steps=fc_steps,
                                                          fc_tiles=fc_tiles,
                                                          target_dims=target_dims,
                                                          plot=False,
                                                          steps_to_new_sample=steps_to_new_sample)

    return inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, steps_to_new_sample*sample_rate_weather


def get_Lizas_data():
    import gzip, pickle
    data_path = './data/Lizas_data.pklz'
    f = gzip.open(data_path, 'rb')
    NetworkData = pickle.load(f)
    f.close()

    mvts_array = NetworkData['dataset']
    specs = NetworkData['specs']
    print('Specs:', specs)
    sample_rate_raw_data_mins = 15.0
    steps_to_new_sample = 12
    inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher = datasets_from_data(data=mvts_array,
                                      sw_len_samples=specs['sw_steps'],
                                      fc_len_samples=specs['num_steps'],
                                      fc_steps=specs['forecast_steps'],
                                      fc_tiles=specs['num_tiles'],
                                      target_dims=specs['target'],
                                      plot=False,
                                      steps_to_new_sample=steps_to_new_sample)

    # ---------------------------------------------------------------------------
    sample_spacing_in_mins = sample_rate_raw_data_mins*steps_to_new_sample

    return inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins


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

    faults = data[:,-1]
    data = data[:,:-1]
    for axis in range(data.shape[-1]):
        min = np.amin(data[:,axis])
        data[:, axis] = (data[:, axis] - min)
        max = np.amax(data[:,axis])
        if max != 0:
            data[:,axis] = data[:, axis]/max
    scaler = StandardScaler()
    scaler.fit(data)
    data = scale(data, axis=0, with_mean=True, with_std=True, copy=True) #scale the dataset


    if plot: #plot if we want to
        __plot_data_array(data[:300,:], label='Scaled Data Set')

    targets = data[:, target_dims] #extract target vars
    target_min = np.amin(targets)
    targets = targets - target_min
    target_min_max = [0.0,  np.amax(targets)]

    # create the list for the arrays
    fc_len_samples = int(fc_len_samples)
    fc_steps = int(fc_steps)
    sw_len_samples = int(sw_len_samples)
    fc_tiles = int(fc_tiles)
    num_samples = int((data.shape[0] - fc_len_samples - sw_len_samples)/steps_to_new_sample)

    pdf_targets = np.zeros([num_samples, fc_steps,fc_tiles])    # pdf forecast
    pdf_supports = np.zeros([num_samples, fc_steps,fc_tiles])
    ev_targets = np.zeros([num_samples, fc_steps,1])     # expected value forecast
    ev_supports = np.zeros([num_samples, fc_steps,1])
    sw_inputs = np.zeros([num_samples, sw_len_samples, data.shape[-1]])      # inputs

    for sample in range(num_samples):
        timestep = sample*steps_to_new_sample
        sliced_input_sample = data[timestep:(timestep + sw_len_samples), :]
        faults_for_slice = faults[timestep:(timestep + fc_len_samples + sw_len_samples)]

        target_timestep = timestep + sw_len_samples
        one_addl_fc_timestep = int(fc_len_samples/fc_steps)
        sliced_target_sample = targets[(target_timestep-one_addl_fc_timestep):(target_timestep + fc_len_samples), :]

        if len(target_dims) > 1:
            sliced_target_sample = __reconstruct(sliced_target_sample)

        if np.sum(faults_for_slice) == 0:  # see if we are free of faults
            sw_inputs[sample, :,:] = sliced_input_sample  # drop the last dimension, since we do not need the fault indicator anymore

            pdf_Support_target = __convert_to_pdf(sliced_target_sample,
                                          num_steps=fc_steps+1, num_tiles=fc_tiles,
                                        min_max=target_min_max)  # pdf for probability distribution thingie
            tolerance = 1e-9
            if 1.0-tolerance <= np.mean(np.sum(pdf_Support_target, axis=-1)) >= 1.0 + tolerance:
                print('ooh, u fucked up boi, ', np.mean(np.sum(pdf_Support_target, axis=-1)))

            pdf_targets[sample, :,:] = pdf_Support_target[1:,:]
            pdf_supports[sample, :,:] = pdf_Support_target[:-1,:]

            ev_target = __convert_to_mv(sliced_target_sample, num_steps=fc_steps+1)  # ev for expected value
            ev_targets[sample, :, 0] = ev_target[1:]
            ev_supports[sample, :, 0] = ev_target[:-1]

        if (int(timestep/steps_to_new_sample) % int(int(data.shape[0] / steps_to_new_sample)/10)) == 0:
            print(int(100*timestep/data.shape[0]) + 10, 'percent converted')

        sample += 1

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
    min_max[1] = min_max[1] + min_max[0]  # rescale the maximum with regards to the minimum
    samples_in_step = int(target.shape[0] / num_steps)  # see how many steps we need to bunch into one pdf-step
    target = target / min_max[1]  # rescale
    target = np.where(target == 1, target - 1e-9, target)  # make sure we dont get overflow
    target = np.floor(num_tiles * target)  # convert to indices
    for step in range(num_steps):
        for sample in range(samples_in_step):
            pdf_target[step, int(target[step * samples_in_step + sample])] += 1.0 / samples_in_step
    return pdf_target

def __convert_to_mv(target, num_steps):  # creates expected value targets

    mv_target = np.zeros([num_steps])
    samples_in_step = int(target.shape[0] / num_steps)
    for step in range(num_steps):
        for _ in range(samples_in_step):
            mv_target[step] += target[step * samples_in_step + _] / samples_in_step
    return mv_target

# Help functions for Daniel to load his Data

def load_dataset(weather_data_folder='\_data\Weather_2016', pv_data_path="/_data/"):

    print('fetching NWP...')

    weather_mvts, start_end_date =_get_weather_data(weather_data_folder)
    print('fetching PV Data')
    gen = _get_PV(pv_data_path, start_end_date)
    print('now concatenating')
    sample_rate_weather = int(gen.shape[0] / weather_mvts.shape[0])
    generation = __stack_ts(gen, sample_rate_weather)
    full_dataset = np.append(generation, weather_mvts, axis=-1)
    return full_dataset, sample_rate_weather

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

def __slice_and_delete(inp, teacher, target, len_slice, seed, sample_spacing_in_mins, input_rate_in_1_per_min=5, previously_taken_start_indices=[], previously_taken_end_indices=[]):
    np.random.seed(seed)
    inp_sw_shape = inp.shape
    one_sw_in_samples = (inp_sw_shape[1]*input_rate_in_1_per_min)/sample_spacing_in_mins
    one_sw_in_samples = int(one_sw_in_samples)

    leftover_samples = int(len_slice)

    day_offset_in_samples = int((24*60)/sample_spacing_in_mins)

    inp_separated = np.zeros([leftover_samples, inp.shape[1], inp.shape[2]])
    teacher_separated = np.zeros([leftover_samples, teacher.shape[1], teacher.shape[2]])
    teacher_blend_separated = np.zeros([leftover_samples])
    target_separated = np.zeros([leftover_samples, target.shape[1], target.shape[2]])
    persistency_forecast = np.zeros([leftover_samples, target.shape[1], target.shape[2]])

    index_start = []
    index_end = []
    failed_tries = 0
    while leftover_samples > 0:
        one_week_in_samples = min(one_sw_in_samples, leftover_samples)

        index_start_slice = np.random.uniform(low=day_offset_in_samples, high=inp.shape[0] - 2*one_sw_in_samples, size=None)
        index_start_slice = int(np.floor(index_start_slice))
        index_end_slice = index_start_slice + one_week_in_samples

        start_falls_within = False
        end_falls_within = False
        for entry in range(len(previously_taken_start_indices)):
            start_falls_within = start_falls_within or ((index_start_slice >= previously_taken_start_indices[entry]) & (index_start_slice <= previously_taken_end_indices[entry]))
            end_falls_within = end_falls_within or ((index_end_slice >= previously_taken_start_indices[entry]) & (index_end_slice <= previously_taken_end_indices[entry]))

        if not start_falls_within and not end_falls_within:
            index_start.append(index_start_slice)
            index_end.append(index_end_slice)
            previously_taken_start_indices.append(index_start_slice)
            previously_taken_end_indices.append(index_end_slice)
            leftover_samples -= one_week_in_samples
        else:
            failed_tries += 1
            if failed_tries > 1e6:
                print('huh, this is a lot of failed tries... damn')
                break

    index_start_data = 0
    for entry in range(len(index_start)):
        index_start_slice = index_start[entry]
        index_end_slice = index_end[entry]
        num_samples = int(index_end_slice - index_start_slice)

        inp_separated[index_start_data:(index_start_data + num_samples) :, :] = inp[index_start_slice:index_end_slice,:,:]

        teacher_separated[index_start_data:(index_start_data + num_samples), :, :] = teacher[index_start_slice:index_end_slice,:,:]

        persistency_forecast[index_start_data:(index_start_data + num_samples), :, :] = target[(index_start_slice - day_offset_in_samples):(
                    index_end_slice - day_offset_in_samples), :, :]
        target_separated[index_start_data:(index_start_data + num_samples), :, :] = target[index_start_slice:index_end_slice, :, :]

        index_start_data += num_samples

        if persistency_forecast.shape[0] != target_separated.shape[0]:
            print('WTF, seems theres a mismatch', persistency_forecast.shape, 'vs', target_separated.shape)
            print('slice persistency from',(index_start_slice-day_offset_in_samples), 'to',  (index_end_slice-day_offset_in_samples), '= ', ((index_end_slice-day_offset_in_samples)-(index_start_slice-day_offset_in_samples)))
            print('slice persistency from', (index_start_slice), 'to', (index_end_slice ), '= ', ((index_end_slice) - (index_start_slice)))

    sliced_dataset = [inp_separated, teacher_separated, teacher_blend_separated]
    return sliced_dataset, target_separated, persistency_forecast, inp, teacher, target, previously_taken_start_indices, previously_taken_end_indices

def __split_dataset(inp, target, teacher, training_ratio, sample_spacing_in_mins, normalizer_value, input_rate_in_1_per_min, split_seeds=[24,42]):
    if training_ratio > 1:
        print('... seems like you want more than a full training set, the training ratio needs to be smaller than 1!')

    remainder_for_test_val = 1.0-training_ratio
    test_len = (remainder_for_test_val/2.0) * inp.shape[0]
    val_len = (remainder_for_test_val/2.0) * inp.shape[0]

    from Losses_and_Metrics import __calculatate_skillscore_baseline
    persistency_baseline = __calculatate_skillscore_baseline(target,
                                                           sample_spacing_in_mins=sample_spacing_in_mins,
                                                           normalizer_value=normalizer_value)
    tf.keras.backend.clear_session()
    print('Persistency baseline for whole Dataset', persistency_baseline)
    dataset = {}
    [dataset['test_inputs'], dataset['test_teacher'], dataset['test_blend']], dataset['test_targets'], test_persistent_forecast,  inp, teacher, target, previously_taken_start_indices, previously_taken_end_indices = __slice_and_delete(inp,
                                                                                                                                                                            teacher,
                                                                                                                                                                            target,
                                                                                                                                                                            test_len,
                                                                                                                                                                            seed=split_seeds[0],
                                                                                                                                                                            sample_spacing_in_mins=sample_spacing_in_mins,
                                                                                                                                                                            input_rate_in_1_per_min=input_rate_in_1_per_min)
    dataset['test_persistency_baseline'] = __calculatate_skillscore_baseline(dataset['test_targets'],
                                                                             persistent_forecast=test_persistent_forecast,
                                                                           sample_spacing_in_mins=sample_spacing_in_mins,
                                                                           normalizer_value=normalizer_value)
    tf.keras.backend.clear_session()
    print('test baseline values:', dataset['test_persistency_baseline'])
    [dataset['val_inputs'], dataset['val_teacher'], dataset['val_blend']], dataset['val_targets'], val_persistent_forecast, inp, teacher, target, previously_taken_start_indices, previously_taken_end_indices = __slice_and_delete(inp,
                                                                                                                                                                      teacher,
                                                                                                                                                                      target,
                                                                                                                                                                      val_len,
                                                                                                                                                                      seed=split_seeds[1],
                                                                                                                                                                      sample_spacing_in_mins=sample_spacing_in_mins,
                                                                                                                                                                      input_rate_in_1_per_min=input_rate_in_1_per_min, previously_taken_start_indices=previously_taken_start_indices, previously_taken_end_indices=previously_taken_end_indices)
    dataset['val_persistency_baseline'] = __calculatate_skillscore_baseline(dataset['val_targets'],
                                                                            persistent_forecast=val_persistent_forecast,
                                                                           sample_spacing_in_mins=sample_spacing_in_mins,
                                                                           normalizer_value=normalizer_value)
    tf.keras.backend.clear_session()

    print('val baseline values:', dataset['val_persistency_baseline'])
    previously_taken_start_indices.sort(reverse=True)
    previously_taken_end_indices.sort(reverse=True)
    for entry in range(len(previously_taken_start_indices)):
        index_start_slice = previously_taken_start_indices[entry]
        index_end_slice = previously_taken_end_indices[entry]
        target = np.delete(target, np.s_[index_start_slice:index_end_slice], axis=0)
        teacher = np.delete(teacher, np.s_[index_start_slice:index_end_slice], axis=0)
        inp = np.delete(inp, np.s_[index_start_slice:index_end_slice], axis=0)

    blend_train = [1] * inp.shape[0]
    dataset['train_inputs'] = np.asarray(inp)
    dataset['train_teacher'] = np.asarray(teacher)
    dataset['train_blend'] = np.asarray(blend_train)
    dataset['train_targets'] = np.asarray(target)

    print('Dataset has', dataset['train_targets'].shape[0], 'training samples', dataset['val_targets'].shape[0], 'val samples', dataset['test_targets'].shape[0], 'test samples')
    return dataset

def __augment_Daniels_dataset(dataset):
    if 'val_inputs' in dataset:

        corrected_shape = dataset['val_inputs'].shape
        new_values = np.zeros(shape=[corrected_shape[0], corrected_shape[1], corrected_shape[-1] - 5 + 1])
        for sample in range(len(dataset['val_inputs'])):
            sample_inputs = dataset['val_inputs'][sample]
            pv_sample = sample_inputs[:, 0:5]
            rest_sample = sample_inputs[:, (pv_sample.shape[-1]):]
            pv_sample = np.mean(pv_sample, axis=-1)
            new_values[sample,:,0] = pv_sample
            new_values[sample,:,1:] = rest_sample

        dataset['val_inputs'] = np.asarray(new_values)

    if 'test_inputs' in dataset:
        print('adjusting test set')
        corrected_shape = dataset['test_inputs'].shape
        new_values = np.zeros(shape=[corrected_shape[0], corrected_shape[1], corrected_shape[-1] - 5 + 1])
        for sample in range(len(dataset['test_inputs'])):
            sample_inputs = dataset['test_inputs'][sample]
            pv_sample = sample_inputs[:, 0:5]
            rest_sample = sample_inputs[:, (pv_sample.shape[-1]):]
            pv_sample = np.mean(pv_sample, axis=-1)
            new_values[sample,:,0] = pv_sample
            new_values[sample,:,1:] = rest_sample

        dataset['test_inputs'] = np.asarray(new_values)

    if 'train_inputs' in dataset:
        print('adjusting train_inputs', dataset['train_inputs'].shape)
        original_inputs_shape = dataset['train_inputs'].shape

        new_values = np.zeros(shape=[original_inputs_shape[0]*5, original_inputs_shape[1], original_inputs_shape[-1] - 5 + 1])
        new_teacher = np.zeros(shape=[dataset['train_teacher'].shape[0]*5, dataset['train_teacher'].shape[1], dataset['train_teacher'].shape[2]])
        new_blend = np.zeros(shape=[dataset['train_blend'].shape[0] * 5])
        new_targets = np.zeros(shape=[dataset['train_targets'].shape[0] * 5, dataset['train_targets'].shape[1], dataset['train_targets'].shape[2]])

        sample_inputs = dataset['train_inputs']
        pv = sample_inputs[:,:, 0:5]
        rest = sample_inputs[:,:, pv.shape[-1]:]
        num_sets = pv.shape[-1]
        num_original_samples = original_inputs_shape[0]
        for pv_set in range(num_sets):
            pv_sample = pv[:,:,pv_set]
            augmented_features = rest
            feature_deltas = np.subtract(augmented_features[:,1:,:], augmented_features[:,:-1,:])

            for sample in range(num_original_samples):
                offset_bias = np.random.randint(low=1, high=num_sets)
                for feature in range(augmented_features.shape[-1]):
                    offsets = feature_deltas[sample, :, feature]
                    offset_noise = np.random.uniform(low=0.0, high=1.0, size=offsets.shape)
                    offset_noise = np.multiply(offsets/(num_sets+1), offset_noise)
                    offsets = np.add(offsets*offset_bias/(num_sets+1), offset_noise)
                    augmented_features[sample,:-1,feature] = np.add(augmented_features[sample,:-1,feature], offsets)

                    new_values[num_original_samples*pv_set + sample,:,0] = pv_sample[sample,:]
                    new_values[num_original_samples*pv_set + sample,:,1:] = augmented_features[sample,:,:]
                    new_teacher[num_original_samples*pv_set + sample,:,:] = dataset['train_teacher'][sample,:,:]
                    new_blend[num_original_samples * pv_set + sample] = dataset['train_blend'][sample]
                    new_targets[num_original_samples * pv_set + sample, :, :] = dataset['train_targets'][sample, :, :]

        dataset['train_inputs'] = new_values
        dataset['train_teacher'] = new_teacher
        dataset['train_blend'] = new_blend
        dataset['train_targets'] = new_targets
        print('train_inputs are now: ', dataset['train_inputs'].shape)

    return dataset

def __create_and_save_1_fold_CrossValidation(name='Daniels_dataset_2', seeds=[43, 42]):
    inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins = get_Daniels_data()
    normalizer_value = np.amax(ev_targets) - np.amin(ev_targets)
    dataset_splitter_kwargs = {'inp':inp,
                              'target':pdf_targets,
                              'teacher':pdf_teacher,
                              'training_ratio':0.6,
                              'sample_spacing_in_mins':sample_spacing_in_mins,
                              'normalizer_value':normalizer_value,
                              'input_rate_in_1_per_min':5}

    dataset_splitter_kwargs['split_seeds'] = seeds
    dataset = __split_dataset(**dataset_splitter_kwargs)
    dataset = __augment_Daniels_dataset(dataset)
    dataset['normalizer_value'] = normalizer_value
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del dataset

    del inp, pdf_teacher, pdf_targets, ev_teacher, ev_targets


