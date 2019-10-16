from sklearn.preprocessing import scale, StandardScaler
import matplotlib.pyplot as plt
import os, datetime, re
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import shutil

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
        target_dims = [0, 1, 2, 3, 4]
        print('Not Implemented, and tested YET, except funky behavior')
        # ToDo: make sure you can add the load data here!

    steps_to_new_sample = 1
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
    steps_to_new_sample = 2
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

def Create_full_dataset(dataset_name='Daniels_Dataset_1',
                    weather_data_folder='\_data\Weather_2016', pv_data_path="/_data/",
                     sw_len_samples=int(5 * 24 * 60),
                     fc_len_samples=int(1 * 24 * 60),
                     fc_steps=24,
                     fc_tiles=40):
    # get the NWP data, separate faults array
    weather_nwp, start_end_date = _get_weather_data(weather_data_folder)
    faults = weather_nwp[:,-1]
    weather_nwp = weather_nwp[:, :-1]
    # get the NWP data, interpolate, normalize
    # get the PV data and normalize
    historical_pv = _get_PV(pv_data_path, start_end_date)
    history_shape = historical_pv.shape
    scaler = StandardScaler()
    historical_pv = np.expand_dims(historical_pv, axis=1)
    scaler.fit(historical_pv)
    historical_pv = scale(historical_pv, axis=0, with_mean=True, with_std=True, copy=True)
    historical_pv = np.squeeze(historical_pv)
    history_min_max = [np.amin(historical_pv), np.amax(historical_pv)]
    print('history min_max', history_min_max)

    nwp_shape = weather_nwp.shape
    history_indices = np.arange(history_shape[0])
    nwp_indices = np.arange(0, history_shape[0], int(history_shape[0] / nwp_shape[0]))
    weather_nwp_interpolated = np.zeros([history_shape[0], nwp_shape[1]])
    faults = np.repeat(faults, int(history_shape[0] / nwp_shape[0]))
    # interpolate
    for feature_axis in range(nwp_shape[1]):
        features = np.asarray(weather_nwp[:, feature_axis], dtype=np.float64)
        insert = np.interp(history_indices, nwp_indices, features)
        weather_nwp_interpolated[:, feature_axis] = insert
    del weather_nwp, nwp_shape

    # normalize and scale
    for axis in range(weather_nwp_interpolated.shape[-1]):
        min = np.amin(weather_nwp_interpolated[:,axis])
        weather_nwp_interpolated[:, axis] = (weather_nwp_interpolated[:, axis] - min)
        if np.amin(weather_nwp_interpolated[:,axis]) != 0.0:
            print('somehow failed to level data to 0 min')
        max = np.amax(weather_nwp_interpolated[:,axis])
        if max != 0:
            weather_nwp_interpolated[:,axis] = weather_nwp_interpolated[:, axis]/max
    scaler = StandardScaler()
    scaler.fit(weather_nwp_interpolated)
    weather_nwp_interpolated = scale(weather_nwp_interpolated, axis=0, with_mean=True, with_std=True, copy=True) #scale the dataset


    print('fetching indices')
    train_indices, val_indices, test_indices = assign_samples_to_set(faults,
                                                                     train_ratio=0.8,
                                                                     sample_length=sw_len_samples+fc_len_samples,
                                                                     val_seed=1,
                                                                     test_seed=2,
                                                                     )
    print('fetches', len(train_indices), 'train samples, ', len(test_indices), 'test samples ', len(val_indices), 'val_samples')

    target_len_samples = int(fc_len_samples)
    target_steps = int(fc_steps)
    target_tiles = int(fc_tiles)
    one_addl_fc_timestep = int(fc_len_samples / fc_steps)
    target_downsampling_rate = target_len_samples / target_steps
    nwp_downsampling_rate = 15
    support_steps = int(sw_len_samples / target_downsampling_rate)
    sw_len_samples = int(sw_len_samples)

    def _features_to_example(nwp_input, historical_support_raw, teacher_raw, target_raw):
        target_pdf = __convert_to_pdf(target_raw, num_steps=target_steps, num_tiles=target_tiles, min_max=history_min_max)  # pdf for probability distribution thingie
        teacher_pdf = __convert_to_pdf(teacher_raw, num_steps=target_steps, num_tiles=target_tiles, min_max=history_min_max)
        historical_support_pdf = __convert_to_pdf(historical_support_raw, num_steps=support_steps, num_tiles=target_tiles, min_max=history_min_max)

        features = {'nwp_input': __create_float_feature(nwp_input.flatten()),

                    'raw_historical_input': __create_float_feature(historical_support_raw.flatten()),
                    'raw_teacher': __create_float_feature(teacher_raw.flatten()),
                    'raw_target': __create_float_feature(target_raw.flatten()),

                    'pdf_historical_input': __create_float_feature(historical_support_pdf.flatten()),
                    'pdf_teacher': __create_float_feature(teacher_pdf.flatten()),
                    'pdf_target': __create_float_feature(target_pdf.flatten())}
        if 'nwp_input' not in features:
            print('WTF, lost NWP input!!')
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example.SerializeToString()


    # Delete Dataset folder if already exists, create new folder
    current_wd = os.getcwd()
    if os.path.isdir(dataset_name):
        print('deleting previous dataset')
        shutil.rmtree(dataset_name, ignore_errors=True)
    os.mkdir(dataset_name)
    dataset_folder_path = current_wd + '/' + dataset_name
    os.chdir(dataset_folder_path)
    from Losses_and_Metrics import __calculatate_skillscore_baseline

    def __save_set_to_folder(sub_dataset, folder_name='train', save_every_xth_step=1):
        os.mkdir(folder_name)  # create a folder, move to the folder
        os.chdir((dataset_folder_path + '/' + folder_name))
        set_nRMSE = []
        set_nME = []
        set_CRPS = []
        # downsample dataset granularity
        if save_every_xth_step > 1:
            sub_dataset = sub_dataset[::save_every_xth_step]
        if len(sub_dataset) > 1e5:
            partials = True
            target_pdf = np.zeros([int(len(sub_dataset)/100), fc_steps, fc_tiles])
            persistency_pdf = np.zeros([int(len(sub_dataset)/100), fc_steps, fc_tiles])
        else:
            partials = False
            target_pdf = np.zeros([len(sub_dataset), fc_steps, fc_tiles])
            persistency_pdf = np.zeros([len(sub_dataset), fc_steps, fc_tiles])
        num_sample = 0
        for sample_index in sub_dataset:
            if sample_index - fc_len_samples - sw_len_samples > weather_nwp_interpolated.shape[0]:
                print('fuck, sth went wrong,index too high!!')

            nwp_input = weather_nwp_interpolated[sample_index:(sample_index+sw_len_samples+fc_len_samples):nwp_downsampling_rate, :]
            history_support = historical_pv[sample_index:(sample_index+sw_len_samples)]

            target_start_index = sample_index + sw_len_samples
            target = historical_pv[target_start_index:(target_start_index+fc_len_samples)]

            teacher_start_index = target_start_index - one_addl_fc_timestep
            teacher = historical_pv[teacher_start_index:(teacher_start_index+fc_len_samples)]

            persistency_start_index = target_start_index - fc_len_samples
            persistency = historical_pv[persistency_start_index:(persistency_start_index+fc_len_samples)]

            target_sample_pdf = __convert_to_pdf(target, num_steps=target_steps, num_tiles=target_tiles, min_max=history_min_max)
            target_pdf[num_sample,:,:]  = target_sample_pdf
            if np.sum(target_sample_pdf) - fc_steps > 1e-6 or np.sum(target_sample_pdf) - fc_steps < -1e6:
                print('sth is wrong', np.sum(target_sample_pdf, axis=-1))
            persistency_pdf[num_sample, :,:] = __convert_to_pdf(persistency, num_steps=target_steps, num_tiles=target_tiles, min_max=history_min_max)
            num_sample += 1
            if partials and num_sample+1 >= len(sub_dataset)/100:
                num_sample = 0
                partial_baseline = __calculatate_skillscore_baseline(target_pdf, persistent_forecast=persistency_pdf, normalizer_value=np.amax(historical_pv))
                set_nME.append(partial_baseline['nME'])
                set_nRMSE.append(partial_baseline['nRMSE'])
                set_CRPS.append(partial_baseline['CRPS'])

            with tf.io.TFRecordWriter(str(sample_index) + '.tfrecord') as writer:

                example = _features_to_example(nwp_input=nwp_input,
                                               historical_support_raw=history_support,
                                               teacher_raw=teacher,
                                               target_raw=target)
                writer.write(example)

                    # we're done with val set, so jump back to the dataset folder
        os.chdir(dataset_folder_path)
        if not partials:
            sample_baseline = __calculatate_skillscore_baseline(target_pdf, persistent_forecast=persistency_pdf, normalizer_value=np.amax(historical_pv))
        else:
            sample_baseline = {
                'nME': tf.reduce_mean(set_nME).numpy(),
                'nRMSE': tf.reduce_mean(set_nRMSE).numpy(),
                'CRPS': tf.reduce_mean(set_CRPS).numpy()
            }
        return sample_baseline

    dataset_info = {}
    sw_as_pdf_len = sw_len_samples*fc_steps/fc_len_samples
    dataset_info['nwp_downsampling_rate'] = nwp_downsampling_rate
    dataset_info['nwp_dims'] = weather_nwp_interpolated.shape[1]
    dataset_info['sw_len_samples'] = sw_len_samples
    dataset_info['fc_len_samples'] = fc_len_samples
    dataset_info['fc_tiles'] = fc_tiles
    dataset_info['fc_steps'] = fc_steps
    dataset_info['input_shape'] = [(sw_len_samples+fc_len_samples)/nwp_downsampling_rate, weather_nwp_interpolated.shape[1]]
    dataset_info['teacher_shape'] = [fc_steps, fc_tiles]
    dataset_info['target_shape'] = [fc_steps, fc_tiles]
    dataset_info['history_support_shape'] = [sw_len_samples*fc_steps/fc_len_samples, fc_tiles]
    dataset_info['normalizer_value'] = np.amax(historical_pv) - np.amin(historical_pv)


    val_baseline = __save_set_to_folder(val_indices,
                                        folder_name='validation',
                                        save_every_xth_step=10)
    dataset_info['val_baseline'] = val_baseline
    print('saved val set, baseline:', val_baseline)

    with open('dataset_info' + '.pickle', 'wb') as handle:
        pickle.dump(dataset_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    test_baseline = __save_set_to_folder(test_indices,
                                        folder_name='test',
                                        save_every_xth_step=10)
    dataset_info['test_baseline'] = test_baseline
    print('saved test set, baseline:', test_baseline)

    train_baseline = __save_set_to_folder(train_indices,
                                        folder_name='train',
                                        save_every_xth_step=2)
    dataset_info['train_baseline'] = train_baseline
    print('saved train set, baseline:', train_baseline)

def assign_samples_to_set(faults, train_ratio=0.8, sample_length=2, val_seed=1, test_seed=2):
    num_val_samples = int(faults.shape[0] * (1.0-train_ratio)/2)
    num_test_samples = int(faults.shape[0] * (1.0-train_ratio)/2)

    num_consecutive_samples = int(1.5*sample_length)
    taken_indices = np.zeros(faults.shape, dtype=np.bool)

    num_val_samples_fetched = 0
    num_test_samples_fetched = 0

    train_indices = []
    val_indices = []
    test_indices = []

    def _fetch(faults, how_many_to_fetch_at_once, taken_indices):
        got_a_good_one = False #did we fetch a set?

        while not got_a_good_one:
            # generate a random guess of where to fetch a batch from
            index_start_subset = np.random.uniform(low=0, high=(faults.shape[0] - how_many_to_fetch_at_once - sample_length), size=None)
            index_start_subset = int(index_start_subset)
            indices_subset = np.arange(index_start_subset, index_start_subset + how_many_to_fetch_at_once)

            # see if it is available, meaning: does it have faults (any index of the set = 1?) or is it already taken?
            available = True
            for subset_index in indices_subset:
                if taken_indices[subset_index]:
                    available = False
                elif np.sum(faults[subset_index:(subset_index + sample_length)]) != 0:
                    available = False
            # if still avaliable after check, set flag to true
            if available:
                got_a_good_one = True

        return indices_subset

    # get exhaustive validation indices, not yet downsampled to every 10th!
    np.random.seed(val_seed)
    while num_val_samples_fetched < num_val_samples:
        how_many_to_fetch = min(num_consecutive_samples, num_val_samples - num_val_samples_fetched)
        next_batch = _fetch(faults, how_many_to_fetch, taken_indices)
        for index in next_batch:
            val_indices.append(index)
            taken_indices[index] = True
        num_val_samples_fetched = len(val_indices)

    # get exhaustive test indices, not yet downsampled to every 10th!
    np.random.seed(test_seed)
    while num_test_samples_fetched < num_test_samples:
        how_many_to_fetch = min(num_consecutive_samples, num_test_samples - num_test_samples_fetched)
        next_batch = _fetch(faults, how_many_to_fetch, taken_indices)
        for index in next_batch:
            test_indices.append(index)
            taken_indices[index] = True
        num_test_samples_fetched = len(test_indices)
    # get remaining training indices
    for index in range(taken_indices.shape[0] - sample_length - 1):
        if not taken_indices[index] and np.sum(faults[index:(index + sample_length)]) == 0:
            train_indices.append(index)
    return train_indices, val_indices, test_indices

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

    pdf_targets = np.zeros([num_samples, fc_steps,fc_tiles], dtype=np.float32)    # pdf forecast
    pdf_supports = np.zeros([num_samples, fc_steps,fc_tiles], dtype=np.float32)
    ev_targets = np.zeros([num_samples, fc_steps,1], dtype=np.float32)     # expected value forecast
    ev_supports = np.zeros([num_samples, fc_steps,1], dtype=np.float32)
    sw_inputs = np.zeros([num_samples, sw_len_samples, data.shape[-1]], dtype=np.float32)      # inputs

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

    pdf_target = np.zeros([num_steps, num_tiles], dtype=np.float32)
    target = target - min_max[0]
    target = target/(min_max[1] - min_max[0])
    samples_in_step = int(target.shape[0] / num_steps)  # see how many steps we need to bunch into one pdf-step
    target = np.where(target >= 1-(.3/num_tiles), 1-(.3/num_tiles), target)  # make sure we dont get overflow
    target = np.floor((num_tiles) * target)  # convert to indices
    for step in range(num_steps):
        for sample in range(samples_in_step):
            pdf_target[step, int(np.floor(target[step * samples_in_step + sample]))] += 1.0 / samples_in_step
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
    wind_in_radians = np.array(wind_in_radians, dtype=np.float32)
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

    inp_separated = np.zeros([leftover_samples, inp.shape[1], inp.shape[2]], dtype=np.float32)
    teacher_separated = np.zeros([leftover_samples, teacher.shape[1], teacher.shape[2]], dtype=np.float32)
    teacher_blend_separated = np.zeros([leftover_samples], dtype=np.float32)
    target_separated = np.zeros([leftover_samples, target.shape[1], target.shape[2]], dtype=np.float32)
    persistency_forecast = np.zeros([leftover_samples, target.shape[1], target.shape[2]], dtype=np.float32)

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
    dataset['train_inputs'] = np.asarray(inp, dtype=np.float32)
    dataset['train_teacher'] = np.asarray(teacher, dtype=np.float32)
    dataset['train_blend'] = np.asarray(blend_train, dtype=np.float32)
    dataset['train_targets'] = np.asarray(target, dtype=np.float32)

    print('Dataset has', dataset['train_targets'].shape[0], 'training samples', dataset['val_targets'].shape[0], 'val samples', dataset['test_targets'].shape[0], 'test samples')
    return dataset

def __save_and_augment_Daniels_dataset(dataset, dataset_name, augment_data=True):

    # prepare some os stuff, we gotta save the current work directory, create the dataset directory and delete the dataset directory from before if it exists!

    current_wd = os.getcwd()
    if os.path.isdir(dataset_name):
        print('deleting previous dataset')
        shutil.rmtree(dataset_name, ignore_errors=True)
    os.mkdir(dataset_name)
    dataset_folder_path = current_wd + '/' + dataset_name
    os.chdir(dataset_folder_path)

    original_inputs_shape = dataset['train_inputs'].shape # get the original inputs shape, so we can infer what shape the augmented inputs will have

    # do for val_set if in dataset
    if 'val_inputs' in dataset:
        print('saving val inputs')
        os.mkdir('validation') # create a folder, move to the folder
        os.chdir((dataset_folder_path + '/validation'))
        for sample in range(len(dataset['val_inputs'])): # for every sample in the dataset, write one tfrecord
            with tf.io.TFRecordWriter(str(sample) + '.tfrecord') as writer:

                if augment_data: # do some daniel specific stuff, adjust to the format of augmented inputs
                    sample_input = np.zeros(shape=[original_inputs_shape[1], original_inputs_shape[-1] - 5 + 1],
                                            dtype=np.float32)
                    original_data = dataset['val_inputs'][sample]
                    pv_sample = original_data[:, 0:5]
                    sample_input[:, 0] = np.mean(pv_sample, axis=-1)
                    sample_input[:, 1:] = original_data[:, (pv_sample.shape[-1]):]
                else:
                    sample_input = dataset['val_inputs'][sample]  # get sample

                # get a tf.example, and write the example to the tf.records path
                example = __example_from_features(sample_input, dataset['val_teacher'][sample, :, :], dataset['val_targets'][sample, :, :])
                writer.write(example)

        # we're done with val set, so jump back to the dataset folder
        os.chdir(dataset_folder_path)
        del dataset['val_inputs']
        del dataset['val_targets']
        del dataset['val_teacher']
        print('val saved')
    # literally a clone of the val stuff
    if 'test_inputs' in dataset:
        print('saving test inputs')
        os.mkdir('test')
        os.chdir((dataset_folder_path + '/test'))
        for sample in range(len(dataset['test_inputs'])):
            with tf.io.TFRecordWriter(str(sample) + '.tfrecord') as writer:

                if augment_data: # do some daniel specific stuff, adjust to the format of augmented inputs
                    sample_input = np.zeros(shape=[original_inputs_shape[1], original_inputs_shape[-1] - 5 + 1],
                                            dtype=np.float32)
                    original_data = dataset['test_inputs'][sample]
                    pv_sample = original_data[:, 0:5]
                    sample_input[:, 0] = np.mean(pv_sample, axis=-1)
                    sample_input[:, 1:] = original_data[:, (pv_sample.shape[-1]):]
                else:
                    sample_input = dataset['test_inputs'][sample]  # get sample

                example = __example_from_features(sample_input, dataset['test_teacher'][sample, :, :], dataset['test_targets'][sample, :, :])
                writer.write(example)
        os.chdir(dataset_folder_path)
        del dataset['test_inputs']
        del dataset['test_targets']
        del dataset['test_teacher']
        print('test saved')
    # for the training set, we augment with noisy interpolation
    if 'train_inputs' in dataset:
        print('saving training inputs')
        original_inputs_shape = dataset['train_inputs'].shape

        os.mkdir('train')
        os.chdir((dataset_folder_path + '/train'))

        if augment_data:
            pv = dataset['train_inputs'][:,:, 0:5]
            num_sets = pv.shape[-1]
            num_original_samples = original_inputs_shape[0]
            num_total_augmented_samples = num_sets*num_original_samples
            original_features = dataset['train_inputs'][:, :, pv.shape[-1]:]
            feature_deltas = np.subtract(original_features[:, 1:, :], original_features[:, :-1, :])

            sample = 0
            for pv_set in range(num_sets):
                pv_sample = pv[:,:,pv_set]


                for original_sample in range(num_original_samples):
                    offset_bias = pv_set
                    augmented_features = np.zeros(shape=[original_inputs_shape[1], original_features.shape[2]],
                                            dtype=np.float32)
                    # for each feature, perform noisy interpolation
                    for feature in range(original_features.shape[-1]):
                        offsets = feature_deltas[original_sample, :, feature]
                        offset_noise = np.random.uniform(low=0.0, high=1.0, size=offsets.shape)
                        offset_noise = np.multiply(offsets/(num_sets+1), offset_noise)
                        offsets = np.add(offsets*offset_bias/(num_sets+1), offset_noise)
                        augmented_features[original_sample,:-1,feature] = np.add(original_features[original_sample,:-1,feature], offsets)

                    sample_input = np.zeros(shape=[original_inputs_shape[1], original_inputs_shape[-1] - 5 + 1],
                                            dtype=np.float32)
                    sample_input[:,0] = pv_sample[original_sample,:]
                    sample_input[:,1:] = augmented_features[original_sample,:,:]

                    with tf.io.TFRecordWriter(str(sample) + '.tfrecord') as writer:
                        example = __example_from_features(sample_input, dataset['train_teacher'][sample,:,:], dataset['train_targets'][sample, :, :])
                        writer.write(example)
                    sample += 1
        else:
            for sample in range(len(dataset['train_inputs'])):
                with tf.io.TFRecordWriter(str(sample) + '.tfrecord') as writer:
                    example = __example_from_features(dataset['train_inputs'][sample], dataset['train_teacher'][sample,:,:], dataset['train_targets'][sample, :, :])
                    writer.write(example)

        del dataset['train_inputs']
        del dataset['train_targets']
        del dataset['train_teacher']
        print('train saved')
    # go back up one folder
    # adjust the data
    print('saving dataset info')
    os.chdir(dataset_folder_path)
    with open('dataset_info' + '.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del dataset
    os.chdir(current_wd)

def __create_float_feature(flattened_value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=flattened_value))
def __create_int_feature(flattened_value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=flattened_value))

def __example_from_features(input_sw, teacher, target):
    features = {'inputs': __create_float_feature(input_sw.flatten()),
               'teacher': __create_float_feature(teacher.flatten()),
               'target': __create_float_feature(target.flatten())}
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def __create_and_save_1_fold_CrossValidation(name='Daniels_dataset_1', seeds=[42, 43]):
    # seeds that seem to work: [1,1]
    inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins = get_Daniels_data()
    normalizer_value = np.amax(ev_targets) - np.amin(ev_targets)
    print(normalizer_value)
    dataset_splitter_kwargs = {'inp':inp,
                              'target':pdf_targets,
                              'teacher':pdf_teacher,
                              'training_ratio':0.8,
                              'sample_spacing_in_mins':sample_spacing_in_mins,
                              'normalizer_value':normalizer_value,
                              'input_rate_in_1_per_min':5}

    dataset_splitter_kwargs['split_seeds'] = seeds
    dataset = __split_dataset(**dataset_splitter_kwargs)
    del inp, pdf_teacher, pdf_targets, ev_teacher, ev_targets

    dataset['input_shape'] = dataset['train_inputs'].shape[1:]
    dataset['teacher_shape'] = dataset['train_teacher'].shape[1:]
    dataset['target_shape'] = dataset['train_targets'].shape[1:]
    dataset['normalizer_value'] = normalizer_value

    __save_and_augment_Daniels_dataset(dataset, dataset_name=name, augment_data=False)
    print('done')



def __create_and_save_1_fold_CrossValidation_Liza(name='Lizas_dataset_2', seeds=[43, 42]):
    inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins = get_Lizas_data()
    normalizer_value = np.amax(ev_targets) - np.amin(ev_targets)
    dataset_splitter_kwargs = {'inp':inp,
                              'target':pdf_targets,
                              'teacher':pdf_teacher,
                              'training_ratio':0.8,
                              'sample_spacing_in_mins':sample_spacing_in_mins,
                              'normalizer_value':normalizer_value,
                              'input_rate_in_1_per_min':15}

    dataset_splitter_kwargs['split_seeds'] = seeds
    dataset = __split_dataset(**dataset_splitter_kwargs)
    dataset['normalizer_value'] = normalizer_value
    del inp, pdf_teacher, pdf_targets, ev_teacher, ev_targets

    dataset['input_shape'] = dataset['train_inputs'].shape[1:]
    dataset['teacher_shape'] = dataset['train_teacher'].shape[1:]
    dataset['target_shape'] = dataset['train_targets'].shape[1:]
    dataset['normalizer_value'] = normalizer_value

    # ToDo: Liza's gotta figure out how augmentation would work for her dataset
    __save_and_augment_Daniels_dataset(dataset, dataset_name=name, augment_data=False)
