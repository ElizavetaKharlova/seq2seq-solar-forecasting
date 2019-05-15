import numpy as np
from sklearn.preprocessing import scale
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



    data[:, :-1] = scale(data[:, :-1], axis=0, with_mean=True, with_std=True, copy=True) #scale the dataset
    if plot: #plot if we want to
        __plot_data_array(data, label='Scaled Data Set')

    target_variable = data[:, target_dims] #extract target vars

    if not isinstance(target_dims, int): #get min_max
        target_min_max = __get_min_max(__reconstruct(target_variable)) # reconstruct into one array, get min_max
    else:
        target_min_max = __get_min_max(target_variable)

    # create the list for the arrays
    pdf_targets = []    # pdf forecast
    ev_targets = []     # expected value forecast
    sw_inputs = []      # inputs

    for timestep in range(0, data.shape[0] - fc_len_samples - sw_len_samples, steps_to_new_sample):
        sliced_sample = data[timestep:(timestep + fc_len_samples + sw_len_samples), :]
        sliced_input = sliced_sample[:-fc_len_samples, :]
        sliced_target = sliced_sample[-fc_len_samples:, target_dims]

        if not isinstance(target_dims, int):
            sliced_target = __reconstruct(sliced_target)

        if np.sum(sliced_input[:, -1]) == 0:  # see if we are free of faults

            sw_inputs.append(
                sliced_input[:, :-1])  # drop the last dimension, since we do not need the fault indicator anymore

            pdf_target = __convert_to_pdf(sliced_target,
                                          num_steps=fc_steps, num_tiles=fc_tiles,
                                        min_max=target_min_max)  # pdf for probability distribution thingie
            pdf_targets.append(pdf_target)

            ev_target = __convert_to_mv(sliced_target, num_steps=fc_steps)  # ev for expected value
            ev_targets.append(ev_target)

        if (int(timestep/steps_to_new_sample) % int(int(data.shape[0] / steps_to_new_sample)/10)) == 0:
            print(int(100*timestep/data.shape[0]), 'percent converted')

    return np.array(sw_inputs), np.array(ev_targets), np.array(pdf_targets)

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

