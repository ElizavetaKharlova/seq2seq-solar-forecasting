import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from Train_Eval import dataset_generator_PV


def process_bins(signal):
    min = 0
    max = 1
    bins = 50

    Heigth_Rectangle = (max * (1 + 1 / ((bins - 1) * 2))) / (bins - 1)
    Width_rectangle = 1

    xi = np.arange(0, signal.shape[0])
    # xi = np.append(xi, (2 * xi[-1] - xi[-2]))

    yi = (Heigth_Rectangle) * np.arange(bins) - Heigth_Rectangle / 2 + min

    zi = signal
    return xi, yi, zi
def plot(sample):
    support_input = sample[0]['support_input']
    support_input = tf.squeeze(support_input, axis=0)
    support_input = support_input.numpy()
    support_input = np.delete(support_input, 1, axis=-1)

    history_input = sample[0]['history_input']
    history_input = tf.squeeze(history_input, axis=0)
    history_input = history_input.numpy()

    targets = sample[1]
    targets = tf.squeeze(targets, axis=0)
    targets = targets[-48:,:]
    targets_as_expected = calculate_ecpected_value(targets)

    #plotted targets
    cmap = plt.cm.Reds
    cmap.set_under(color='white')
    norm = mpl.colors.Normalize(vmin=1e-7, vmax=1)
    fig_targets, ax_targets = plt.subplots(1, 1)

    x_pdf, y_pdf, z_pdf = process_bins(targets.numpy())
    x_pdf = np.linspace(-24,24,targets_as_expected.shape[0]) -.3/24
    y_pdf = y_pdf - 0.25/50
    NN_Prediction = ax_targets.pcolormesh(x_pdf, y_pdf, np.swapaxes(z_pdf, 0, 1), cmap=cmap, alpha=1, edgecolors='None', norm=norm)
    cbar_NN_Prediction = plt.colorbar(NN_Prediction)
    cbar_NN_Prediction.set_label('Probabilities')
    ax_targets.plot(np.linspace(-24,24,targets_as_expected.shape[0]) , targets_as_expected.numpy(), alpha=0.5)
    ax_targets.plot(np.linspace(0, 24, targets_as_expected[:-24].shape[0]), targets_as_expected[:-24].numpy(), alpha=0.8, label='Persistent Forecast', color='green')
    ax_targets.set_xlabel('Hours')
    ax_targets.set_ylabel('Pv Power Output')
    ax_targets.legend()

    fig_sw, ax_sw = plt.subplots(support_input.shape[-1], 1, sharex=True)
    for dimension in range(support_input.shape[-1]):
        if dimension == 0:
            label = 'Temp'
        if dimension == 1:
            label = 'Solar Irradiation'
        if dimension == 2:
            label = 'Rel Humidity'
        if dimension == 3:
            label = 'Sea Level Pressure'
        if dimension == 4:
            label = 'Wind(cos)'
        if dimension == 5:
            label = 'Wind(sin)'

        ax_sw[dimension].plot(np.linspace(-5,0, support_input.shape[0]), support_input[:,dimension], label=label)
        ax_sw[dimension].legend(loc='upper right')

    ax_sw[-1].set_xlabel('SW Days')
    plt.show()

def calculate_ecpected_value(signal):
    num_bins = 50
    indices = tf.range(num_bins, dtype=tf.float32)  # (last_output_dim_size)
    expected_value = tf.multiply(signal, indices)  # (batches, timesteps, last_output_dim_size)
    expected_value = tf.reduce_sum(expected_value, axis=-1)
    return expected_value / num_bins

def load_sample(dataset_path='Daniels_Dataset_1'):
    support_shape = [int(6*24*60/15), 7]
    history_shape = [int(5*24), 50]
    target_shape = [int(1 * 24), 50]
    generator = dataset_generator_PV(dataset_path=dataset_path,
                              train_batch_size=1,
                              support_shape=support_shape,
                              history_shape=history_shape,
                              raw_history_shape=[15, support_shape[1]*15],
                              val_target_shape=target_shape,
                              )

    dataset = generator.pdf_generator_training_dataset()
    for sample in dataset.take(1):
        plot(sample)

load_sample()