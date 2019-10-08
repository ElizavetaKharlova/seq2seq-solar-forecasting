import numpy as np
import pickle
import matplotlib.pyplot as plt
from Train_Eval import Model_Container
import tensorflow as tf

def train_LSTM_baseline_3fold_on_Daniel_data():

    datasets = ['Daniels_dataset_1', 'Daniels_dataset_2', 'Daniels_dataset_3', ]
    metrics = {}
    for set in datasets:
        metrics[set] = []
        for run in range(3):
            model_kwargs = {'model_type': 'MiMo-attn-tcn',
                            'model_size': 'med',
                            'use_attention': True,
                            'attention_hidden': False,
                            'self_recurrent': False,
                            'use_dropout': True,
                            'dropout_rate': 0.3,
                            'use_norm': False}
            train_kwargs = {'batch_size': 256}

            experiment = Model_Container(dataset_folder=set,
                                      model_kwargs=model_kwargs,
                                      train_kwargs=train_kwargs,
                                      try_distribution_across_GPUs=False,)
            metrics[set].append(experiment.get_results())
            del experiment
            tf.keras.backend.clear_session()

    experiment_name = model_kwargs['model_size'] + '_' + model_kwargs['model_type']
    with open(experiment_name+'.pickle', 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    __plot_training_curves(metrics, experiment_name=experiment_name)

def train_on_Lizas_data():

    # ToDo: Gotta recompile ur datasets obviously
    data_files = ['Lizas_dataset_2',]
    metrics = {}
    for set in data_files:
        metrics[set] = []
        for run in range(2):
            model_kwargs = {'model_type': 'MiMo-attn-tcn',
                            'model_size':'small',}
            train_kwargs = {'batch_size':512}

            experiment = Model_Container(dataset_folder=set,
                                         model_kwargs=model_kwargs,
                                         train_kwargs=train_kwargs,
                                         try_distribution_across_GPUs=False, )
            metrics[set].append(experiment.get_results())
            del experiment
            tf.keras.backend.clear_session()

    experiment_name = model_kwargs['model_size'] + '_' + model_kwargs['model_type']
    with open(experiment_name+'.pickle', 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    __plot_training_curves(metrics, experiment_name=experiment_name)


def __plot_training_curves(metrics, experiment_name):

    val_color = [255 / 255, 165 / 255, 0 / 255]
    test_color = 'g'
    train_color = 'b'

    val_metrics = {}
    test_metrics = {}
    train_metrics = {}

    run_linestyles = [':', '--', '-.', '-']

    for set in metrics.keys():
        for key in metrics[set][0].keys():
            if 'val' in str(key):
                val_metrics[key] = [run[key] for run in metrics[set]]

            if 'test' in str(key):
                test_metrics[key] = [run[key] for run in metrics[set]]

            elif 'val' not in str(key) and 'test' not in str(key):
                train_metrics[key] = [run[key] for run in metrics[set]]

        for train_key in train_metrics.keys():
            for run in range(len(train_metrics[train_key])):
                plt.plot(train_metrics[train_key][run],
                         label=str(train_key),
                         color=train_color,
                         linestyle=run_linestyles[run])

                saved_val_index = int(np.argmin(val_metrics['val_nRMSE'][run]))
                for val_key in val_metrics.keys():
                    if str(train_key) in str(val_key) and 'skill' not in str(val_key):
                        plt.plot(val_metrics[val_key][run],
                                 label=str(val_key) + '_run'+ str(run),
                                 color=val_color,
                                 linestyle=run_linestyles[run])
                        plt.plot(saved_val_index, val_metrics[val_key][run][saved_val_index],
                                 label='chosen' + str(val_key) + '_run' + str(run),
                                 color=val_color,
                                 marker='o', markersize=4,
                                 linestyle=run_linestyles[run])

                        if np.argmax(val_metrics[val_key][run]) != saved_val_index:
                            best_val_index = int(np.argmin(val_metrics[val_key][run]))
                            best_val = np.amin(val_metrics[val_key][run])
                            plt.plot(best_val_index, best_val,
                                     label='bext' + str(val_key) + '_run' + str(run),
                                     color=val_color,
                                     marker='*', markersize=4,
                                     linestyle=run_linestyles[run])

                for test_key in test_metrics.keys():
                    if str(train_key) in str(test_key) and 'skill' not in str(test_key):
                        plt.plot(saved_val_index, test_metrics[test_key][run],
                                 label=str(test_key) + '_run'+ str(run),
                                    marker='o', markersize=4,
                                 color=test_color,
                                 linestyle=run_linestyles[run])

            plt.title(str(len(train_metrics[train_key])) + 'for dataset ' + str(set))
            plt.ylabel(str(train_key))
            plt.xlabel('number of Epochs')
            plt.grid()
            plt.legend()
            plt.savefig(experiment_name, dpi=500, format='pdf')
            plt.show()