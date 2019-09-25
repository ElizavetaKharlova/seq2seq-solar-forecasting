from Train_Eval import __build_model, __train_model, __plot_training_curves
import numpy as np
import pickle
from Dataset_Loaders import get_Lizas_data, get_Daniels_data, __split_dataset, __augment_Daniels_dataset
import tensorflow as tf
def train_on_Lisas_data(model_name):
    inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins = get_Lizas_data()
    normalizer_value = np.amax(ev_targets) - np.amin(ev_targets)
    dataset = __split_dataset(inp=inp, target=pdf_targets, teacher=pdf_teacher, training_ratio=0.6, sample_spacing_in_mins=sample_spacing_in_mins, normalizer_value=normalizer_value, input_rate_in_1_per_min=15)
    del inp, pdf_teacher, pdf_targets, ev_teacher, ev_targets

    model = __build_model(input_shape=dataset['train_inputs'].shape[1:],
                          out_shape=dataset['train_targets'].shape[1:],
                          model_type=model_name,
                          normalizer_value=normalizer_value)

    metrics = __train_model(model, batch_size=64, dataset=dataset)
    __plot_training_curves(metrics, experiment_name='Some_Experiment')

def train_one_3fold_on_Daniel_data(model_name):

    data_files = ['Daniels_dataset_1.pickle',
                  # 'Daniels_dataset_2.pickle',
                  # 'Daniels_dataset_3.pickle',
                  ]
    experiment_name = 'some_Experiment'
    metrics = []
    for set in data_files:
        with open(set, 'rb') as handle:
            dataset = pickle.load(handle)

        tf.keras.backend.clear_session()
        model = __build_model(input_shape=dataset['train_inputs'].shape[1:],
                              out_shape=dataset['train_targets'].shape[1:],
                              model_type=model_name,
                              normalizer_value=dataset['normalizer_value'])

        metrics.append(__train_model(model, batch_size=128, dataset=dataset))


        saved_epoch = np.argmax(metrics[-1]['val_nRMSE'])
        metrics[-1]['val_nRMSE_skill'] = 1 - (metrics[-1]['val_nRMSE'][saved_epoch]/dataset['val_persistency_baseline']['nRMSE'])
        metrics[-1]['val_nRMSE_skill'] = 1 - (
                    metrics[-1]['val_CRPS'][saved_epoch] / dataset['val_persistency_baseline']['CRPS'])
        metrics[-1]['val_nRMSE_skill'] = 1 - (
                    metrics[-1]['test_nRMSE']/ dataset['test_persistency_baseline']['nRMSE'])
        metrics[-1]['test_CRPS_skill'] = 1 - (
                    metrics[-1]['test_CRPS']/ dataset['test_persistency_baseline']['CRPS'])
        tf.keras.backend.clear_session()
        del dataset

    __plot_training_curves([metrics], experiment_name=experiment_name)

    with open(experiment_name+'.pickle', 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)