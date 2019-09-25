# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf
import numpy as np
#from Building_Blocks import build_model
from Benchmarks import build_model
import copy

def __get_max_min_targets(train_targets, test_targets):
    import numpy as np
    max_value_train = np.amax(train_targets, axis=0)
    max_value_train = np.amax(max_value_train, axis=0)
    max_value_test = np.amax(test_targets, axis=0)
    max_value_test = np.amax(max_value_test, axis=0)
    max_value = np.maximum(max_value_train, max_value_test)

    min_value_train = np.amin(train_targets, axis=0)
    min_value_train = np.amin(min_value_train, axis=0)
    min_value_test = np.amin(test_targets, axis=0)
    min_value_test = np.amin(min_value_test, axis=0)
    min_value = np.minimum(min_value_test, min_value_train)
    return max_value, min_value

import matplotlib.pyplot as plt

def __plot_training_curves(metrics, experiment_name):
    val_metrics = {}
    test_metrics = {}
    train_metrics = {}

    metrics_dict = metrics[0]
    k_fold_mode = True
    for key in metrics_dict[0].keys():
        print(metrics)
        if 'val' in str(key):
            val_metrics[key] = [metrics[index][key] for index in range(len(metrics))]
        if 'test' in str(key):
            test_metrics[key] = [metrics[index][key] for index in range(len(metrics))]
        else:
            train_metrics[key] = [metrics[index][key] for index in range(len(metrics))]

    val_color = [255, 165, 0]
    test_color = 'g'
    train_color = 'b'


    for train_key in train_metrics.keys():
        if k_fold_mode:
            for set in range(len(train_metrics[train_key])):
                plt.plot(train_metrics[set][key],
                         label=str(train_key),
                         color=train_color)

                saved_val_index = np.amax(val_metrics[set]['val_nRMSE'])
                for val_key in val_metrics.keys():
                    if str(train_key) in str(val_key):
                        plt.plot(val_metrics[set][val_key],
                                 label=str(val_key) + str(set),
                                 color=val_color)
                        plt.plot(x=saved_val_index, y=val_metrics[set][val_key][saved_val_index],
                                 label='chosen' + str(val_key) + 'set' + str(set),
                                 color=val_color,
                                 marker='o', markersize=2)

                        if np.argmax(val_metrics[set][val_key]) != saved_val_index:
                            best_val_index = np.argmax(val_metrics[set][val_key])
                            best_val = np.amax(val_metrics[set][val_key])
                            plt.plot(x=best_val_index, y=best_val,
                                     label='bext' + str(val_key) + 'set' + str(set),
                                     color=val_color,
                                     marker='*', markersize=2)

                for test_key in test_metrics.keys():
                    if str(train_key) in str(test_key):
                        plt.plot(x=saved_val_index, y=test_metrics[set][test_key],
                                 label=str(test_key), marker='o', markersize=2,
                                 color=test_color)

        plt.ylabel(str(key))
        plt.xlabel('number of Epochs')
        plt.grid()
        plt.legend()

        plt.savefig(experiment_name, dpi=500, format='pdf')

        plt.show()

def __build_model(input_shape, out_shape, model_type='Encoder-Decoder', normalizer_value=1.0):
    tf.keras.backend.clear_session()  # make sure we are working clean
    from Building_Blocks import decoder_LSTM_block, block_LSTM
    from Benchmarks import DenseTCN
    from Models import encoder_decoder_model, mimo_model

    if model_type == 'MiMo-LSTM-downsample':
        encoder_specs = {'units': [[96, 96, 96]],
                        'use_dropout': False,
                        'dropout_rate': 0.15,
                        'use_norm': False,
                        'use_hw': False,
                        'return_state': False,
                        'use_quasi_dense': False,
                        'only_last_layer_output': True}
                            
        model = mimo_model(function_block=block_LSTM(**encoder_specs),
                            input_shape=input_shape,
                            output_shape=out_shape,
                           downsample_input=True,
                           downsampling_rate=(60/5),
                            mode='snip')

    elif model_type == 'MiMo-LSTM-nodownsample':
        encoder_specs = {'units': [[96, 96, 96]],
                         'use_dropout': True,
                         'dropout_rate': 0.15,
                         'use_norm': True,
                         'use_hw': False,
                         'return_state': False,
                         'use_quasi_dense': False,
                         'only_last_layer_output': True}

        model = mimo_model(function_block=block_LSTM(**encoder_specs),
                           input_shape=input_shape,
                           output_shape=out_shape,
                           downsample_input=False,
                           downsampling_rate=(60 / 5),
                           mode='snip')

    elif model_type == 'Encoder-Decoder':
        common_specs = {'units': [[96, 96, 96]],
                        'use_dropout': True,
                        'dropout_rate': 0.2,
                        'use_norm': True,
                        'use_hw': False,
                        'use_quasi_dense': False,
                        'only_last_layer_output': True}

        encoder_specs = copy.deepcopy(common_specs)
        decoder_specs = copy.deepcopy(common_specs)
        decoder_specs['use_attention'] = False
        decoder_specs['self_recurrent'] = False
        encoder_block = block_LSTM(**encoder_specs)
        decoder_block = decoder_LSTM_block(**decoder_specs)

        projection_model= tf.keras.layers.Dense(units=out_shape[-1],
                                                            activation=tf.keras.layers.Softmax(axis=-1))
        projection_model = tf.keras.layers.TimeDistributed(projection_model)
        model_kwargs = {'encoder_block':encoder_block,
                          "encoder_stateful":True,
                          'decoder_block':decoder_block,
                          'use_teacher':True,
                          'decoder_uses_attention_on': decoder_specs['use_attention'],
                          'decoder_stateful':True,
                          'self_recurrent_decoder': decoder_specs['self_recurrent'],
                          'projection_block': projection_model,
                          'input_shape': input_shape,
                          'output_shape': out_shape}
        model = encoder_decoder_model(**model_kwargs)

    from Losses_and_Metrics import loss_wrapper
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='tile-to-forecast'),
                  metrics=[loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nME', normalizer_value=normalizer_value),
               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nRMSE', normalizer_value=normalizer_value),
               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='CRPS'),
               ]) #compile, print summary
    model.summary()
    return model

def __train_model(model, batch_size, dataset):
    relative_decrease = 0
    decrease = 0
    best_val_metric = np.inf
    prev_val_metric = np.inf

    metrics = {}
    while decrease < 20:
        train_history = model.fit(x=[dataset['train_inputs'], dataset['train_teacher'], dataset['train_blend']],
                                  y=dataset['train_targets'],
                                  batch_size=batch_size,
                                  epochs=1,
                                  shuffle=True,
                                  verbose=False,
                                  validation_data=([dataset['val_inputs'], dataset['val_teacher'], dataset['val_blend']],
                                                   dataset['val_targets']))
        print(train_history.history)
        for key in train_history.history.keys():
            if 'val' in str(key):
                if key in metrics:
                    metrics[key].append(train_history.history[key][0])
                else:
                    metrics[key] = [train_history.history[key][0]]
            else:
                if key in metrics:
                    metrics[key].append(train_history.history[key][0])
                else:
                    metrics[key] = [train_history.history[key][0]]

        if best_val_metric > metrics['val_nRMSE'][
            -1]:  # if we see no increase in absolute performance, increase the death counter
            decrease = 0  # reset the death counter
            best_val_metric = metrics['val_nRMSE'][-1]
            best_wts = model.get_weights()
            print('saving a new model')
        else:
            decrease += 1

        if prev_val_metric < metrics['val_nRMSE'][-1]:
            relative_decrease += 1

        if relative_decrease > 2:
            # if we have no relative increase in quality towards the previous iteration
            # then decrease the blend factor
            dataset['train_blend'] = dataset['train_blend'] - 0.05
            dataset['train_blend'] = tf.maximum(0.0, dataset['train_blend'])
            print('lowering blend factor')
            relative_decrease = 0

        prev_val_metric = train_history.history['val_nRMSE'][0]

    model.set_weights(best_wts)
    test_results = model.evaluate(x=[dataset['test_inputs'], dataset['test_teacher'], dataset['test_blend']],
                                  y=dataset['test_targets'],
                                  batch_size=batch_size,
                                  verbose=False)
    metrics['test_loss'] = test_results[0]
    metrics['test_nME'] = test_results[1]
    metrics['test_nRMSE'] = test_results[2]
    metrics['test_CRPS'] = test_results[3]

    return metrics









