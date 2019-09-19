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
def __plot_training_curves(train_dict, val_dict):

    for key in train_dict.keys():
        for val_key in val_dict.keys():
            if str(key) in str(val_key):
                plt.plot(train_dict[key], label=str(key))
                plt.plot(val_dict[val_key], label=str(val_key))
                plt.ylabel(str(key))
                plt.xlabel('number of Epochs')
                plt.title('Loss and Metrics Curves')
                plt.legend()
                plt.show()

def __slice_and_delete(inp, teacher, target, len_slice, seed):
    np.random.seed(seed)

    num_total_samples = inp.shape[0]
    len_slice = int(len_slice)
    index_start = np.random.uniform(low=0,
                                       high=num_total_samples - len_slice,
                                       size=None)
    index_start = int(np.floor(index_start))
    index_end = index_start + len_slice
    inp_slice = inp[index_start:index_end,:,:]
    inp_slice = tf.convert_to_tensor(inp_slice, dtype=tf.float32)

    teacher_slice = teacher[index_start:index_end,:,:]
    teacher_blend = [0] * teacher_slice.shape[0]
    teacher_slice = tf.convert_to_tensor(teacher_slice, dtype=tf.float32)
    teacher_blend = tf.convert_to_tensor(teacher_blend, dtype=tf.float32)

    slice_inputs = [inp_slice , teacher_slice, teacher_blend]

    target_slice = target[index_start:index_end,:,:]
    target_slice = tf.convert_to_tensor(target_slice, dtype=tf.float32)

    return slice_inputs, target_slice, np.delete(inp, np.s_[index_start:index_end], axis=0) , np.delete(teacher, np.s_[index_start:index_end], axis=0) , np.delete(target, np.s_[index_start:index_end], axis=0)

def __split_dataset(inp, target, teacher, training_ratio):
    if training_ratio > 1:
        print('... seems like you want more than a full training set, the training ratio needs to be smaller than 1!')

    remainder_for_test_val = 1.0-training_ratio
    test_len = (remainder_for_test_val/2.0) * inp.shape[0]
    val_len =(remainder_for_test_val/2.0) * inp.shape[0]
    print('provided', inp.shape[0], 'samples')
    dataset = {}
    dataset['test_inputs'], dataset['test_targets'], inp, teacher, target = __slice_and_delete(inp, teacher, target, test_len, seed=78)
    dataset['val_inputs'], dataset['val_targets'], inp, teacher, target = __slice_and_delete(inp, teacher, target, val_len, seed=87)
    blend_train = [1] * inp.shape[0]
    dataset['train_inputs'] = [tf.convert_to_tensor(inp, dtype=tf.float32), tf.convert_to_tensor(teacher, dtype=tf.float32) ,tf.convert_to_tensor(blend_train, dtype=tf.float32)]
    dataset['train_targets'] = tf.convert_to_tensor(target, dtype=tf.float32)

    print('Dataset has', dataset['train_targets'].shape[0], 'training samples', dataset['val_targets'].shape[0], 'val samples', dataset['test_targets'].shape[0], 'test samples')
    tf.keras.backend.clear_session()
    return dataset

def __get_persistency_baselines_for_dataset(dataset, sample_spacing_in_mins, normalizer_value):
    from Losses_and_Metrics import __calculatate_skillscore_baseline
    tf.keras.backend.clear_session()
    val_skill_baseline = __calculatate_skillscore_baseline(dataset['val_targets'],
                                                           sample_spacing_in_mins=sample_spacing_in_mins,
                                                           normalizer_value=normalizer_value)
    test_skill_baseline = __calculatate_skillscore_baseline(dataset['test_targets'],
                                                            sample_spacing_in_mins=sample_spacing_in_mins,
                                                            normalizer_value=normalizer_value)
    train_skill_baseline = __calculatate_skillscore_baseline(dataset['train_targets'],
                                                             sample_spacing_in_mins=sample_spacing_in_mins,
                                                             normalizer_value=normalizer_value)
    tf.keras.backend.clear_session()
    print('The skill baselines are:', train_skill_baseline, ' for training', val_skill_baseline, ' for validation',
          test_skill_baseline, ' for testing')

def __build_model(input_shape, out_shape, model_type='Encoder-Decoder', normalizer_value=1.0):
    tf.keras.backend.clear_session()  # make sure we are working clean
    from Building_Blocks import decoder_LSTM_block, block_LSTM
    from Benchmarks import DenseTCN
    from Models import encoder_decoder_model, mimo_model

    if model_type is 'MiMo':
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
                           downsample_input=True,
                           downsampling_rate=(60/5),
                            mode='snip')

    elif model_type == 'Encoder-Decoder':
        # common_specs = {'units': [[64,64,64,64]],
        #                 'use_dropout': True,
        #                 'dropout_rate': 0.15,
        #                 'use_norm': True,
        #                 'use_hw': True,
        #                 'use_quasi_dense': False,
        #                 'only_last_layer_output': True}
        #
        # encoder_specs = copy.deepcopy(common_specs)
        # decoder_specs = copy.deepcopy(common_specs)
        # decoder_specs['use_attention'] = True
        # encoder_block = block_LSTM(**encoder_specs)
        # decoder_model=decoder_LSTM_block(**decoder_specs)

        from Building_Blocks import Transformer_encoder, Transformer_decoder
        encoder_block = Transformer_encoder(num_layers=3,
                                            num_units_per_head_per_layer=[[40,40,40], [60,60,60], [80,80,80]],
                                            num_proj_units_per_layer=[80, 80, 80],
                                            ts_reduce_by_per_layer=[1,1,2],
                                            )
        decoder_block = Transformer_decoder(num_layers=3,
                                            num_units_per_head_per_layer=[[40,40,40], [60,60,60], [80,80,80]],
                                            num_proj_units_per_layer=[80, 80, 80],
                                            ts_reduce_by_per_layer=[1,1,1],
                                            )
        # # # ToDo: would it be smarter to have this in the encoder decoder thingie instead of outside?
        projection_model=tf.keras.layers.TimeDistributed(
                                        tf.keras.layers.Dense(units=out_shape[-1],
                                                            activation=tf.keras.layers.Softmax(axis=-1)))

        model = encoder_decoder_model(encoder_block=encoder_block,
                                      encoder_stateful=False,

                                      decoder_block=decoder_block,
                                      use_teacher=True,
                                      decoder_uses_attention=True,
                                      decoder_stateful=False,

                                      projection_block=projection_model,

                                      input_shape=input_shape,
                                      output_shape=out_shape)

    from Losses_and_Metrics import loss_wrapper
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.5*1e-3),
                  loss=loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='tile-to-forecast'),
                  metrics=[loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nME', normalizer_value=normalizer_value),
               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nRMSE', normalizer_value=normalizer_value),
               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='CRPS'),
               ]) #compile, print summary
    model.summary()
    return model

def __train_model(model):
    relative_decrease = 0
    decrease = 0

    best_val_metric = np.inf

    val_metrics = {}
    train_metrics = {}

    while decrease < 20:
        train_history = model.fit(x=dataset['train_inputs'], y=dataset['train_targets'],
                                  batch_size=128,
                                  epochs=1,
                                  shuffle=True,
                                  verbose=False,
                                  validation_data=(dataset['val_inputs'], dataset['val_targets']))
        print(train_history.history)
        for key in train_history.history.keys():
            if 'val' in str(key):
                if key in val_metrics:
                    val_metrics[key].append(train_history.history[key][0])
                else:
                    val_metrics[key] = [train_history.history[key][0]]
            else:
                if key in train_metrics:
                    train_metrics[key].append(train_history.history[key][0])
                else:
                    train_metrics[key] = [train_history.history[key][0]]

        if best_val_metric > val_metrics['val_nRMSE'][
            -1]:  # if we see no increase in absolute performance, increase the death counter
            decrease = 0  # reset the death counter
            best_val_metric = val_metrics['val_nRMSE'][-1]
            best_wts = model.get_weights()
            print('saving a new model')
        else:
            decrease += 1
            relative_decrease += 1

        if relative_decrease > 3:
            # if we have no relative increase in quality towards the previous iteration
            # then decrease the blend factor
            dataset['train_inputs'][2] = dataset['train_inputs'][2] - 0.05
            dataset['train_inputs'][2] = tf.maximum(0.0, dataset['train_inputs'][2])
            print('lowering blend factor')
            relative_decrease = 0

        prev_val_loss = train_history.history['val_nRMSE'][0]

    __plot_training_curves(train_metrics, val_metrics)

    model.set_weights(best_wts)
    test_results = model.evaluate(x=dataset['test_inputs'],
                                  y=dataset['test_targets'],
                                  batch_size=128,
                                  verbose=False)
    print('test results', test_results)

from Dataset_Loaders import get_Daniels_data, get_Lizas_data
inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins = get_Daniels_data()
# inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins = get_Lizas_data()
normalizer_value = np.amax(ev_targets) - np.amin(ev_targets)
# print('Ev normalizer value', normalizer_value)

dataset = __split_dataset(inp=inp, target=pdf_targets, teacher=pdf_teacher, training_ratio=0.6)
del inp, pdf_teacher, pdf_targets, ev_teacher, ev_targets
__get_persistency_baselines_for_dataset(dataset=dataset, sample_spacing_in_mins=sample_spacing_in_mins, normalizer_value=normalizer_value)


model = __build_model(input_shape=dataset['train_inputs'][0].shape[1:],
                      out_shape=dataset['val_targets'].shape[1:],
                      model_type='Encoder-Decoder',
                      normalizer_value=normalizer_value)
__train_model(model)





