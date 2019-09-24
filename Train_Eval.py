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

def __slice_and_delete(inp, teacher, target, len_slice, seed, sample_spacing_in_mins):
    np.random.seed(seed)
    inp_sw_shape = inp.shape
    one_week_in_samples = (inp_sw_shape[1]*5)/sample_spacing_in_mins
    one_week_in_samples = int(one_week_in_samples)

    leftover_samples = int(len_slice)

    inp_separated = None
    teacher_separated = None
    teacher_blend_separated = None
    target_separated = None

    while leftover_samples > 0:
        one_week_in_samples = min(one_week_in_samples, leftover_samples)

        index_start = np.random.uniform(low=0, high=inp.shape[0] - 2*one_week_in_samples, size=None)
        index_start = int(np.floor(index_start))
        index_end = index_start + one_week_in_samples

        inp_slice = inp[index_start:index_end,:,:]
        inp_slice = tf.convert_to_tensor(inp_slice, dtype=tf.float32)
        inp = np.delete(inp, np.s_[index_start:index_end], axis=0)
        if inp_separated is None:
            inp_separated = inp_slice
        else:
            inp_separated = tf.concat([inp_separated, inp_slice], axis=0)

        teacher_slice = teacher[index_start:index_end,:,:]
        teacher_slice = tf.convert_to_tensor(teacher_slice, dtype=tf.float32)
        teacher = np.delete(teacher, np.s_[index_start:index_end], axis=0)
        if teacher_separated is None:
            teacher_separated = teacher_slice
        else:
            teacher_separated = tf.concat([teacher_separated, teacher_slice], axis=0)

        teacher_blend = [0] * teacher_slice.shape[0]
        teacher_blend = tf.convert_to_tensor(teacher_blend, dtype=tf.float32)
        if teacher_blend_separated is None:
            teacher_blend_separated = teacher_blend
        else:
            teacher_blend_separated = tf.concat([teacher_blend_separated, teacher_blend], axis=0)

        target_slice = target[index_start:index_end,:,:]
        target = np.delete(target, np.s_[index_start:index_end], axis=0)
        target_slice = tf.convert_to_tensor(target_slice, dtype=tf.float32)
        if target_separated is None:
            target_separated = target_slice
        else:
            target_separated = tf.concat([target_separated, target_slice], axis=0)

        leftover_samples -= one_week_in_samples

    slice_inputs = [inp_separated , teacher_separated, teacher_blend_separated]
    return slice_inputs, target_separated, inp , teacher , target

def __split_dataset(inp, target, teacher, training_ratio, sample_spacing_in_mins, normalizer_value):
    if training_ratio > 1:
        print('... seems like you want more than a full training set, the training ratio needs to be smaller than 1!')

    remainder_for_test_val = 1.0-training_ratio
    test_len = (remainder_for_test_val/2.0) * inp.shape[0]
    val_len = (remainder_for_test_val/2.0) * inp.shape[0]

    #ToDO: fix persistency baseline
    from Losses_and_Metrics import __calculatate_skillscore_baseline
    persistency_baseline = __calculatate_skillscore_baseline(target,
                                                           sample_spacing_in_mins=sample_spacing_in_mins,
                                                           normalizer_value=normalizer_value)
    print('Persistency baseline for whole Dataset, becuase I need to reimplement some of this ... ', persistency_baseline)

    dataset = {}
    dataset['test_inputs'], dataset['test_targets'], inp, teacher, target = __slice_and_delete(inp, teacher, target, test_len, seed=99, sample_spacing_in_mins=sample_spacing_in_mins)
    dataset['val_inputs'], dataset['val_targets'], inp, teacher, target = __slice_and_delete(inp, teacher, target, val_len, seed=420, sample_spacing_in_mins=sample_spacing_in_mins)
    blend_train = [1] * inp.shape[0]
    dataset['train_inputs'] = [tf.convert_to_tensor(inp, dtype=tf.float32), tf.convert_to_tensor(teacher, dtype=tf.float32) , tf.convert_to_tensor(blend_train, dtype=tf.float32)]
    dataset['train_targets'] = tf.convert_to_tensor(target, dtype=tf.float32)

    print('Dataset has', dataset['train_targets'].shape[0], 'training samples', dataset['val_targets'].shape[0], 'val samples', dataset['test_targets'].shape[0], 'test samples')
    tf.keras.backend.clear_session()
    return dataset

def __augment_dataset(dataset):
    if 'val_inputs' in dataset:
        print('adjusting val inputs', dataset['val_inputs'][0].shape)
        dataset['val_inputs'][0] = dataset['val_inputs'][0].numpy()

        corrected_shape = dataset['val_inputs'][0].shape
        new_values = np.zeros(shape=[corrected_shape[0], corrected_shape[1], corrected_shape[-1] - 5 + 1])
        for sample in range(len(dataset['val_inputs'][0])):
            sample_inputs = dataset['val_inputs'][0][sample]
            pv_sample = sample_inputs[:, 0:5]
            rest_sample = sample_inputs[:, (pv_sample.shape[-1]):]
            pv_sample = tf.reduce_mean(pv_sample, axis=-1)
            new_values[sample,:,0] = pv_sample.numpy()
            new_values[sample,:,1:] = rest_sample

        dataset['val_inputs'][0] = tf.convert_to_tensor(new_values, dtype=tf.float32)
        print('val inputs are now: ', dataset['val_inputs'][0].shape)

    if 'test_inputs' in dataset:
        print('adjusting test_inputs', dataset['test_inputs'][0].shape)
        dataset['test_inputs'][0] = dataset['test_inputs'][0].numpy()

        corrected_shape = dataset['test_inputs'][0].shape
        new_values = np.zeros(shape=[corrected_shape[0], corrected_shape[1], corrected_shape[-1] - 5 + 1])
        for sample in range(len(dataset['test_inputs'][0])):
            sample_inputs = dataset['test_inputs'][0][sample]
            pv_sample = sample_inputs[:, 0:5]
            rest_sample = sample_inputs[:, (pv_sample.shape[-1]):]
            pv_sample = tf.reduce_mean(pv_sample, axis=-1)
            new_values[sample,:,0] = pv_sample.numpy()
            new_values[sample,:,1:] = rest_sample

        dataset['test_inputs'][0] = tf.convert_to_tensor(new_values, dtype=tf.float32)
        print('test_inputs are now: ', dataset['test_inputs'][0].shape)

    if 'train_inputs' in dataset:
        print('adjusting train_inputs', dataset['train_inputs'][0].shape)

        dataset['train_inputs'][0] = dataset['train_inputs'][0].numpy()

        original_teacher = dataset['train_inputs'][1].numpy()
        original_blend = dataset['train_inputs'][2].numpy()
        original_targets = dataset['train_targets'].numpy()

        original_inputs = dataset['train_inputs'][0].shape
        new_values = np.zeros(shape=[original_inputs[0]*5, original_inputs[1], original_inputs[-1] - 5 + 1])
        new_teacher = np.zeros(shape=[original_teacher.shape[0]*5, original_teacher.shape[1], original_teacher.shape[2]])
        new_blend = np.zeros(
            shape=[original_blend.shape[0] * 5])
        new_targets = np.zeros(
            shape=[original_targets.shape[0] * 5, original_targets.shape[1], original_targets.shape[2]])

        sample_inputs = dataset['train_inputs'][0]
        pv = sample_inputs[:,:, 0:5]
        rest = sample_inputs[:,:, pv.shape[-1]:]
        num_sets = pv.shape[-1]
        num_original_samples = dataset['train_inputs'][0].shape[0]
        for pv_set in range(num_sets):
            pv_sample = pv[:,:,pv_set]
            augmented_features = rest
            feature_deltas = np.subtract(augmented_features[:,1:,:], augmented_features[:,:-1,:])

            for sample in range(num_original_samples):
                offset_bias = np.random.randint(low=1, high=num_sets)
                for feature in range(augmented_features.shape[-1]):
                    offsets = feature_deltas[sample, :, feature]
                    # offset_noise = np.random.uniform(low=0.0, high=1.0, size=offsets.shape)
                    offset_noise = 0.0
                    offset_noise = np.multiply(offsets/(num_sets+1), offset_noise)
                    offsets = np.add(offsets*offset_bias/(num_sets+1), offset_noise)
                    augmented_features[sample,:-1,feature] = np.add(augmented_features[sample,:-1,feature], offsets)

                    new_values[num_original_samples*pv_set + sample,:,0] = pv_sample[sample,:]
                    new_values[num_original_samples*pv_set + sample,:,1:] = augmented_features[sample,:,:]
                    new_teacher[num_original_samples*pv_set + sample,:,:] = original_teacher[sample,:,:]
                    new_blend[num_original_samples * pv_set + sample] = original_blend[sample]
                    new_targets[num_original_samples * pv_set + sample, :, :] = original_targets[sample, :, :]

        dataset['train_inputs'][0] = tf.convert_to_tensor(new_values, dtype=tf.float32)
        dataset['train_inputs'][1] = tf.convert_to_tensor(new_teacher, dtype=tf.float32)
        dataset['train_inputs'][2] = tf.convert_to_tensor(new_blend, dtype=tf.float32)
        dataset['train_targets'] = tf.convert_to_tensor(new_targets, dtype=tf.float32)
        print('train_inputs are now: ', dataset['train_inputs'][0].shape)

    return dataset

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
        common_specs = {'units': [[96, 96, 96]],
                        'use_dropout': True,
                        'dropout_rate': 0.2,
                        'use_norm': True,
                        'use_hw': True,
                        'use_quasi_dense': False,
                        'only_last_layer_output': True}

        encoder_specs = copy.deepcopy(common_specs)
        decoder_specs = copy.deepcopy(common_specs)
        decoder_specs['use_attention'] = True
        decoder_specs['self_recurrent'] = True
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

def __train_model(model, batch_size):
    relative_decrease = 0
    decrease = 0

    best_val_metric = np.inf

    val_metrics = {}
    train_metrics = {}

    while decrease < 20:
        train_history = model.fit(x=dataset['train_inputs'], y=dataset['train_targets'],
                                  batch_size=batch_size,
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

        if relative_decrease > 4:
            # if we have no relative increase in quality towards the previous iteration
            # then decrease the blend factor
            dataset['train_inputs'][2] = dataset['train_inputs'][2] - 0.05
            dataset['train_inputs'][2] = tf.maximum(0.0, dataset['train_inputs'][2])
            print('lowering blend factor')
            relative_decrease = 0

        prev_val_loss = train_history.history['val_nRMSE'][0]

    __plot_training_curves(train_metrics, val_metrics)

    model.set_weights(best_wts)
    test_results = model.evaluate(x=dataset['val_inputs'],
                                  y=dataset['val_targets'],
                                  batch_size=batch_size,
                                  verbose=False)
    print('test results', test_results)

from Dataset_Loaders import get_Daniels_data, get_Lizas_data
inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins = get_Daniels_data()
# inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins = get_Lizas_data()
normalizer_value = np.amax(ev_targets) - np.amin(ev_targets)
# print('Ev normalizer value', normalizer_value)

dataset = __split_dataset(inp=inp, target=pdf_targets, teacher=pdf_teacher, training_ratio=0.6, sample_spacing_in_mins=sample_spacing_in_mins, normalizer_value=normalizer_value)
del inp, pdf_teacher, pdf_targets, ev_teacher, ev_targets
# Only Daniel
# dataset = __augment_dataset(dataset)

model = __build_model(input_shape=dataset['train_inputs'][0].shape[1:],
                      out_shape=dataset['train_targets'].shape[1:],
                      model_type='Encoder-Decoder',
                      normalizer_value=normalizer_value)
__train_model(model, batch_size=64)





