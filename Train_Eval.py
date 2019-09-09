# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf
import numpy as np
#from Building_Blocks import build_model
from Benchmarks import build_model

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

def __delete_list_slice(list, start, len_slice):
    # for step in range(len_slice):
    #     del list[start]
    # return list

    obj = np.s_[start:(start+len_slice)]
    return np.delete(list, obj, axis=0) 

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
    inp = __delete_list_slice(inp, index_start, len_slice)
    inp_slice = tf.convert_to_tensor(inp_slice, dtype=tf.float32)

    teacher_slice = teacher[index_start:index_end,:,:]
    teacher = __delete_list_slice(teacher, index_start, len_slice)
    teacher_blend = [0] * teacher_slice.shape[0]
    teacher_slice = tf.convert_to_tensor(teacher_slice, dtype=tf.float32)
    teacher_blend = tf.convert_to_tensor(teacher_blend, dtype=tf.float32)

    slice_inputs = [inp_slice , teacher_slice, teacher_blend]

    target_slice = target[index_start:index_end,:,:]
    target_slice = tf.convert_to_tensor(target_slice, dtype=tf.float32)
    target = __delete_list_slice(target, index_start, len_slice)

    return slice_inputs, target_slice, inp, teacher, target

def __split_dataset(inp, target, teacher, training_ratio):
    if training_ratio > 1:
        print('... seems like you want more than a full training set, the training ratio needs to be smaller than 1!')

    remainder_for_test_val = 1.0-training_ratio
    test_len =  (remainder_for_test_val/2.0) * inp.shape[0]
    val_len = (remainder_for_test_val/2.0) * inp.shape[0]

    dataset = {}
    dataset['test_inputs'], dataset['test_targets'], inp, teacher, target = __slice_and_delete(inp, teacher, target, test_len, seed=59)
    dataset['val_inputs'], dataset['val_targets'], inp, teacher, target = __slice_and_delete(inp, teacher, target, val_len, seed=85)
    blend_train = [1] * inp.shape[0]
    dataset['train_inputs'] = [tf.convert_to_tensor(inp, dtype=tf.float32), tf.convert_to_tensor(teacher, dtype=tf.float32) ,tf.convert_to_tensor(blend_train, dtype=tf.float32)]
    dataset['train_targets'] = tf.convert_to_tensor(target, dtype=tf.float32)

    print('Dataset has', dataset['train_targets'].shape[0], 'training samples', dataset['val_targets'].shape[0], 'val samples', dataset['test_targets'].shape[0], 'test samples')
    tf.keras.backend.clear_session()
    return dataset

def __build_model(input_shape, out_shape):
    tf.keras.backend.clear_session()  # make sure we are working clean
    from Building_Blocks import decoder_LSTM_block, block_LSTM
    from Models import encoder_decoder_model, mimo_model

    decoder_model=decoder_LSTM_block(num_layers=3,
                                      num_units=128,
                                      use_dropout=True,
                                      dropout_rate=0.2,
                                      use_attention=True,
                                      attention_hidden=False)
    encoder_block = block_LSTM(num_layers=3,
                             num_units=128,
                             use_dropout=True,
                             dropout_rate=0.2,
                             use_norm=True,
                            only_last_layer_output=False)
    # # ToDo: would it be smarter to have this in the encoder decoder thingie instead of outside?
    projection_model=tf.keras.layers.Dense(units=out_shape[-1], activation=tf.keras.layers.Softmax(axis=-1))

    model = encoder_decoder_model(encoder_block=encoder_block,
                                  decoder_block=decoder_model,
                                  projection_block=projection_model,
                                  use_teacher=True,
                                  input_shape=in_shape,
                                  output_shape=out_shape,
                                  )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # set optimizers, metrics and loss

    from Losses_and_Metrics import loss_wrapper
    metrics = [loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nME'),
               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nRMSE'),
               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='KL-D')]
    model.compile(optimizer=optimizer,
                  loss=loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='tile-to-forecast'),
                  metrics=metrics) #compile, print summary
    model.summary()
    return model

from Dataset_Loaders import get_Daniels_data, get_Lizas_data
inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins = get_Daniels_data()
#inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher, sample_spacing_in_mins = get_Lizas_data()

out_shape = pdf_targets.shape[1:]
in_shape = inp.shape[1:]

dataset = __split_dataset(inp=inp, target=pdf_targets, teacher=pdf_teacher, training_ratio=0.6)
del inp, pdf_teacher, pdf_targets, ev_teacher, ev_targets
from Losses_and_Metrics import __calculatate_skillscore_baseline
val_skill_baseline = __calculatate_skillscore_baseline(dataset['val_targets'], sample_spacing_in_mins=sample_spacing_in_mins)
test_skill_baseline = __calculatate_skillscore_baseline(dataset['test_targets'], sample_spacing_in_mins=sample_spacing_in_mins)
train_skill_baseline = __calculatate_skillscore_baseline(dataset['train_targets'], sample_spacing_in_mins=sample_spacing_in_mins)
tf.keras.backend.clear_session()
print('The skill baselines are:', train_skill_baseline, ' for training', val_skill_baseline, ' for validation', test_skill_baseline, ' for testing')
relative_decrease = 0
epoch = 0
decrease = 0

best_val_loss = np.inf
prev_val_loss = np.inf
best_val_metric = np.inf

val_metrics = {}
train_metrics = {}

model = __build_model(input_shape=in_shape, out_shape=out_shape)

while decrease < 10:
    train_history = model.fit(x=dataset['train_inputs'], y=dataset['train_targets'],
                              batch_size=64,
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

    if best_val_metric > val_metrics['val_nRMSE'][-1]:  # if we see no increase in absolute performance, increase the death counter
        decrease = 0  # reset the death counter
        best_val_metric = val_metrics['val_nRMSE'][-1]
        best_wts = model.get_weights()
        print('saving a new model')
    else:
        decrease += 1

    if prev_val_loss < train_history.history['val_nRMSE'][0]:  # see if we are having a relative decrease
        relative_decrease += 1
    else:
        relative_decrease = 0

    if relative_decrease > 1:
        # if we have no relative increase in quality towards the previous iteration
        # then decrease the blend factor
        dataset['train_inputs'][2] = dataset['train_inputs'][2] - 0.05
        dataset['train_inputs'][2] = tf.maximum(0.0, dataset['train_inputs'][2] )
        print('lowering blend factor')
        relative_decrease = 0

    epoch += 1
    prev_val_loss = train_history.history['val_nRMSE'][0]

__plot_training_curves(train_metrics, val_metrics)

model.set_weights(best_wts)
test_results = model.evaluate(x=dataset['test_inputs'],
                              y=dataset['test_targets'],
                              batch_size=64,
                              verbose=False)
print('test results', test_results)





