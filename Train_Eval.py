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
                plt.xlabel('number of Epochs')
                plt.show()

def __delete_list_slice(list, start, len_slice):
    print('deleting from ', len(list), 'from ', start, 'to ', start+len_slice)
    for step in range(len_slice):
        del list[start]
    return list

def __slice_and_delete(inp, teacher, target, len_slice, seed):
    np.random.seed(seed)

    num_total_samples = len(inp)
    len_slice = int(len_slice)
    index_start = np.random.uniform(low=0,
                                       high=num_total_samples - len_slice,
                                       size=None)
    index_start = int(np.floor(index_start))
    index_end = index_start + len_slice

    inp_slice = inp[index_start:index_end][:][:]
    inp = __delete_list_slice(inp, index_start, len_slice)

    teacher_slice = teacher[index_start:index_end][:][:]
    teacher = __delete_list_slice(teacher, index_start, len_slice)

    slice_inputs = [np.array(inp_slice) , np.array(teacher_slice), np.array([0] * len(inp_slice))]

    target_slice = target[index_start:index_end][:][:]
    target = __delete_list_slice(target, index_start, len_slice)

    return slice_inputs, np.array(target_slice), np.array(inp), np.array(teacher), np.array(target)

def __split_dataset(inp, target, teacher, training_ratio):
    if training_ratio > 1:
        print('... seems like you want more than a full training set, the training ratio needs to be smaller than 1!')
    inp = inp.tolist()
    target = target.tolist()
    teacher = teacher.tolist()

    remainder_for_test_val = 1.0-training_ratio
    test_len =  (remainder_for_test_val/2.0) * len(inp)
    val_len = (remainder_for_test_val/2.0) * len(inp)

    dataset = {}
    dataset['test_inputs'], dataset['test_targets'], inp, teacher, target = __slice_and_delete(inp, teacher, target, test_len, seed=42)
    print(len(inp))
    dataset['val_inputs'], dataset['val_targets'], inp, teacher, target = __slice_and_delete(inp.tolist(), teacher.tolist(), target.tolist(), val_len, seed=23)
    dataset['train_inputs'] = [inp, teacher, np.array([1]*inp.shape[0])]
    dataset['train_targets'] = target

    print('Dataset has', dataset['train_targets'].shape[0], 'training samples', dataset['val_targets'].shape[0], 'val samples', dataset['test_targets'].shape[0], 'test samples')

    return dataset

def __build_model():
    tf.keras.backend.clear_session()  # make sure we are working clean
    from Building_Blocks import decoder_LSTM_block, block_LSTM
    from Models import encoder_decoder_model, mimo_model
    decoder_model=decoder_LSTM_block(num_layers=3,
                                      num_units=64,
                                      use_dropout=True,
                                      dropout_rate=0.2,
                                      use_attention=True,
                                      attention_hidden=False)
    encoder_block = block_LSTM(num_layers=3,
                             num_units=64,
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
    metrics = [loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='nME'),
               loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='nMSE'),
               loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='KL-D')]
    model.compile(optimizer=optimizer,
                  loss=loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='tile-to-forecast'),
                  metrics=metrics) #compile, print summary
    model.summary()
    return model

from Dataset_Loaders import get_Daniels_data, get_Lizas_data
inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher = get_Daniels_data()

# inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher = get_Lizas_data()

out_shape = pdf_targets.shape[1:]
in_shape = inp.shape[1:]

dataset = __split_dataset(inp=inp, target=pdf_targets, teacher=pdf_teacher, training_ratio=0.6)

relative_decrease = 0
epoch = 0
decrease = 0

best_val_loss = np.inf
prev_val_loss = np.inf
best_val_metric = np.inf

val_metrics = {}
train_metrics = {}

model = __build_model()


while decrease < 10:
    train_history = model.fit(x=dataset['train_inputs'],
                              # train for a given set of epochs, look at history
                              y=dataset['train_targets'],
                              batch_size=64,
                              epochs=1,
                              shuffle=True,
                              validation_data=(dataset['val_inputs'], dataset['val_targets']))

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

    val_loss = train_history.history['val_loss'][0]

    if best_val_metric > val_metrics['val_loss_nMSE'][-1]:  # if we see no increase in absolute performance, increase the death counter
        decrease = 0  # reset the death counter
        best_val_metric = val_metrics['val_loss_nMSE'][-1]
        best_wts = model.get_weights()
        print('saving a new model')
    else:
        decrease += 1

    if prev_val_loss < val_loss:  # see if we are having a relative decrease
        relative_decrease += 1
    else:
        relative_decrease = 0

    if relative_decrease > 1:
        # if we have no relative increase in quality towards the previous iteration
        # then decrease the blend factor
        blend_factor_train = blend_factor_train - 0.1
        blend_factor_train = np.maximum(0.0, blend_factor_train)
        print('lowering blend factor')
        relative_decrease = 0

    epoch += 1
    prev_val_loss = val_loss

__plot_training_curves(train_metrics, val_metrics)

model.set_weights(best_wts)
test_results = model.evaluate(x=dataset['test_inputs'],
                              y=dataset['test_targets'],
                              batch_size=64,
                              verbose=1)





