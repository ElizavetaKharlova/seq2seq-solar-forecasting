# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf
from tensorflow import keras

from Building_Blocks import build_model
from Losses_and_Metrics import loss_rME, loss_rMSE
from Dataset_loaders import get_Daniels_data, get_Lizas_data

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

inp_train, inp_test, ev_train, ev_test, ev_teacher_train, ev_teacher_test, blend_factor = get_Daniels_data()
# inp_train, inp_test, ev_train, ev_test, ev_teacher_train, ev_teacher_test, blend_factor = get_Lizas_data()

keras.backend.clear_session()  # make sure we are working clean

max_value, min_value = __get_max_min_targets(train_targets=ev_train, test_targets=ev_test)

E_D_layers = 4
E_D_units = 300
out_shape = ev_train.shape
in_shape = inp_train.shape
# ToDo: enable giving the builder a specific encoder and a specific decoder, that would be nicer maybe? dunno ...

model = build_model(E_D_layers, E_D_units, in_shape, out_shape) #get a E_D_model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) # set optimizers, metrics and loss
loss = loss_rMSE(max_value=max_value, min_value=min_value)
metrics = [loss_rME(max_value=max_value, min_value=min_value)]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics) #compile, print summary
model.summary()

import numpy as np
hist_loss = []
hist_val_loss = []
hist_metric = []
hist_val_metric = []
decrease = 0
relative_decrease = 0
epoch = 0
best_val_loss = 1e6
prev_val_loss = 1e6

sample_size = int(0.8 * inp_train.shape[0])
inp_val = inp_train[sample_size:, :, :]
inp_train_set = inp_train[:sample_size, :, :]

ev_teacher_val = ev_teacher_train[sample_size:, :, :]
ev_teacher_train_set = ev_teacher_train[:sample_size, :, :]

ev_train_val = ev_train[sample_size:, :, :]
ev_train_train_set = ev_train[:sample_size, :, :]

blend_factor_val = blend_factor[sample_size:, :]
blend_factor_val = np.zeros(blend_factor_val.shape)
blend_factor_train_set = blend_factor[:sample_size, :]
# ToDo: insert loop that adjusts the blend_factor from 1 to 0 depending on how the validation loss develops
# ToDo: make this nicer, maybe with a isTrainig flag or sth?
while decrease < 6:
    train_history = model.fit(x=[inp_train_set, ev_teacher_train_set, blend_factor_train_set],
                              # train for a given set of epochs, look at history
                              y=ev_train_train_set,
                              batch_size=64,
                              epochs=2,
                              validation_split=0.0)

    val_history = model.evaluate(x=[inp_val, ev_teacher_val, blend_factor_val],
                                 # train for a given set of epochs, look at history
                                 y=ev_train_val,
                                 batch_size=128)
    val_metric = val_history[0]
    val_loss = val_history[1]

    if best_val_loss > val_loss:  # if we see no increase in absolute performance, increase the death counter
        decrease = 0  # reset the death counter
        best_val_loss = val_loss
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
        blend_factor = blend_factor - 0.8
        print('lowering blend factor')
        relative_decrease = 0

    epoch += 1
    prev_val_loss = val_loss

    hist_loss.append(train_history.history['loss'])
    hist_val_loss.append(val_history[0])
    hist_metric.append(train_history.history['rME'])
    hist_val_metric.append(val_history[1])
model.set_weights(best_wts)




