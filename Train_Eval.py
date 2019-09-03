# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf

#from Building_Blocks import build_model
from Benchmarks import build_model
from Dataset_Loaders import get_Daniels_data, get_Lizas_data

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

inp_train, inp_test, ev_train, ev_test, ev_teacher_train, ev_teacher_test, blend_factor, pdf_train, pdf_test, pdf_teacher_train, pdf_teacher_test = get_Daniels_data()

# inp_train, inp_test, ev_train, ev_test, ev_teacher_train, ev_teacher_test, blend_factor, pdf_train, pdf_test, pdf_teacher_train, pdf_teacher_test = get_Lizas_data()

# inp_train, inp_test, ev_train, ev_test, ev_teacher_train, ev_teacher_test, blend_factor = get_Lizas_data()

tf.keras.backend.clear_session()  # make sure we are working clean

max_value, min_value = __get_max_min_targets(train_targets=ev_train, test_targets=ev_test)

E_D_layers = 3
E_D_units = 80
out_shape = pdf_train.shape
in_shape = inp_train.shape


# ToDo: enable giving the builder a specific encoder and a specific decoder, that would be nicer maybe? dunno ...

model = build_model(E_D_layers=E_D_layers, E_D_units=E_D_units,
                in_shape=in_shape, out_shape=out_shape,
                dropout_rate=0.2,
                use_dropout=False,
                CNN_encoder=False, LSTM_encoder=True,
                LSTM_decoder=True,
                use_attention=True) #get a E_D_model

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # set optimizers, metrics and loss

from Losses_and_Metrics import loss_wrapper

# loss = loss_pdf_rMSE(max_value=max_value, min_value=min_value)
metrics = [loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='nME'),
           loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='nMSE'),
           loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='KL-D')]
model.compile(optimizer=optimizer,
              loss=loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='tile-to-forecast'),
              metrics=metrics) #compile, print summary
model.summary()


import numpy as np
hist_loss = []
hist_val_loss = []
hist_metric = []
hist_val_metric = []

relative_decrease = 0
epoch = 0
best_val_loss = np.inf
prev_val_loss = np.inf
best_val_metric = np.inf

# ToDo: insert loop that adjusts the blend_factor from 1 to 0 depending on how the validation loss develops
# ToDo: make this nicer, maybe with a isTrainig flag or sth?
val_metrics = {}
train_metrics = {}
decrease = 0
while decrease < 10:
    train_history = model.fit(x=[inp_train, pdf_teacher_train, blend_factor],
                              # train for a given set of epochs, look at history
                              y=pdf_train,
                              batch_size=12,
                              epochs=1,
                              shuffle=True,
                              validation_split=0.2)
    print(train_history.history.keys())
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
        blend_factor = blend_factor - 0.1
        blend_factor = [max(factor, 0.0) for factor in blend_factor]
        print('lowering blend factor')
        relative_decrease = 0

    epoch += 1
    prev_val_loss = val_loss

model.set_weights(best_wts)
test_results = model.evaluate(x=[inp_test, pdf_teacher_test, blend_factor],
                              y=[pdf_test],
                              batch_size=64,
                              verbose=1)
print(test_results)




