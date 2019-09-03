# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf
import numpy as np
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



inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher = get_Daniels_data()

# inp, ev_targets, ev_teacher, pdf_targets, pdf_teacher = get_Lizas_data()

out_shape = pdf_targets.shape[1:]
in_shape = inp.shape[1:]

from sklearn.model_selection import train_test_split
test_size = 0.15
inp_train, inp_test = train_test_split(inp, test_size=test_size, random_state=42)
pdf_targets_train, pdf_targets_test = train_test_split(pdf_targets, test_size=test_size, random_state=42)
pdf_teacher_train, pdf_teacher_test = train_test_split(pdf_teacher, test_size=test_size, random_state=42)


inp_train, inp_val = train_test_split(inp_train, test_size=test_size/(1-test_size), random_state=42)
pdf_targets_train, pdf_targets_val = train_test_split(pdf_targets_train, test_size=test_size/(1-test_size), random_state=42)
pdf_teacher_train, pdf_teacher_val = train_test_split(pdf_teacher_train, test_size=test_size/(1-test_size), random_state=42)
blend_factor_val = [1] * inp_val.shape[0]
blend_factor_val = np.array(blend_factor_val)
blend_factor_train = [1] * inp_train.shape[0]
blend_factor_train = np.array(blend_factor_train)

tf.keras.backend.clear_session()  # make sure we are working clean
from Building_Blocks import decoder_LSTM_block, block_LSTM
from Models import encoder_decoder_model, mimo_model
# decoder_model=decoder_LSTM_block(num_layers=3,
#                                  num_units=50,
#                                  use_dropout=False,
#                                  use_attention=True,
#                                  attention_hidden=False)
# # ToDo: would it be smarter to have this in the encoder decoder thingie instead of outside?
# projection_model=tf.keras.layers.Dense(units=out_shape[-1], activation=tf.keras.layers.Softmax(axis=-1))
#
# model = encoder_decoder_model(encoder_block=None,
#                               decoder_block=decoder_model,
#                               projection_block=projection_model,
#                               use_teacher=True,
#                               input_shape=in_shape,
#                               output_shape=out_shape,
#                               )

lstm_block = block_LSTM(num_layers=3,
                         num_units=70,
                         use_dropout=True,
                         dropout_rate=0.2,
                         use_norm=True,
                        only_last_layer_output=True)
model = mimo_model(function_block=lstm_block,
                   input_shape=in_shape,
                   output_shape=out_shape,
                   mode='project')
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # set optimizers, metrics and loss

from Losses_and_Metrics import loss_wrapper
metrics = [loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='nME'),
           loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='nMSE'),
           loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='KL-D')]
model.compile(optimizer=optimizer,
              loss=loss_wrapper(last_output_dimension_size=out_shape[-1], loss_type='tile-to-forecast'),
              metrics=metrics) #compile, print summary
model.summary()



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
# inp_train = inp_train.tolist()
# pdf_teacher_train = pdf_teacher_train.tolist()
# pdf_targets_train = pdf_targets_train.tolist()

# inp_val = inp_val.tolist()
# pdf_teacher_val = pdf_teacher_val.tolist()
# pdf_targets_val = pdf_targets_val.tolist()
decrease = 0
while decrease < 10:
    train_history = model.fit(x=[inp_train, pdf_teacher_train, blend_factor_train],
                              # train for a given set of epochs, look at history
                              y=pdf_targets_train,
                              batch_size=64,
                              epochs=1,
                              shuffle=True,
                              validation_data=([inp_val, pdf_teacher_val, blend_factor_val], pdf_targets_val))

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
test_results = model.evaluate(x=[inp_test],
                              y=[pdf_targets_test],
                              batch_size=64,
                              verbose=1)
print(test_results)




