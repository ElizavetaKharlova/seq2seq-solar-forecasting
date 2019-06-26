# Daniel loads his data / If we want different data we do that here
# For a encoder decoder model we will need 3 inputs:
# encoder input, decoder support, blend factor (for how much we want to do teacher forcing)
#from load_DER_data import load_dataset
#raw_data = load_dataset()
#
#from datasets_utils import datasets_from_data
#inp, ev, pdf = datasets_from_data(raw_data,
#                                  sw_len_samples=int(7*24*(60/5)),
#                                  fc_len_samples=int(1*25*(60/5)),
#                                  fc_steps=25,
#                                  fc_tiles=33,
#                                  target_dims=[0,1,2,3,4],
#                                  plot=True,
#                                  steps_to_new_sample=15)

# Load Liza's data
# -------------------------------------------------------------------------
def get_data(data_path):
    from datasets_utils import datasets_from_data
    import gzip, pickle
    #    data_path = './data/Lizas_data.pklz'
    f = gzip.open(data_path,'rb')
    NetworkData = pickle.load(f)
    f.close()
    
    mvts_array = NetworkData['dataset']
    specs = NetworkData['specs']
    print('Specs:', specs)
    
    inputs, ev_targets, pdf_targets = datasets_from_data(data=mvts_array,
                                                         sw_len_samples=specs['sw_steps'],
                                                         fc_len_samples=specs['num_steps'],
                                                         fc_steps=specs['forecast_steps'],
                                                         fc_tiles=specs['num_tiles'],
                                                         target_dims=specs['target'],
                                                         plot=False,
                                                         steps_to_new_sample=1)
    print('Input shape:', inputs.shape, 'Target shape:', ev_targets.shape, pdf_targets.shape)
    return inputs, ev_targets, pdf_targets

def split_train_test(whole_set, split_size):
    split_ind = int(split_size*whole_set.shape[0])
    return whole_set[split_ind:], whole_set[:split_ind]

data_path = './data/Lizas_data.pklz'
inp, ev, pdf = get_data(data_path)

# ---------------------------------------------------------------------------

import numpy as np
ev = np.expand_dims(ev, axis=-1)
inp_train = inp[:int(0.8*inp.shape[0]),:,:]
inp_test = inp[int(0.8*inp.shape[0]):,:,:]
pdf_train = pdf[:int(0.8*inp.shape[0]),1:,:]
pdf_test = pdf[int(0.8*inp.shape[0]):,1:,:]
ev_train = ev[:int(0.8*inp.shape[0]),1:,:]
ev_test = ev[int(0.8*inp.shape[0]):,1:,:]
ev_teacher_train = ev[:int(0.8*inp.shape[0]),:-1,:]
ev_teacher_test = ev[int(0.8*inp.shape[0]):,:-1,:]

blend_factor = np.expand_dims(np.ones(inp_train.shape[0]), axis=-1)
print(blend_factor.shape)



print('The training set has an input data shape of ',
      inp_train.shape,
      'to expected value targets of ',
      ev_train.shape,
      'or alternatively pdf_targets of ',
      pdf_train.shape)
print('-----------------------------------------------')
print('The testing set has an input data shape of ',
      inp_test.shape,
      'to expected value targets of ',
      pdf_test.shape,
      'or alternatively pdf_targets of ',
      ev_test.shape)

import tensorflow as tf
from tensorflow import keras
keras.backend.clear_session() # make sure we are working clean

# Application of feed-forward & recurrent dropout on encoder & decoder hidden states.
class DropoutWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, zoneout_prob):
        super(ZoneoutWrapper, self).__init__()
        self.zoneout_prob = zoneout_prob
        self._layer = layer
    
    def call(self, inputs, initial_state):
        inputs = tf.nn.dropout(inputs, self.zoneout_prob)
        output, state_h, state_c = self._layer(inputs, initial_state)
        state_h = tf.nn.dropout(state_h, self.zoneout_prob)
        state_c = tf.nn.dropout(state_c, self.zoneout_prob)
        inputs = tf.nn.dropout(inputs, self.zoneout_prob)
        
        return output, state_h, state_c

# Application of Zopneout on hidden encoder & decoder states.
class ZoneoutWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, zoneout_prob):
        super(ZoneoutWrapper, self).__init__()
        self.zoneout_prob = zoneout_prob
        self._layer = layer
    
    def call(self, inputs, initial_state):
        inputs = tf.nn.dropout(inputs, self.zoneout_prob) # add feed forward dropout (i think the performance is better)
        output, state_h, state_c = self._layer(inputs, initial_state)
        state_h = (1-self.zoneout_prob)*tf.nn.dropout((state_h - initial_state[0]), self.zoneout_prob) + state_h
        state_c = (1-self.zoneout_prob)*tf.nn.dropout((state_c - initial_state[1]), self.zoneout_prob) + state_c
        
        return output, state_h, state_c


# this builds the wrapper class for the multilayer LSTM we will be using for most of the stuff
# in addition to the multiple LSTM layers it adds
# dropout
# layer norm (or maybe recurrent layer norm, whatever feels best)
# inits are the num_layers, num_units per layer and dropout rate
# call needs the inputs and the initial state
# it gives the outputs of all RNN layers (or should it be just one?)
# and the states at the last step
class encoder_LSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, dropout_rate):
        super(encoder_LSTM, self).__init__()
        self.layers = num_layers
        self.units = num_units
        self.encoder = [] # define the encoder layers
        self.encoder.append(tf.keras.layers.LSTM(self.units,
                                                 return_sequences=True,
                                                 return_state=True,
                                                 recurrent_initializer='glorot_uniform'))
        for layer in range(self.layers-1):
            self.encoder.append(ZoneoutWrapper(tf.keras.layers.LSTM(self.units,
                                                     return_sequences=True,
                                                     return_state=True,
                                                     recurrent_initializer='glorot_uniform'), zoneout_prob=dropout_rate)
    
    def call(self, encoder_inputs):
        encoder_states = []
        encoder_outputs = []
        enc_output_layer = None
        for layer in range(self.layers):
            if layer == 0:
                enc_output_layer, state_h, state_c = self.encoder[layer](encoder_inputs)
                encoder_states.append([state_h, state_c])
                encoder_outputs.append(enc_output_layer)
            else:
                enc_output_layer, state_h, state_c = self.encoder[layer](enc_output_layer, initial_state=encoder_states[layer])
                encoder_states.append([state_h, state_c])
                encoder_outputs.append(enc_output_layer)
        
        return encoder_states, encoder_outputs

class decoder_LSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, dropout_rate):
        super(decoder_LSTM, self).__init__()
        self.layers = num_layers
        self.units = num_units
        self.decoder = [] # define the encoder layers
        self.reshape_to_1ts = tf.keras.layers.Reshape((1,out_shape[1]))
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        for layer in range(self.layers):
            self.decoder.append(ZoneoutWrapper(tf.keras.layers.LSTM(self.units,
                                                     return_sequences=True,
                                                     return_state=True,
                                                     recurrent_initializer='glorot_uniform'), zoneout_prob=dropout_rate))
        
        self.decoder_wrap = [] #currently it seems that this layer is still creating errors of we have more than one... why?
        if self.units > 512:
            self.decoder_wrap.append(tf.keras.layers.Dense(units=256,
                                                           activation='relu'))
        self.decoder_wrap.append(tf.keras.layers.Dense(units=out_shape[1],
                                                       activation='relu'))

    def call(self, decoder_inputs, encoder_states, blend_factor_input):
        dec_states = encoder_states #encoder states are now decoder states
        blend_factor = self.reshape_to_1ts(blend_factor_input) #because, otherwise keras will be acting wierd as the decoder_output is (1,1) and this would be (1)
                                                       
        for t in range(out_shape[0]):
            buffer = self.reshape_to_1ts(decoder_inputs[:,t,:]) # slice a timestep of the support
            if t == 0: # if no previous output we cannot blend
                dec_t_minus_1 = buffer
            else: # otherwise blend as desired
                dec_t_minus_1 = (1.0-blend_factor)*dec_t_minus_1 + blend_factor*buffer
                                                                           
            for layer in range(self.layers): # feed last timesteps outputs as inputs to the decoder and propagate
                if layer == 0:
                    output_layer, state_h, state_c = self.decoder[layer](dec_t_minus_1, initial_state=dec_states[layer])
                    dec_states[layer] = [state_h, state_c] # replace the previous decoder states
                else:
                    output_layer, state_h, state_c = self.decoder[layer](output_layer, initial_state=dec_states[layer])
                    dec_states[layer] = [state_h, state_c] # replace the previous decoder states
                                                                                               
            dec_t_minus_1 = output_layer #go though the decoder projection
            for layer in range(len(self.decoder_wrap)):
                dec_t_minus_1 = self.decoder_wrap[layer](dec_t_minus_1)
                                                                                                           
            dec_t_minus_1 = self.reshape_to_1ts(dec_t_minus_1)
                                                                                                           
            if t == 0: #if no previous we cannot append
                decoder_output = dec_t_minus_1
            else: #else we append
                decoder_output = self.concat_layer([decoder_output, dec_t_minus_1])
                                                                                                                   
        return decoder_output


# this builds a basic encoder decoder model with:
# encoder layers / units == decoder layers / units
# output shape is the desired shape of the output [timesteps, dimensions]
# input shape is the supplied encoder input shape [inp_tmesteps, input_dimensions]
# the model will have three inputs:
# the encoder input [inp_tmesteps, input_dimensions]
# the decoder input  [timesteps, dimensions](for teacher forcing and blending to non-teacher forcing)
# the blend factor between decoder model output and real target out
# the model will have one output
# output [timesteps, dimensions]
def build_model(E_D_layers, E_D_units, in_shape, out_shape):
    layers = E_D_layers # remember units and layers
    units = E_D_units
    #   Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(inp_train.shape[1], inp_train.shape[2]))
    encoder = encoder_LSTM(layers, units, dropout_rate=0.2)
    # ------------------------------------------------------------------------------------------
    #   Decoder
    decoder_inputs = tf.keras.layers.Input(shape=(out_shape[0], out_shape[1]))
    blend_factor_input = tf.keras.layers.Input(shape=(1))
    decoder = decoder_LSTM(layers, units, dropout_rate=0.2)
    # ------------------------------------------------------------------------------------------
    encoder_states, encoder_outputs = encoder(encoder_inputs)
    decoder_output = decoder(decoder_inputs, encoder_states=encoder_states, blend_factor_input=blend_factor_input)
    
    print(decoder_output)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([encoder_inputs, decoder_inputs, blend_factor_input], decoder_output)
    return model

# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf
from tensorflow import keras
keras.backend.clear_session() # make sure we are working clean


E_D_layers = 2
E_D_units = 600
out_shape = [ev_train.shape[1], ev_train.shape[2]]
in_shape = [inp_train.shape[1], inp_train.shape[2]]
model = build_model(E_D_layers, E_D_units, in_shape, out_shape) #get a E_D_model

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) # set optimizers, metrics and loss
loss = tf.keras.losses.MeanSquaredError()
metrics = [tf.keras.metrics.MeanAbsolutePercentageError()]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics) #compile, print summary
model.summary()

#ToDo: insert loop that adjusts the blend_factor from 1 to 0 depending on how the validation loss develops
for epoch in range(20):
    history = model.fit(x=[inp_train, ev_teacher_train, blend_factor], #train for a given set of epochs, look at history
                        y=ev_train,
                        batch_size=32,
                        epochs=1,
                        validation_split=0.2)
    val_metric = history.history['val_mean_absolute_percentage_error']
    if epoch == 0:
        val_metric_before = val_metric

    if val_metric > val_metric_before:
        blend_factor = 0.9*blend_factor
    
    val_metric_before = val_metric
