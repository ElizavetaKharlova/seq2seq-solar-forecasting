def Daniel_Data_Loader():
    # Daniel loads his data / If we want different data we do that here
    # For a encoder decoder model we will need 3 inputs:
    # encoder input, decoder support, blend factor (for how much we want to do teacher forcing)
    from load_DER_data import load_dataset
    raw_data = load_dataset()
    
    from datasets_utils import datasets_from_data
    inp, ev, pdf = datasets_from_data(raw_data,
                                      sw_len_samples=int(7*24*(60/5)),
                                      fc_len_samples=int(1*25*(60/5)),
                                      fc_steps=25,
                                      fc_tiles=33,
                                      target_dims=[0,1,2,3,4],
                                      plot=True,
                                      steps_to_new_sample=15)
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
        
                                      
    return inp_train, inp_test, ev_train, ev_test, ev_teacher_train, ev_teacher_test, blend_factor

# Recommending Liza to put her dataloader here, so we can work in the same document maybe xD
def get_Lizas_data():
    from datasets_utils import datasets_from_data
    import gzip, pickle
    data_path = './data/Lizas_data.pklz'
    f = gzip.open(data_path,'rb')
    NetworkData = pickle.load(f)
    f.close()
    
    mvts_array = NetworkData['dataset']
    specs = NetworkData['specs']
    print('Specs:', specs)
    
    inp, ev, pdf = datasets_from_data(data=mvts_array,
                                      sw_len_samples=specs['sw_steps'],
                                      fc_len_samples=specs['num_steps'],
                                      fc_steps=specs['forecast_steps'],
                                      fc_tiles=specs['num_tiles'],
                                      target_dims=specs['target'],
                                      plot=False,
                                      steps_to_new_sample=1)

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
          
    return inp_train, inp_test, ev_train, ev_test, ev_teacher_train, ev_teacher_test, blend_factor


def get_sample_data():
    from pandas import read_csv
    from numpy import split
    from numpy import array
    
    # split a univariate dataset into train/test sets
    def split_dataset(data):
        # split into standard weeks
        train, test = data[1:-328], data[-328:-6]
        # restructure into windows of weekly data
        train = array(split(train, len(train)/7))
        test = array(split(test, len(test)/7))
        return train, test
    
    # load the new file
    dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    train, test = split_dataset(dataset.values)
    
    # convert history into inputs and outputs
    def to_supervised(train, n_input, n_out=7):
        # flatten data
        data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end < len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)
    
    n_input=14
    inp_train, ev_train_full = to_supervised(train, n_input)
    inp_test, ev_test_full = to_supervised(test, n_input)


    import numpy as np
    ev_train_full = np.expand_dims(ev_train_full, axis=-1)
    ev_test_full = np.expand_dims(ev_test_full, axis=-1)
    ev_train=ev_train_full[:,1:,:]
    ev_teacher_train = ev_train_full[:,:-1,:]
    ev_test=ev_test_full[:,1:,:]
    ev_teacher_test = ev_test_full[:,:-1,:]
    print('shape of ev_train', ev_train.shape, 'ev_teacher_train:', ev_teacher_test.shape)

    blend_factor = np.expand_dims(np.ones(inp_train.shape[0]), axis=-1)
    print(blend_factor.shape)



    print('The training set has an input data shape of ',
          inp_train.shape,
          'to expected value targets of ',
          ev_train.shape)
    print('-----------------------------------------------')
    print('The testing set has an input data shape of ',
          inp_test.shape,
          'or alternatively pdf_targets of ',
          ev_test.shape)

    return inp_train, inp_test, ev_train, ev_test, ev_teacher_train, ev_teacher_test, blend_factor

import tensorflow as tf
from tensorflow import keras
keras.backend.clear_session() # make sure we are working clean

# Application of feed-forward & recurrent dropout on encoder & decoder hidden states.
class DropoutWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, dropout_prob, isTraining):
        super(DropoutWrapper, self).__init__()
        self.dropout_prob = dropout_prob
        self._layer = layer
        self.isTraining = isTraining
    
    def call(self, inputs, initial_state):
        if self.isTraining:
            # add feed forward dropout
            inputs = tf.nn.dropout(inputs, self.dropout_prob)
            # get the output and hidden states in one layer of the network
            output, state_h, state_c = self._layer(inputs, initial_state)
            # apply dropout to each state of an LSTM layer
            state_h = tf.nn.dropout(state_h, self.dropout_prob)
            state_c = tf.nn.dropout(state_c, self.dropout_prob)
            # ToDo: GRU functionality?
        else: # test/validation time
            output, state_h, state_c = self._layer(inputs, initial_state)
        return output, state_h, state_c

# Application of Zoneout on hidden encoder & decoder states.
class ZoneoutWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, zoneout_prob, isTraining):
        super(ZoneoutWrapper, self).__init__()
        self.zoneout_prob = zoneout_prob
        self._layer = layer
        self.isTraining = isTraining

    def call(self, inputs, initial_state):
        if self.isTraining: # apply dropout at the training stage
            # add feed forward dropout
            inputs = tf.nn.dropout(inputs, self.zoneout_prob)
            # get outputs & hidden states in one layer of the network
            output, state_h, state_c = self._layer(inputs, initial_state)
            # apply zoneout to each state of an LSTM
            state_h = (1-self.zoneout_prob)*tf.nn.dropout((state_h - initial_state[0]), self.zoneout_prob) + state_h
            state_c = (1-self.zoneout_prob)*tf.nn.dropout((state_c - initial_state[1]), self.zoneout_prob) + state_c
            # ToDo: add functionality for GRU?
        else: # test/validation time
            output, state_h, state_c = self._layer(inputs, initial_state)
            state_h = (1-self.zoneout_prob)*state_h + self.zoneout_prob*initial_state[0]
            state_c = (1-self.zoneout_prob)*state_c + self.zoneout_prob*initial_state[1]
        return output, state_h, state_c

# this builds the wrapper class for the multilayer LSTM
# in addition to the multiple LSTM layers it adds
# dropout
# layer norm (or maybe recurrent layer norm, whatever feels best)
# inits are the num_layers, num_units per layer and dropout rate
# call needs the inputs and the initial state
# it gives the outputs of all RNN layers (or should it be just one?)
# and the states at the last step
class MultiLayer_LSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, use_dropout=True, dropout_rate=0.0, use_norm=True, isTraining=True):
        super(MultiLayer_LSTM, self).__init__()
        
        self.num_layers = num_layers # simple, just save all the things we might need
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.use_norm = use_norm
        self.isTraining = isTraining
        
        self.LSTM = [] # Multiple layers work easiest as a list of layers, so here we start
        self.norm = []
        for layer in range(self.num_layers):
            # get one LSTM layer per layer we speciefied, with the units we need
            
            # Do we need to initialize the layer in the loop?
            one_LSTM_layer = tf.keras.layers.LSTM(self.num_units,
                                                  return_sequences=True,
                                                  return_state=True,
                                                  recurrent_initializer='glorot_uniform')
                
            # if it is the first layer of we do not want to use dropout, we won't
            # otherwise add the zoneout wrapper
                                                  
            if layer == 0 or self.use_dropout==False:
                self.LSTM.append(one_LSTM_layer)
            else:
                self.LSTM.append(ZoneoutWrapper(one_LSTM_layer, zoneout_prob=self.dropout_rate, isTraining=self.isTraining))
                                                          
            if self.use_norm:
                self.norm.append(tf.keras.layers.LayerNormalization(axis=-1,
                                                                    center=True,
                                                                    scale=True))


    def call(self, inputs, initial_states):
        if initial_states is None:  # ToDo: think about noisy initialization?
            initial_states = []
            for layer in range(self.num_layers):
                state_h = tf.zeros([tf.shape(inputs)[0], self.num_units])
                state_c = tf.zeros([tf.shape(inputs)[0], self.num_units])
                initial_states.append([state_h, state_c])
        
        all_out = [] #again, we are working with lists, so we might just as well do the same for the outputs
        states = []
        
        out = inputs #just a simple trick to prevent usage of one if_else loop for the first layer
        for layer in range(self.num_layers):
            out, state_h, state_c = self.LSTM[layer](out, initial_state=initial_states[layer])
            if self.use_norm:
                out = self.norm[layer](out)
            all_out.append(out)
            states.append([state_h, state_c])

        return all_out, states

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        query_stacked = tf.concat(query, -1)
        values_stacked = tf.concat(values, -1)

        # make sure the first dimension is the batch dimension
        if query_stacked.shape[0] is not None:
            query_stacked = tf.transpose(query_stacked, [1,0,2])
            values_stacked = tf.transpose(values_stacked, [1, 0, 2])
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values_stacked) + self.W2(query_stacked)))
    
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
    
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values_stacked
        context_vector = tf.reduce_sum(context_vector, axis=1)
    
        return context_vector, attention_weights

# This builds the Encoder_LSTM
# basically just a simple pseudowrapper for the MultiLayer_LSTM layer
# ToDO: add a projection layer, that might be important for the attention mechs later on
# this will reduce the size of attention needed :-)
class encoder_LSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, use_dropout=True, dropout_rate=0.0):
        super(encoder_LSTM, self).__init__()
        self.layers = num_layers
        self.units = num_units
        self.encoder = MultiLayer_LSTM(num_layers, num_units, use_dropout, dropout_rate)
    
    def call(self, encoder_inputs, initial_states=None):
        encoder_outputs, encoder_states = self.encoder(encoder_inputs, initial_states=initial_states)
        
        return encoder_outputs, encoder_states

# this builds the decoder LSTM
# using the MultiLayer_LSTM, all we really need to do is do stepwise unrolling
# the decoder needs additional inputs compared to the decoder ...
class decoder_LSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, out_shape, use_dropout=True, dropout_rate=0.0, use_attention=True, attention_hidden=False):
        super(decoder_LSTM, self).__init__()
        self.layers = num_layers
        self.units = num_units
        self.out_shape = out_shape
        self.use_attention = use_attention
        self.attention_hidden = attention_hidden
        
        self.reshape_to_1ts = tf.keras.layers.Reshape((1,self.out_shape[1])) # Keras is finicky with dimensions, this is to make sure the dim is specified
        self.concat_timestep = tf.keras.layers.Concatenate(axis=1) # concatenates the timeteps together
        
        self.attention = BahdanauAttention(num_units)
        self.decoder = MultiLayer_LSTM(num_layers, num_units, use_dropout, dropout_rate)
        
        self.decoder_wrap = []
        if self.units > 512:
            self.decoder_wrap.append(tf.keras.layers.Dense(units=256,
                                                           activation='relu'))
        self.decoder_wrap.append(tf.keras.layers.Dense(units=self.out_shape[1]))
    
    def call(self, decoder_inputs, encoder_outputs, initial_states, blend_factor_input):
        
        blend_factor = self.reshape_to_1ts(blend_factor_input) #because, otherwise keras will be acting wierd as the decoder_output is (1,1) and this would be (1)
        
        dec_states_t = initial_states #get the initial states for the first timestep
        output_decoder_LSTM = encoder_outputs # for the first time step for attention 
        
        for t in range(self.out_shape[0]): #for loop over all timesteps
            
            # determine the input to the decoder
            target_support_t = self.reshape_to_1ts(decoder_inputs[:,t,:]) # slice a timestep of the support
            if t == 0: # if no previous output we cannot blend
                dec_in_t = target_support_t
            else: # otherwise blend as desired
                dec_in_t = (1.0-blend_factor)*dec_out_t + blend_factor*target_support_t
        
            # add the Attention context vector
            if self.use_attention:
                # apply attention either on the hidden layers of encoder/decoder (attention_hidden=True) or the output features of encoder/decoder
                if self.attention_hidden:
                    context_vector, attention_weights = self.attention(dec_states_t, initial_states)
                else:
                    context_vector, attention_weights = self.attention(output_decoder_LSTM, encoder_outputs)
                # add attention to the input
                dec_in_t = tf.concat([tf.expand_dims(context_vector, 1), dec_in_t], axis=-1)
            
            # use the decodeer_LSTM
            output_decoder_LSTM, dec_states_t = self.decoder(dec_in_t, initial_states=dec_states_t)
            
            # do the output projection
            dec_out_t_not_projected = output_decoder_LSTM[-1] #get the lat layer output of the decoder
            for layer in range(len(self.decoder_wrap)):
                dec_out_t_not_projected = self.decoder_wrap[layer](dec_out_t_not_projected)
            dec_out_t = self.reshape_to_1ts(dec_out_t_not_projected)
            
            # construct the output time-series
            if t == 0:
                decoder_output = dec_out_t
            else: #else we append
                decoder_output = self.concat_timestep([decoder_output, dec_out_t])
        
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
def build_model(E_D_layers, E_D_units, in_shape, out_shape, dropout_rate=0.3):
    layers = E_D_layers # remember units and layers
    units = E_D_units
    #   Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(inp_train.shape[1], inp_train.shape[2]))
    encoder = encoder_LSTM(layers, units, use_dropout=True, dropout_rate=dropout_rate)
    # ------------------------------------------------------------------------------------------
    #   Decoder
    decoder_inputs = tf.keras.layers.Input(shape=(out_shape[0], out_shape[1]))
    blend_factor_input = tf.keras.layers.Input(shape=(1))
    decoder = decoder_LSTM(layers, units, use_dropout=True, dropout_rate=dropout_rate, out_shape=out_shape)
    # ------------------------------------------------------------------------------------------
    encoder_outputs, encoder_states = encoder(encoder_inputs)
    decoder_output = decoder(decoder_inputs, encoder_outputs=encoder_outputs, initial_states=encoder_states, blend_factor_input=blend_factor_input)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([encoder_inputs, decoder_inputs, blend_factor_input], decoder_output)
    return model

# Cost functions and Metrics
# calculate the relative mean error
def calculate_rME(y_true, y_pred, min_value, max_value):
    
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)
    
    max_offset = tf.math.abs(min_value)
    min_offset = tf.math.abs(max_value)
    abs_offset = tf.add(max_offset, min_offset)
    
    loss = tf.math.add(y_true, -y_pred) # (batches, steps, 1)
    loss = tf.math.abs(loss) # to make sure the error is positive
    loss = tf.math.add(min_offset, loss) # (batches, steps, 1), make the minimum value 0
    loss = tf.math.divide(loss, abs_offset) # (batches, steps, 1), normalize the errors relative to the largest error
    loss = tf.squeeze(loss, axis=-1) # (batches, steps)
    loss = tf.math.reduce_mean(loss, axis=-1) # (bacthes), get the mean error per timestep
        
    loss = tf.math.multiply(tf.cast(100.0, dtype=tf.float64), loss) # (batches) convert into percents
        
    return loss

def rME_Metric(max_value, min_value):
    def rME(y_true, y_pred):
        return calculate_rME(y_true, y_pred, min_value, max_value)
    return rME

# same thing, just relative mean squared error
def calculate_rMSE(y_true, y_pred, min_value, max_value):
    
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)
    
    max_offset = tf.math.abs(min_value)
    min_offset = tf.math.abs(max_value)
    abs_offset = tf.add(max_offset, min_offset)
    
    y_true = tf.math.add(y_true, min_offset)
    y_pred = tf.math.add(y_pred, min_offset)
        
    y_true = tf.square(y_true)
    y_pred = tf.square(y_pred)
        
    loss = tf.math.add(y_true, -y_pred) # (batches, steps, 1)
    loss = tf.math.abs(loss)
    loss = tf.math.divide(loss, tf.square(abs_offset)) # (batches, steps, 1), normalize the errors relative to the largest error
    loss = tf.squeeze(loss, axis=-1) # (batches, steps)
    loss = tf.math.reduce_mean(loss, axis=-1) # (bacthes), get the mean error per timestep
        
    loss = tf.math.multiply(tf.cast(100.0, dtype=tf.float64), loss) # (batches) convert into percents
        
    return loss

def rMSE_Metric(max_value, min_value):
    def rMSE(y_true, y_pred):
        return calculate_rMSE(y_true, y_pred, min_value, max_value)
    return rMSE

# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf
from tensorflow import keras
keras.backend.clear_session() # make sure we are working clean

E_D_layers = 4
E_D_units = 600
inp_train, inp_test, ev_train, ev_test, ev_teacher_train, ev_teacher_test, blend_factor = get_Lizas_data()

import numpy as np
max_value_train = np.amax(ev_train, axis=0)
max_value_train = np.amax(max_value_train, axis=0)
max_value_test = np.amax(ev_test, axis=0)
max_value_test = np.amax(max_value_test, axis=0)
max_value = np.maximum(max_value_train, max_value_test)
print(max_value)

min_value_train = np.amin(ev_train, axis=0)
min_value_train = np.amin(min_value_train, axis=0)
min_value_test = np.amin(ev_test, axis=0)
min_value_test = np.amin(min_value_test, axis=0)
min_value = np.minimum(min_value_test, min_value_train)
print(max_value)

out_shape = [ev_train.shape[1], ev_train.shape[2]]
in_shape = [inp_train.shape[1], inp_train.shape[2]]
model = build_model(E_D_layers, E_D_units, in_shape, out_shape) #get a E_D_model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) # set optimizers, metrics and loss
loss = rMSE_Metric(max_value=max_value, min_value=min_value)
metrics = [rME_Metric(max_value=max_value, min_value=min_value)]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics) #compile, print summary
model.summary()

#ToDo: insert loop that adjusts the blend_factor from 1 to 0 depending on how the validation loss develops
hist_loss = []
hist_val_loss = []
hist_metric = []
hist_val_metric = []
decrease = 0
relative_decrease = 0
epoch  = 0
best_val_loss = 1e6
prev_val_loss = 1e6

sample_size = int(0.8*inp_train.shape[0])
inp_val = inp_train[sample_size:, :, :]
inp_train_set = inp_train[:sample_size, :, :]

ev_teacher_val = ev_teacher_train[sample_size:, :, :]
ev_teacher_train_set = ev_teacher_train[:sample_size , :, :]

ev_train_val = ev_train[sample_size:, :, :]
ev_train_train_set = ev_train[:sample_size , :, :]

blend_factor_val = blend_factor[sample_size:, :]
blend_factor_val = np.zeros(blend_factor_val.shape)
blend_factor_train_set = blend_factor[:sample_size , :]
print('inp test ', inp_train_set.shape, 'ev_teacher', ev_teacher_train_set.shape, 'ev_train', ev_train_train_set.shape, 'blend ', blend_factor_train_set.shape)
print('inp test ', inp_val.shape, 'ev_teacher', ev_teacher_val.shape, 'ev_train', ev_train_val.shape, 'blend ', blend_factor_val.shape)
while decrease < 6:
    train_history = model.fit(x=[inp_train_set, ev_teacher_train_set, blend_factor_train_set], #train for a given set of epochs, look at history
                              y=ev_train_train_set,
                              batch_size=128,
                              epochs=2,
                              validation_split=0.0)
        
    val_history = model.evaluate(x=[inp_val, ev_teacher_val, blend_factor_val], #train for a given set of epochs, look at history
                                                           y=ev_train_val,
                                                           batch_size=128)
    val_metric = val_history[0]
    val_loss = val_history[1]
                              
    if best_val_loss > val_loss: #if we see no increase in absolute performance, increase the death counter
        decrease = 0 # reset the death counter
        best_val_loss = val_loss
        best_wts = model.get_weights()
        print('saving a new model')
    else:
        decrease += 1


    if prev_val_loss < val_loss: #see if we are having a relative decrease
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
