# contains all the building blocks necessary to build the architectures we will be working on
# In a sense, this will be the main modiefied file

# ToDO: clean up commentary
#ToDo: change in a way, that we can pass different encoders and decoders as arguments or something
# ToDO: change everything to tf.float32 for speeeeed


import tensorflow as tf
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

# Application of feed-forward & recurrent dropout on encoder & decoder hidden states.
class DropoutWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, dropout_prob):
        super(DropoutWrapper, self).__init__()
        self.dropout_prob = dropout_prob
        self._layer = layer
        # self.isTraining = isTraining

        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_prob)

    def call(self, inputs, initial_state):

        # get the flag for training
        istraining = tf.keras.backend.learning_phase()
        # see which inputs to use

        def drop_stuff(input_to_drop_from):
            # inp_shape = tf.shape(input_to_drop_from)
            # noise_shape = [inp_shape[0], 1, inp_shape[2]]
            return tf.nn.dropout(input_to_drop_from, self.dropout_prob)

        inputs = tf.keras.backend.in_train_phase(x=drop_stuff(inputs), alt=inputs)

        # get the output and hidden states in one layer of the network
        output, state_h, state_c = self._layer(inputs, initial_state)
        # apply dropout to each state of an LSTM layer
        # see which states to use

        state_h = tf.keras.backend.in_train_phase(x=drop_stuff(state_h), alt=state_h)
        state_c = tf.keras.backend.in_train_phase(x=drop_stuff(state_c), alt=state_c)
        output = tf.keras.backend.in_train_phase(x=drop_stuff(output), alt=output)
        return output, state_h, state_c


# Application of Zoneout on hidden encoder & decoder states.
class ZoneoutWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, zoneout_prob):
        super(ZoneoutWrapper, self).__init__()
        self.zoneout_prob = zoneout_prob
        self._layer = layer

    def call(self, inputs, initial_state):
        # add feed forward dropout
        # get the flag for training
        istraining = tf.keras.backend.learning_phase()
        # see which inputs to use
        def drop_inputs():
            return tf.nn.dropout(inputs, self.dropout_prob, )
        inputs = tf.keras.backend.in_train_phase(x=drop_inputs(), alt=inputs, training=istraining)

        # get outputs & hidden states in one layer of the network
        output, state_h, state_c = self._layer(inputs, initial_state)
        # apply zoneout to each state of an LSTM
        def zone_state_h():
            return (1 - self.zoneout_prob) * tf.nn.dropout((state_h - initial_state[0]), self.zoneout_prob) + state_h
        def no_zone_state_h():
            return state_h
        state_h = tf.keras.backend.in_train_phase(x=zone_state_h(), alt=no_zone_state_h(), training=istraining)

        def zone_state_c():
            return (1 - self.zoneout_prob) * tf.nn.dropout((state_c - initial_state[1]), self.zoneout_prob) + state_c
        def no_zone_state_c():
            return state_c
        state_c = tf.keras.backend.in_train_phase(x=zone_state_c(), alt=no_zone_state_c(), training=istraining)
        return output, state_h, state_c

class Highway_LSTM_wrapper(tf.keras.layers.Layer):
    def __init__(self, wrapped_layer, units, iniital_bias):
        super(Highway_LSTM_wrapper, self).__init__()

        self.wrapped_layer = wrapped_layer
        self.units = units
        self.initial_bias = iniital_bias

        self.Transform_gate = tf.keras.layers.Dense(units, activation='sigmoid', bias_initializer=tf.initializers.constant(iniital_bias))

    def call(self, input, initial_state):

        H_x, state_h, state_c = self.wrapped_layer(input, initial_state)
        T_x = self.Transform_gate(input)

        return tf.add(tf.multiply(H_x, T_x), tf.multiply((1.0-T_x), input)), state_h, state_c

class Norm_LSTM_wrapper(tf.keras.layers.Layer):
    def __init__(self, layer, norm_type='Layer norm'):
        super(Norm_LSTM_wrapper, self).__init__()

        self.layer = layer
        if norm_type == 'Layer norm':
            self.norm = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)
        else:
            print('wrong norm type supplied to NormWrapper, supplied', norm_type)

    def call(self, input, initial_state):
        layer_out, state_h, state_c = self.layer(input, initial_state)
        return self.norm(layer_out), state_h, state_c

# this builds the wrapper class for the multilayer LSTM
# in addition to the multiple LSTM layers it adds
# dropout
# layer norm (or maybe recurrent layer norm, whatever feels best)
# inits are the num_layers, num_units per layer and dropout rate
# call needs the inputs and the initial state
# it gives the outputs of all RNN layers (or should it be just one?)
# and the states at the last step
class MultiLayer_LSTM(tf.keras.layers.Layer):
    def __init__(self, units=[[20, 20], [20,20]], use_dropout=True, dropout_rate=0.0, use_norm=True, use_hw=True, use_quasi_dense=True):
        super(MultiLayer_LSTM, self).__init__()

        self.num_layers = len(units)  # simple, just save all the things we might need
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.use_norm = use_norm
        self.use_hw = use_hw
        self.use_quasi_dense = use_quasi_dense

        self.LSTM = []  # Multiple layers work easiest as a list of layers, so here we start
        self.norm = []
        self.Tx = []
        self.block_transition = []
        self.do_hw = []

        for block in range(len(self.units)):
            self.in_block_layers = []
            for layer in range(len(self.units[block])):
                # get one LSTM layer per layer we speciefied, with the units we need

                # Do we need to initialize the layer in the loop?
                one_LSTM_layer = tf.keras.layers.LSTM(self.units[block][layer],
                                                      activation=tf.nn.tanh,
                                                      return_sequences=True,
                                                      return_state=True,
                                                      recurrent_initializer='glorot_uniform')

                # if it is the first layer of we do not want to use dropout, we won't
                if layer > 0 and self.use_dropout:
                    one_LSTM_layer = DropoutWrapper(one_LSTM_layer, dropout_prob=self.dropout_rate)

                # do we wanna norm
                if self.use_norm:
                    one_LSTM_layer = Norm_LSTM_wrapper(one_LSTM_layer, norm_type='Layer norm')

                # do we wanna highway
                if self.use_hw and layer > 0:
                    if self.units[block][layer] == self.units[block][layer-1]:
                        one_LSTM_layer = Highway_LSTM_wrapper(one_LSTM_layer, self.units[block][layer], -1.5)

                self.in_block_layers.append(one_LSTM_layer) # will be len(layers_in_block)
            self.LSTM.append(self.in_block_layers)  # will be [blocks][layers_in_block]

    def call(self, inputs, initial_states=None):

        if initial_states is None:  # ToDo: think about noisy initialization?
            initial_states = []
            for block in range(len(self.units)):
                initial_states_block = []
                for layer in range(len(self.units[block])):
                    state_h = tf.zeros([tf.shape(inputs)[0], self.units[block][layer]])
                    state_c = tf.zeros([tf.shape(inputs)[0], self.units[block][layer]])
                    initial_states_block.append([state_h, state_c])
                initial_states.append(initial_states_block)
        all_out = []  # again, we are working with lists, so we might just as well do the same for the outputs
        states_out = []

        # just a simple trick to prevent usage of one if_else loop for the first layer
        inputs_next_layer = inputs
        for block in range(len(self.units)):
            inputs_next_block = inputs_next_layer
            states_block = []
            for layer in range(len(self.units[block])):
                layer_out, state_h, state_c = self.LSTM[block][layer](inputs_next_layer, initial_state=initial_states[block][layer])

                if layer == len(self.units[block])-1 and self.use_quasi_dense:
                    inputs_next_layer = tf.concat([inputs_next_block, layer_out], axis=-1)
                else:
                    inputs_next_layer = layer_out
                all_out.append(inputs_next_layer)
                states_block.append([state_h, state_c])
            states_out.append(states_block)
        return all_out, states_out

class Multilayer_CNN(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, use_dropout=True, dropout_rate=0.0, kernel_size=3, strides=1):
        super(Multilayer_CNN, self).__init__()
        
        self.num_layers = num_layers
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.kernel_size = kernel_size
        self.strides = strides
        
        self.CNN = []
        self.dropout = []
        
        for layer in range(self.num_layers):
            one_CNN_layer = tf.keras.layers.Conv1D(self.num_units,
                                                   kernel_size=self.kernel_size,
                                                   strides=self.strides,
                                                   padding='causal',
                                                   dilation_rate=2**layer) # increase dilation rate each layer (1,2,4,8...)
                
            self.CNN.append(one_CNN_layer)
            #if self.use_dropout:
                #self.dropout.append(tf.keras.layers.Dropout(self.dropout_rate))

    def call(self, inputs):
        all_out = []
        out = inputs
        isTraining = tf.keras.backend.learning_phase()
        for layer in range(self.num_layers):
            out = self.CNN[layer](out)
            if self.use_dropout and isTraining:
                #out = self.dropout[layer](out, training=isTraining)
                out = tf.nn.dropout(out, self.dropout_rate)
            all_out.append(out)
        return all_out


class Attention(tf.keras.Model):
    def __init__(self, units, bahdanau=True):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.W3 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.bahdanau = bahdanau
    
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
        
        if self.bahdanau:
            # Bahdaau style attention: score = tanh(Q + V)
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            score = self.V(tf.nn.tanh(self.W1(values_stacked) + self.W2(query_stacked)))
            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * values_stacked
        else:
            # Trasformer style attention: score=Q*K/sqrt(dk)
            # in this case keys = values
            matmul_qk = tf.matmul(self.W1(query_stacked), self.W2(values_stacked), transpose_b=True)  # (..., seq_len_q, seq_len_k)
            
            # scale matmul_qk
            dk = tf.cast(tf.shape(values)[-1], tf.float32)
            score = matmul_qk / tf.math.sqrt(dk)
            
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = tf.matmul(attention_weights, self.W3(values_stacked))
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights


# This builds the Encoder_LSTM
# basically just a simple pseudowrapper for the MultiLayer_LSTM layer
# ToDO: add a projection layer, that might be important for the attention mechs later on
# this will reduce the size of attention needed :-)

class block_LSTM(tf.keras.layers.Layer):
    def __init__(self, units=[[20, 20], [20,20]], use_dropout=False, dropout_rate=0.0, use_norm=True, use_hw=True, only_last_layer_output=True):
        super(block_LSTM, self).__init__()
        self.only_last_layer_output = only_last_layer_output
        ml_LSTM_params = {'units': units,
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_norm': use_norm,
                        'use_hw': use_hw,
                        }
        self.multilayer_LSTMs = MultiLayer_LSTM(**ml_LSTM_params)

    def call(self, encoder_inputs, initial_states=None):
        block_output, block_states = self.multilayer_LSTMs(encoder_inputs, initial_states=initial_states)

        if self.only_last_layer_output:
            block_output =  block_output[-1]

        return block_output, block_states

class encoder_CNN(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, use_dropout=True, dropout_rate=0.0):
        super(encoder_CNN, self).__init__()
        self.layers = num_layers
        self.units = num_units
        self.encoder = Multilayer_CNN(num_layers, num_units, use_dropout, dropout_rate)
    
    def call(self, encoder_inputs, initial_states=None):
        encoder_outputs= self.encoder(encoder_inputs)
        
        return encoder_outputs


# this builds the decoder LSTM
class decoder_LSTM_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=[[20, 20], [20,20]],
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_hw=True,
                 use_norm=True,
                 use_attention=False,
                 attention_hidden=False,
                 only_last_layer_output=True):
        super(decoder_LSTM_block, self).__init__()
        self.units = units
        self.use_attention = use_attention
        self.attention_hidden = attention_hidden
        self.only_last_layer_output = only_last_layer_output
        #TODO: maybe fix attention
        self.attention = Attention(units[-1][-1], bahdanau=False) # choose the attention style (Bahdanau, Transformer)
        ml_LSTM_params = {'units': units,
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_norm': use_norm,
                        'use_hw': use_hw,
                        }
        self.decoder = MultiLayer_LSTM(**ml_LSTM_params)


    def call(self, decoder_inputs, decoder_state, attention_value):
        # add the Attention context vector
        if self.use_attention:
            context_vector, attention_weights = self.attention(decoder_inputs, values=attention_value)
            # add attention to the input
            decoder_inputs = tf.concat([tf.expand_dims(context_vector, 1), decoder_inputs], axis=-1)
        # use the decodeer_LSTM
        decoder_out, decoder_state = self.decoder(decoder_inputs, initial_states=decoder_state)

        if self.only_last_layer_output:
            decoder_out = decoder_out[-1]
        return decoder_out, decoder_state

