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
class DropoutWrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, dropout_prob, units):
        super(DropoutWrapper, self).__init__(layer)
        self.i_o_dropout_layer = tf.keras.layers.Dropout(rate=dropout_prob, noise_shape=(1,units))
        # self.state_dropout_layer = tf.keras.layers.Dropout(rate=dropout_prob)

    def call(self, inputs, initial_state):
        istraining = tf.keras.backend.learning_phase()
        inputs = self.i_o_dropout_layer(inputs, training=istraining)
        # get the output and hidden states in one layer of the network
        output, state_h, state_c = self.layer(inputs, initial_state)
        # apply dropout to each state of an LSTM layer
        # state_h = self.state_dropout_layer(state_h, training=istraining)
        # state_c = self.state_dropout_layer(state_c, training=istraining)
        output = self.i_o_dropout_layer(output, training=istraining)
        return output, state_h, state_c

# Application of Zoneout on hidden encoder & decoder states.
class ZoneoutWrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, zoneout_prob):
        super(ZoneoutWrapper, self).__init__(layer)
        self.zoneout_prob = zoneout_prob

    def call(self, inputs, initial_state):
        # add feed forward dropout
        # get the flag for training
        # see which inputs to use
        def drop_inputs():
            return tf.nn.dropout(inputs, self.dropout_prob, )
        inputs = tf.keras.backend.in_train_phase(x=drop_inputs(), alt=inputs)

        # get outputs & hidden states in one layer of the network
        output, state_h, state_c = self._layer(inputs, initial_state)
        # apply zoneout to each state of an LSTM
        def zone_state_h():
            return (1 - self.zoneout_prob) * tf.nn.dropout((state_h - initial_state[0]), self.zoneout_prob) + state_h
        def no_zone_state_h():
            return state_h
        state_h = tf.keras.backend.in_train_phase(x=zone_state_h(), alt=no_zone_state_h())

        def zone_state_c():
            return (1 - self.zoneout_prob) * tf.nn.dropout((state_c - initial_state[1]), self.zoneout_prob) + state_c
        def no_zone_state_c():
            return state_c
        state_c = tf.keras.backend.in_train_phase(x=zone_state_c(), alt=no_zone_state_c())
        return output, state_h, state_c

class Highway_LSTM_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, units, iniital_bias):
        super(Highway_LSTM_wrapper, self).__init__(layer)
        transform_gate = tf.keras.layers.Dense(units, activation='sigmoid', bias_initializer=tf.initializers.constant(iniital_bias))
        self.transform_gate = tf.keras.layers.TimeDistributed(transform_gate)

    def call(self, input, initial_state):
        H_x, state_h, state_c = self.layer(input, initial_state)
        T_x = self.transform_gate(input)
        hw_out = tf.add(tf.multiply(H_x, T_x), tf.multiply((1.0-T_x), input))
        return hw_out, state_h, state_c

class Norm_LSTM_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, norm_type='Layer norm'):
        super(Norm_LSTM_wrapper, self).__init__(layer)
        if norm_type == 'Layer norm':
            self.norm = tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True)
        else:
            print('wrong norm type supplied to NormWrapper, supplied', norm_type)

    def call(self, input, initial_state):
        layer_out, state_h, state_c = self.layer(input, initial_state)
        layer_out = self.norm(layer_out)
        return layer_out, state_h, state_c

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

        self.units = units
        self.use_dropout = use_dropout
        self.use_norm = use_norm
        self.use_hw = use_hw
        self.use_quasi_dense = use_quasi_dense

        self.LSTM_list_o_lists = []  # Multiple layers work easiest as a list of layers, so here we start

        for block in range(len(self.units)):
            LSTM_list = []

            for layer in range(len(self.units[block])):
                # get one LSTM layer per layer we speciefied, with the units we need
                # Do we need to initialize the layer in the loop?
                if self.use_dropout:
                    one_LSTM_layer = tf.keras.layers.LSTM(self.units[block][layer],
                                                          activation=tf.nn.tanh,
                                                          return_sequences=True,
                                                          return_state=True,
                                                          implementation=2,
                                                          use_bias=True,
                                                          recurrent_dropout=dropout_rate,
                                                          recurrent_initializer='glorot_uniform')
                else:
                    one_LSTM_layer = tf.keras.layers.LSTM(self.units[block][layer],
                                                          activation=tf.nn.tanh,
                                                          return_sequences=True,
                                                          return_state=True,
                                                          use_bias=True,
                                                          implementation=2,
                                                          recurrent_initializer='glorot_uniform')

                # if it is the first layer of we do not want to use dropout, we won't
                if layer > 0 and self.use_dropout:
                    one_LSTM_layer = DropoutWrapper(one_LSTM_layer, dropout_prob=dropout_rate, units=self.units[block][layer])

                # do we wanna norm
                if self.use_norm:
                    one_LSTM_layer = Norm_LSTM_wrapper(one_LSTM_layer, norm_type='Layer norm')

                # do we wanna highway
                if self.use_hw and layer > 0:
                    if self.units[block][layer] == self.units[block][layer-1]:
                        one_LSTM_layer = Highway_LSTM_wrapper(one_LSTM_layer, self.units[block][layer], -1.5)

                LSTM_list.append(one_LSTM_layer) # will be len(layers_in_block)
            self.LSTM_list_o_lists.append(LSTM_list)

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
        states_list_o_lists = []
        # just a simple trick to prevent usage of one if_else loop for the first layer
        inputs_next_layer = inputs

        for block in range(len(self.units)):
            inputs_next_block = inputs_next_layer
            states_list = []

            for layer in range(len(self.units[block])):

                layer_out, state_h, state_c = self.LSTM_list_o_lists[block][layer](inputs_next_layer, initial_state=initial_states[block][layer])

                if layer == len(self.units[block])-1 and self.use_quasi_dense:
                    inputs_next_layer = tf.concat([inputs_next_block, layer_out], axis=-1)
                else:
                    inputs_next_layer = layer_out

                all_out.append(inputs_next_layer)
                states_list.append([state_h, state_c])

            states_list_o_lists.append(states_list)
        return all_out, states_list_o_lists

class Attention(tf.keras.Model):
    def __init__(self, units, mode='Transformer',
                 only_context=True, context_mode='timeseries-like',
                 use_norm=True, use_dropout=True, dropout_rate=0.15):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, activation=None, kernel_initializer='glorot_uniform', use_bias=False)
        self.W1 = tf.keras.layers.TimeDistributed(self.W1)

        self.W2 = tf.keras.layers.Dense(units, activation=None, kernel_initializer='glorot_uniform', use_bias=False)
        self.W2 = tf.keras.layers.TimeDistributed(self.W2)

        # if use_norm:
        #     self.W1 = Norm_wrapper(self.W1)
        #     self.W2 = Norm_wrapper(self.W2)
        # if use_dropout:
        #     self.W1 = Dropout_wrapper(self.W1, dropout_rate=dropout_rate)
        #     self.W2 = Dropout_wrapper(self.W2, dropout_rate=dropout_rate)

        self.context_mode = context_mode
        self.mode = mode
        if self.mode == 'Bahdanau':
            self.V = tf.keras.layers.Dense(1)
        elif self.mode == 'Transformer':
            self.W3 = tf.keras.layers.Dense(units, activation=None,kernel_initializer='glorot_uniform', use_bias=False)
            self.W3 = tf.keras.layers.TimeDistributed(self.W3)

            # if use_norm:
            #     self.W3 = Norm_wrapper(self.W3)
            # if use_dropout:
            #     self.W3 = Dropout_wrapper(self.W3, dropout_rate=dropout_rate)

        self.only_context = only_context
    
    def call(self, query, value, key=None):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        if key is None:
            key = value
        key = self.W2(key)
        query = self.W1(query)

        if self.mode == 'Bahdanau':
            # Bahdaau style attention: score = tanh(Q + V)
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(self.V(tf.nn.tanh(query + key)), axis=1)
            context_vector = attention_weights * value

        elif self.mode == 'Transformer':
            value = self.W3(value)

            # Trasformer style attention: score=Q*K/sqrt(dk)
            # in this case keys = values
            score = tf.matmul(query, key, transpose_b=True)  # (..., seq_len_q, seq_len_k)

            # scale score
            score = score / tf.math.sqrt(tf.cast(tf.shape(value)[-1], tf.float32))
            score = tf.nn.softmax(score)

            context_vector = tf.matmul(score, value)

        # context_vector shape after sum == (batch_size, hidden_size)
        if self.context_mode == 'force singular step':
            context_vector = tf.reduce_sum(context_vector, axis=1)

        if self.only_context:
            return context_vector
        else:
            return context_vector, attention_weights

class block_LSTM(tf.keras.layers.Layer):
    def __init__(self, units=[[20, 20], [20,20]],
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_norm=True,
                 use_hw=True,
                 use_quasi_dense=True,
                 return_state=True,
                 only_last_layer_output=True):
        super(block_LSTM, self).__init__()
        self.only_last_layer_output = only_last_layer_output
        self.return_state = return_state
        ml_LSTM_params = {'units': units,
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_norm': use_norm,
                        'use_hw': use_hw,
                        'use_quasi_dense': use_quasi_dense,
                        }
        self.multilayer_LSTMs = MultiLayer_LSTM(**ml_LSTM_params)

    def call(self, encoder_inputs, initial_states=None):
        block_output, block_states = self.multilayer_LSTMs(encoder_inputs, initial_states=initial_states)

        if self.only_last_layer_output:
            block_output =  block_output[-1]

        if self.return_state:
            return block_output, block_states
        else:
            return block_output

class decoder_LSTM_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=[[20, 20], [20,20]],
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_hw=True,
                 use_norm=True,
                 use_quasi_dense=True,
                 use_attention=False,
                 return_state=True,
                 attention_hidden=False,
                 only_last_layer_output=True):
        super(decoder_LSTM_block, self).__init__()
        self.units = units
        self.only_last_layer_output = only_last_layer_output
        #TODO: maybe fix attention
        self.use_attention = use_attention
        if self.use_attention:
            self.attention_hidden = attention_hidden
            self.attention = Attention(units[-1][-1], mode='Transformer') # choose the attention style (Bahdanau, Transformer)

        ml_LSTM_params = {'units': units,
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_norm': use_norm,
                        'use_hw': use_hw,
                        'use_quasi_dense': use_quasi_dense
                        }
        self.decoder = MultiLayer_LSTM(**ml_LSTM_params)


    def call(self, decoder_inputs, decoder_state, attention_value):
        # add the Attention context vector
        # decoder inputs is the whole forecast!!
        if self.use_attention:
            if self.attention_hidden:
                print('attn on hidden states not implemented yet!')
            else:
                context_vector = self.attention(decoder_inputs, value=attention_value)
                # add attention to the input
                decoder_inputs = tf.concat([tf.expand_dims(context_vector, axis=1), decoder_inputs], axis=-1)

        # use the decodeer_LSTM
        decoder_out, decoder_state = self.decoder(tf.expand_dims(decoder_inputs[:,-1,:], axis=1), initial_states=decoder_state)

        if self.only_last_layer_output:
            decoder_out = decoder_out[-1]
        return decoder_out, decoder_state

########################################################################################################################
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
                                                   dilation_rate=2 ** layer)  # increase dilation rate each layer (1,2,4,8...)

            self.CNN.append(one_CNN_layer)
            # if self.use_dropout:
            # self.dropout.append(tf.keras.layers.Dropout(self.dropout_rate))

    def call(self, inputs):
        all_out = []
        out = inputs
        isTraining = tf.keras.backend.learning_phase()
        for layer in range(self.num_layers):
            out = self.CNN[layer](out)
            if self.use_dropout and isTraining:
                # out = self.dropout[layer](out, training=isTraining)
                out = tf.nn.dropout(out, self.dropout_rate)
            all_out.append(out)
        return all_out

class encoder_CNN(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, use_dropout=True, dropout_rate=0.0):
        super(encoder_CNN, self).__init__()
        self.layers = num_layers
        self.units = num_units
        self.encoder = Multilayer_CNN(num_layers, num_units, use_dropout, dropout_rate)

    def call(self, encoder_inputs, initial_states=None):
        encoder_outputs = self.encoder(encoder_inputs)

        return encoder_outputs

class Norm_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, norm='layer'):
        super(Norm_wrapper, self).__init__(layer)

        if norm == 'layer':
            self.norm = tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True,)
        else:
            print('norm type', norm, 'not found, check the code to see if it is a typo')

    def call(self, inputs):
        return self.norm(self.layer(inputs))

class Dropout_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, dropout_rate):
        super(Dropout_wrapper, self).__init__(layer)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs):
        return self.dropout(self.layer(inputs), training=tf.keras.backend.learning_phase())

class multihead_attentive_layer(tf.keras.layers.Layer):
    def __init__(self, num_heads, num_units_per_head=[80],
                 num_units_projection=[120],
                 residual_mode=True,
                 reduce_ts_len_by=[2],
                 use_dropout=True, dropout_rate=0.0,
                 use_norm='layer'):
        super(multihead_attentive_layer, self).__init__()
        # units is a list of lists, containing the numbers of units in the attention head
        self.residual_mode = residual_mode

        self.num_heads = num_heads
        if reduce_ts_len_by > 1:
            self.pool = tf.keras.layers.AveragePooling1D(pool_size=reduce_ts_len_by,
                                                              strides=reduce_ts_len_by)
        else:
            self.pool = None
        self.project = tf.keras.layers.Dense(units=num_units_projection,
                              activation=None,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              )

        self.transform = tf.keras.layers.Dense(units=num_units_projection,
                              activation='relu',
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              )
        self.multihead_attention = []
        for head in range(self.num_heads):
            attention = Attention(num_units_per_head[head],
                                  mode='Transformer',
                                  only_context=True,
                                  use_norm=use_norm,
                                  use_dropout=use_dropout,
                                  dropout_rate=dropout_rate)

            #ToDo: add attention dropout wrapper
            self.multihead_attention.append(attention)

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.projection_dropout = tf.keras.layers.Dropout(dropout_rate)
            self.transform_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.use_norm = use_norm
        if self.use_norm:
            self.projection_norm = tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True,)
            self.transform_norm = tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True,)

    def call(self,query, value):
        multihead_out = []
        for head in range(self.num_heads):
            head_output = self.multihead_attention[head](query, value)
            multihead_out.append(head_output)
        multihead_out = tf.concat(multihead_out, axis=-1)
        projected_multihead_out = self.project(multihead_out)
        if self.use_dropout:
            projected_multihead_out = self.projection_dropout(projected_multihead_out, training=tf.keras.backend.learning_phase())
        if self.residual_mode and (projected_multihead_out.shape[1:] == query.shape[1:]):
            projected_multihead_out = projected_multihead_out + query
        if self.use_norm:
            projected_multihead_out = self.projection_norm(projected_multihead_out)


        transformed_output = self.transform(projected_multihead_out)
        if self.use_dropout:
            transformed_output = self.transform_dropout(transformed_output, training=tf.keras.backend.learning_phase())
        if self.residual_mode and (transformed_output.shape[1:] == projected_multihead_out.shape[1:]):
            transformed_output = transformed_output + projected_multihead_out
        if self.use_norm:
            transformed_output = self.transform_norm(transformed_output)

        # if self.pool is not None:
        #     transformed_output = self.pool(transformed_output)

        return transformed_output

class Transformer_encoder(tf.keras.layers.Layer):
    def __init__(self,
                num_units_per_head_per_layer, num_proj_units_per_layer, ts_reduce_by_per_layer,
                 residual_mode=True,
                 use_dropout=True,
                 dropout_rate=0.15,
                 norm='layer'):
        super(Transformer_encoder, self).__init__()
        self.num_layers = len(num_units_per_head_per_layer)
        self.self_attention = []
        for layer in range(self.num_layers):
            num_units = num_units_per_head_per_layer[layer]
            recue_ts_len = ts_reduce_by_per_layer[layer]
            self.self_attention.append(multihead_attentive_layer(num_heads=len(num_units_per_head_per_layer),
                                                              num_units_per_head=num_units,
                                                              reduce_ts_len_by=recue_ts_len,
                                                              num_units_projection=num_proj_units_per_layer[layer],
                                                              use_dropout=use_dropout,
                                                              residual_mode=residual_mode,
                                                              dropout_rate=dropout_rate,
                                                              use_norm=norm,
                                                              )
                                    )

    def call(self, next_input):
        for layer in range(self.num_layers):
            next_input = self.self_attention[layer](next_input, value=next_input)
        return next_input


class Transformer_decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers, num_units_per_head_per_layer, num_proj_units_per_layer, ts_reduce_by_per_layer,
                 use_dropout=True,
                 residual_mode=True,
                 dropout_rate=0.15,
                 norm='layer'):
        super(Transformer_decoder, self).__init__()
        self.num_layers = num_layers
        self.transformer = []
        for layer in range(num_layers):
            self.transformer.append(multihead_attentive_layer(num_heads=len(num_units_per_head_per_layer[layer]),
                                                              num_units_per_head=num_units_per_head_per_layer[layer],
                                                              reduce_ts_len_by=ts_reduce_by_per_layer[layer],
                                                              num_units_projection=num_proj_units_per_layer[layer],
                                                              use_dropout=use_dropout,
                                                              residual_mode=residual_mode,
                                                              dropout_rate=dropout_rate,
                                                              use_norm=norm,
                                                              )
                                    )

    def call(self, decoder_inputs, attention_value):
        next_input = decoder_inputs
        for layer in range(self.num_layers):

            if layer == 0:
                next_input = self.transformer[layer](decoder_inputs, value=decoder_inputs)
            else:
                next_input = self.transformer[layer](next_input, value=attention_value)

        return next_input