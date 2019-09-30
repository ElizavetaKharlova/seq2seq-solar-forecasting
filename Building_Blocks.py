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
        self.state_dropout_layer = tf.keras.layers.Dropout(rate=dropout_prob)

    def call(self, inputs, initial_state):
        istraining = tf.keras.backend.learning_phase()
        inputs = self.i_o_dropout_layer(inputs, training=istraining)
        # get the output and hidden states in one layer of the network
        output, state_h, state_c = self.layer(inputs, initial_state)
        # apply dropout to each state of an LSTM layer
        state_h = self.state_dropout_layer(state_h, training=istraining)
        state_c = self.state_dropout_layer(state_c, training=istraining)
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
        self.transform_gate  = tf.keras.layers.Dense(units, activation='sigmoid', bias_initializer=tf.initializers.constant(iniital_bias))
        # self.transform_gate = tf.keras.layers.TimeDistributed(self.transform_gate )

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
        if self.use_norm:
            self.LSTM_norm_list_o_lists = []
        if self.use_hw:
            self.T_x_list_o_lists = []
        if self.use_quasi_dense:
            self.dense_drop = tf.keras.layers.Dropout(rate=dropout_rate)

        for block in range(len(self.units)):
            LSTM_list = []
            LSTM_norm_list = []
            T_x_list = []
            for layer in range(len(self.units[block])):
                # get one LSTM layer per layer we speciefied, with the units we need
                # Do we need to initialize the layer in the loop?
                one_LSTM_layer = tf.keras.layers.LSTM(units=self.units[block][layer],
                                                      activation='tanh',
                                                      recurrent_activation='sigmoid',
                                                      recurrent_dropout=0,
                                                      dropout=0,
                                                      unroll=False,
                                                      use_bias=True,
                                                      return_sequences=True,
                                                      return_state=True,
                                                      kernel_initializer='glorot_uniform',
                                                      recurrent_initializer='orthogonal',
                                                      bias_initializer='zeros')

                # if it is the first layer of we do not want to use dropout, we won't
                if layer > 0 and self.use_dropout:
                    one_LSTM_layer = DropoutWrapper(one_LSTM_layer, dropout_prob=dropout_rate, units=self.units[block][layer])

                # do we wanna norm
                if self.use_norm and not (block == len(self.units) and layer == len(self.units[block])):
                    LSTM_norm_list.append(tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True))
                #     one_LSTM_layer = Norm_LSTM_wrapper(one_LSTM_layer, norm_type='Layer norm')

                # do we wanna highway
                if self.use_hw:
                    if layer == 0:
                        T_x = None
                    elif units[block][layer-1] == units[block][layer]:
                        T_x = tf.keras.layers.Dense(units[block][layer], activation='sigmoid',
                                                                    bias_initializer=tf.initializers.constant(-1.5))
                    else:
                        T_x = None
                    T_x_list.append(T_x)

                    # if self.units[block][layer] == self.units[block][layer-1]:
                    #     one_LSTM_layer = Highway_LSTM_wrapper(one_LSTM_layer, self.units[block][layer], -1.5)

                LSTM_list.append(one_LSTM_layer) # will be len(layers_in_block)
            self.LSTM_list_o_lists.append(LSTM_list)
            if self.use_norm:
                self.LSTM_norm_list_o_lists.append(LSTM_norm_list)
            if self.use_hw:
                self.T_x_list_o_lists.append(T_x_list)

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

                if self.use_hw and self.T_x_list_o_lists[block][layer] is not None:
                    t_x = self.T_x_list_o_lists[block][layer](inputs_next_layer)
                    layer_out = t_x*layer_out + (1.0-t_x)*inputs_next_layer

                if self.use_norm:
                    layer_out = self.LSTM_norm_list_o_lists[block][layer](layer_out)

                if layer == len(self.units[block])-1 and self.use_quasi_dense:
                    inputs_next_layer = tf.concat([inputs_next_block, layer_out], axis=-1)
                    inputs_next_layer = self.dense_drop(inputs_next_layer, training=tf.keras.backend.learning_phase())
                else:
                    inputs_next_layer = layer_out

                all_out.append(inputs_next_layer)
                states_list.append([state_h, state_c])

            states_list_o_lists.append(states_list)
        return all_out, states_list_o_lists

class Attention(tf.keras.Model):
    def __init__(self, units, mode='Transformer',
                 only_context=True, context_mode='timeseries-like'):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, kernel_initializer='glorot_uniform', use_bias=False)
        #self.W1 = tf.keras.layers.TimeDistributed(self.W1)

        self.W2 = tf.keras.layers.Dense(units, kernel_initializer='glorot_uniform', use_bias=False)
        #self.W2 = tf.keras.layers.TimeDistributed(self.W2)

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
            self.W3 = tf.keras.layers.Dense(units,kernel_initializer='glorot_uniform', use_bias=False)
            #self.W3 = tf.keras.layers.TimeDistributed(self.W3)

            # if use_norm:
            #     self.W3 = Norm_wrapper(self.W3)
            # if use_dropout:
            #     self.W3 = Dropout_wrapper(self.W3, dropout_rate=dropout_rate)

        self.only_context = only_context
    
    def call(self, query, value=None, key=None):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        if value is None:
            value = query
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
            if len(query.shape) < len(key.shape):
                query = tf.expand_dims(query,1)
            score = tf.matmul(query, key, transpose_b=True)
            # scale score
            score = score / tf.math.sqrt(tf.cast(tf.shape(value)[-1], tf.float32))
            score = tf.nn.softmax(score, axis=1)

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
                 only_last_layer_output=True,
                 self_recurrent=True):
        super(decoder_LSTM_block, self).__init__()
        self.units = units
        self.self_recurrent = self_recurrent
        self.only_last_layer_output = only_last_layer_output
        self.use_attention = use_attention
        self.attention_hidden = attention_hidden
        self.built =False
        self.ml_LSTM_params = {'units': units,
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_norm': use_norm,
                        'use_hw': use_hw,
                        'use_quasi_dense': use_quasi_dense
                        }

        if self.use_attention:
            if self.attention_hidden:
                # ToDo: implement luong attention call method:
                self.h_attention = []
                self.c_attention = []
                self.h_attn_list = []
                self.c_attn_list = []
                self.augment_h = []
                self.augment_c = []
                self.w_h = []
                self.w_c = []
                for block in range(len(self.units)):
                    for layer in range(len(self.units[block])):
                        self.h_attn_list.append(Attention(self.units[block][layer], mode='Transformer', context_mode='force singular step'))
                        self.c_attn_list.append(Attention(self.units[block][layer], mode='Transformer', context_mode='force singular step'))
                        self.w_h.append(tf.keras.layers.Dense(units[block][layer], kernel_initializer='glorot_uniform', use_bias=False))
                        self.w_c.append(tf.keras.layers.Dense(units[block][layer], kernel_initializer='glorot_uniform', use_bias=False))
                    self.augment_h.append(self.w_h)
                    self.augment_c.append(self.w_c)
                    self.h_attention.append(self.h_attn_list)
                    self.c_attention.append(self.c_attn_list)

            else:
                self.attention = Attention(self.units[-1][-1], mode='Transformer') # choose the attention style (Bahdanau, Transformer)
            if self.self_recurrent:
                self.self_attention = Attention(self.units[0][0], mode='Transformer')
        self.decoder = MultiLayer_LSTM(**self.ml_LSTM_params)
        self.built = True

    def call(self, decoder_inputs, decoder_init_state, attention_value=None, attention_query=None):

        # add the Attention context vector
        # decoder inputs is the whole forecast!!
        # decoder_inputs_step = tf.expand_dims(decoder_inputs[:,-1,:], axis=1)

        if self.use_attention and attention_value is not None:
            if not self.attention_hidden:
                if self.self_recurrent:
                    self_attn = self.self_attention(attention_query, value=attention_query)
                    # does this make sense??
                    attention_query = tf.concat([attention_query, self_attn], axis=-1)
                context_vector = self.attention(attention_query, value=attention_value)
                # add attention to the input
                decoder_inputs = tf.concat([context_vector, decoder_inputs], axis=-1)

        decoder_out, decoder_state = self.decoder(decoder_inputs, initial_states=decoder_init_state)

        # Luong attention to find an augmented state based on current hidden state and encoder outputs
        if self.attention_hidden:
            decoder_attn_augmented_state = decoder_state #TODO: don't do overwrite, fix this to append
            for block in range(len(self.units)):
                for layer in range(len(self.units[block])):
                    state_h = decoder_state[block][layer][0]
                    state_c = decoder_state[block][layer][1]
                    # calculate attention on hidden state and encoder outputs
                    context_h = self.h_attention[block][layer](state_h, value=attention_value)
                    context_c = self.c_attention[block][layer](state_c, value=attention_value)
                    # concat context vector and hidden state
                    context_h = tf.concat([context_h, state_h], axis=-1)
                    context_c = tf.concat([context_c, state_c], axis=-1)
                    # calculate new state
                    context_h = self.augment_h[block][layer](context_h)
                    context_c = self.augment_c[block][layer](context_c)
                    augmented_state_h = tf.nn.tanh(context_h)
                    augmented_state_c = tf.nn.tanh(context_c)
                    # save the attention augmented states
                    decoder_attn_augmented_state[block][layer] = [augmented_state_h, augmented_state_c]

            # get new output with augmented states
            decoder_out, decoder_state = self.decoder(decoder_inputs,
                                                        initial_states=decoder_attn_augmented_state)


        #TODO: why repeat?
        if self.self_recurrent:
            decoder_state = decoder_init_state

        if self.self_recurrent:
            decoder_state = decoder_init_state
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


class DenseTCN(tf.keras.layers.Layer):
    """
    Dense temporal convolutional network
    requires number of block and layers within a block to initialize
    can change growth rate, squeeze factor, kernel sizes, strides, dropout and norm parameters
    requires inputs to run
    can have an arbitrary number of streams determined by the size of the kernel array
    outputs the raw network output in size [Batches, Timesteps, Features]
    requires a MIMO or a recursive wrapper to output predictions
    """
    def __init__(self, num_blocks, num_layers_per_block,
                growth_rate=12, 
                squeeze_factor=0.5, 
                use_dropout=False, 
                dropout_rate=0.0, 
                use_norm=False, 
                kernel_sizes=[3, 5]):
        super(DenseTCN, self).__init__()
        
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.growth_rate = growth_rate
        # bottleneck features are the 1x1 conv size for each layer
        self.bottleneck_features = 4*growth_rate
        # squeeze factor is the percentage of features passed from one block to the next
        self.squeeze_factor = squeeze_factor
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.kernel_sizes = kernel_sizes
        self.use_norm = use_norm
        # build flag
        self.already_built = False
        
        if self.use_norm:
            self.norm = []
        # create a stack of squeeze layers to be used between blocks (is not used after the last block)
        self.squeeze = []

    # initialize one CNN layer with arbitrary parameters
    def cnn_layer(self, num_features, kernel_size, dilation_rate):
        return tf.keras.layers.Conv1D(num_features,
                                        kernel_size=kernel_size,
                                        dilation_rate=dilation_rate,
                                        padding='causal')
                                        
    # build a stack for one layer of the network
    def build_layer(self, num_layer, kernel_size):
        layer = []
        layer.append(self.cnn_layer(self.bottleneck_features, kernel_size=1, dilation_rate=1))
        layer.append(self.cnn_layer(self.growth_rate, kernel_size, dilation_rate=2**num_layer))
        if self.use_norm:
            self.norm.append(tf.keras.layers.BatchNormalization())
        return layer

    # stack the network layers within a stream 
    def build_stream(self, kernel_size):
        stream = []
        for num_layer in range(self.num_layers_per_block):
            stream.append(self.build_layer(num_layer, kernel_size))
        return stream

    # build a stack of blocks which includes several streams with different kernel sizes
    def build_block(self, in_shape):
        self.blocks = []
        for block in range(self.num_blocks):
            all_streams = []
            for kernel_size in self.kernel_sizes:
                all_streams.append(self.build_stream(kernel_size))
            self.blocks.append(all_streams)
            # calculate the number of features in the squeeze layer
            # shape of input to the block + growth rate for all layers in the block
            if block == 0:
                num_features = int(self.squeeze_factor*(in_shape+self.num_layers_per_block*self.growth_rate))
            else: 
                num_features = int(self.squeeze_factor*(num_features + self.num_layers_per_block*self.growth_rate))
            # create stack of squeeze layers
            self.squeeze.append(self.cnn_layer(num_features, kernel_size=1, dilation_rate=1))
        # update build flag
        self.already_built = True
        return None

    # run one basic dense network layer
    # return output concatednated with input
    def run_layer(self, block, num_layer, num_stream, inputs):
        out = inputs
        out = self.blocks[block][num_stream][num_layer][0](out)
        out = self.blocks[block][num_stream][num_layer][1](out)
        if self.use_norm:
            out = self.norm[num_layer]
        if self.use_dropout:
            out = tf.nn.dropout(out,rate=self.dropout_rate)
        #TODO: not sure which activation to use
        out = tf.keras.activations.relu(out)
        return tf.concat([out, inputs], 2)
    
    # run output through one stream
    def run_stream(self, block, num_stream, inputs):
        out = inputs
        for num_layer in range(self.num_layers_per_block):
            out = self.run_layer(block, num_layer, num_stream, out)
        return out

    # run the output through blocks
    # within the block run through several streams according to size of the kernel array
    def run_block(self, block, inputs):
        out = []
        for num_stream in range(len(self.blocks[block])):
            # concatenate the outputs of several streams
            if num_stream == 0:
                out = self.run_stream(block, num_stream, inputs)
            else:
                out = tf.concat([out, self.run_stream(block, num_stream, inputs)], 2)
        # return output concatednated with input
        return out


    def call(self, inputs):
        out = inputs
        # call the init function upon first call
        if not self.already_built:
            self.build_block(inputs.shape[1])
        # iterate through blocks
        for block in range(self.num_blocks):
            out = self.run_block(block, out)
            # apply squeeze after each block except last
            if block < (self.num_blocks-1):
                out = self.squeeze[block](out)
        return out

########################################################################################################################
class attentive_TCN(tf.keras.layers.Layer):
    '''
    Encoder consisting of alternating self-attention and Desne TCN blocks
    requires units in shape of [[attention block units], [dense tcn units] ... ]
    the first block is always self-attention in order to get the most information from the input
    outputs after each block (dense tcn and self-attention) are concatenated
    '''
    def __init__(self, units, use_dropout=False, dropout_rate=0.15, use_norm=False):
        super(attentive_TCN, self).__init__()

        self.units = units
        self.encoder = []
        for block in range(len(self.units)):
            # first block is always attention
            if block%2 == 0:
                # can accommodate several self-attention layers
                encoder_block = []
                for layer in range(len(self.units[block])):
                    encoder_block.append(Attention(self.units[block][layer], mode='Transformer'))
                self.encoder.append(encoder_block)
            else:
                self.encoder.append(DenseTCN(num_blocks=1, 
                                            num_layers_per_block=len(self.units[block]), 
                                            growth_rate=12, 
                                            squeeze_factor=0.5, 
                                            use_dropout = use_dropout,
                                            dropout_rate=dropout_rate,
                                            use_norm=use_norm,
                                            kernel_sizes=[3]))


    def call(self, encoder_inputs):
        for block in range(len(self.encoder)):
            # for the first and later attention layers, concatenate the outputs
            if block%2 == 0:
                for layer in range(len(self.encoder[block])):
                    out = self.encoder[block][layer](encoder_inputs)
                    out = tf.concat([out,encoder_inputs],-1)
                    encoder_inputs = out
            else:
                # dense tcn block concatenates the outputs automatically
                out = self.encoder[block](out)
                encoder_inputs = out
        return out


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
                 num_units_projection=[120]):
        super(multihead_attentive_layer, self).__init__()
        # units is a list of lists, containing the numbers of units in the attention head
        self.num_heads = num_heads
        if num_units_projection == None:
            self.projection = False
        else:
            self.projection = True
            self.project = tf.keras.layers.Dense(units=num_units_projection,
                                  activation=None,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  )

        self.multihead_attention = []
        for head in range(self.num_heads):
            attention = Attention(num_units_per_head[head],
                                  mode='Transformer',
                                  only_context=True)
            self.multihead_attention.append(attention)

    def call(self,query, value):

        multihead_out = [head(query, value) for head in self.multihead_attention]
        if self.projection:
            multihead_out = tf.concat(multihead_out, axis=-1)
            return self.project(multihead_out)
        else:
            return multihead_out

class Transformer_encoder(tf.keras.layers.Layer):
    def __init__(self,
                num_units_per_head_per_layer, num_proj_units_per_layer,
                 use_dropout=True,
                 dropout_rate=0.15,
                 norm='layer'):
        super(Transformer_encoder, self).__init__()
        self.use_dropout = use_dropout
        if norm == 'layer':
            self.use_norm = True
        else:
            self.use_norm = False

        self.num_layers = len(num_units_per_head_per_layer)
        self.self_attention = []
        self.input_projection = []
        self.transition_layer = []
        if self.use_dropout:
            self.self_attn_dropout = []
            self.transition_dropout = []
        if self.use_norm:
            self.self_attn_norm = []
            self.transition_norm = []

        for layer in range(self.num_layers):
            num_units = num_units_per_head_per_layer[layer]
            channel_depth = num_proj_units_per_layer[layer]

            in_projection = tf.keras.layers.Dense(units=channel_depth,
                              activation='relu',
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              )
            in_projection = Norm_wrapper(in_projection, norm='layer')
            self.input_projection.append(in_projection)


            self.self_attention.append(multihead_attentive_layer(num_heads=len(num_units_per_head_per_layer),
                                                              num_units_per_head=num_units,
                                                              num_units_projection=channel_depth))
            if self.use_dropout:
                self.self_attn_dropout.append(tf.keras.layers.Dropout(dropout_rate))
                self.transition_dropout.append(tf.keras.layers.Dropout(dropout_rate))

            if norm=='layer':
                self.self_attn_norm.append(tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True))
                self.transition_norm.append(tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True))

            transition_layer = tf.keras.layers.Dense(units=channel_depth,
                              activation='relu',
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              )
            self.transition_layer.append(transition_layer)

    def call(self, layer_input):
        is_training = tf.keras.backend.learning_phase()
        for layer in range(self.num_layers):
            projected_input = self.input_projection[layer](layer_input)
            self_attention = self.self_attention[layer](projected_input, value=projected_input)
            if self.use_dropout:
                self_attention = self.self_attn_dropout[layer](self_attention, training=is_training)
            # do residuals
            self_attention = self_attention + projected_input
            if self.use_norm:
                self_attention = self.self_attn_norm[layer](self_attention)

            transition = self.transition_layer[layer](self_attention)
            if self.use_dropout:
                transition = self.transition_dropout[layer](transition, training=is_training)
            # do residuals
            transition = transition + self_attention
            if self.use_norm:
                transition = self.transition_norm[layer](transition)

            layer_input = transition
        return layer_input


class Transformer_decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_units_per_head_per_layer, num_proj_units_per_layer,
                 use_dropout=True,
                 dropout_rate=0.15,
                 norm='layer'):
        super(Transformer_decoder, self).__init__()
        self.use_dropout = use_dropout
        if norm == 'layer':
            self.use_norm = True
        else:
            self.use_norm = False

        self.num_layers = len(num_units_per_head_per_layer)
        self.self_attention = []
        self.encoder_attnetion = []
        self.input_projection = []
        self.transition_layer = []
        if self.use_dropout:
            self.self_attn_dropout = []
            self.attn_dropout = []
            self.transition_dropout = []
        if self.use_norm:
            self.self_attn_norm = []
            self.attn_norm = []
            self.transition_norm = []

        for layer in range(self.num_layers):
            num_units = num_units_per_head_per_layer[layer]
            channel_depth = num_proj_units_per_layer[layer]

            in_projection = tf.keras.layers.Dense(units=channel_depth,
                                                  activation='relu',
                                                  kernel_initializer='glorot_uniform',
                                                  bias_initializer='zeros',
                                                  )
            in_projection = Norm_wrapper(in_projection, norm='layer')
            self.input_projection.append(in_projection)

            self.self_attention.append(multihead_attentive_layer(num_heads=len(num_units_per_head_per_layer),
                                                                 num_units_per_head=num_units,
                                                                 num_units_projection=channel_depth))
            self.encoder_attnetion.append(multihead_attentive_layer(num_heads=len(num_units_per_head_per_layer),
                                                                 num_units_per_head=num_units,
                                                                 num_units_projection=channel_depth))
            if self.use_dropout:
                self.self_attn_dropout.append(tf.keras.layers.Dropout(dropout_rate))
                self.attn_dropout.append(tf.keras.layers.Dropout(dropout_rate))
                self.transition_dropout.append(tf.keras.layers.Dropout(dropout_rate))

            if norm == 'layer':
                self.self_attn_norm.append(tf.keras.layers.LayerNormalization(axis=-1,
                                                                         center=True,
                                                                         scale=True))
                self.attn_norm.append(tf.keras.layers.LayerNormalization(axis=-1,
                                                                         center=True,
                                                                         scale=True))
                self.transition_norm.append(tf.keras.layers.LayerNormalization(axis=-1,
                                                                               center=True,
                                                                               scale=True))

            transition_layer = tf.keras.layers.Dense(units=channel_depth,
                                                     activation='relu',
                                                     kernel_initializer='glorot_uniform',
                                                     bias_initializer='zeros',
                                                     )
            self.transition_layer.append(transition_layer)

    def call(self, decoder_inputs, attention_value):

        is_training = tf.keras.backend.learning_phase()
        layer_input = decoder_inputs
        for layer in range(self.num_layers):
            projected_input = self.input_projection[layer](layer_input)
            self_attention = self.self_attention[layer](projected_input, value=projected_input)
            if self.use_dropout:
                self_attention = self.self_attn_dropout[layer](self_attention, training=is_training)
            # do residuals
            self_attention = self_attention + projected_input
            if self.use_norm:
                self_attention = self.self_attn_norm[layer](self_attention)

            encoder_attention = self.encoder_attnetion[layer](self_attention, value=attention_value)
            if self.use_dropout:
                encoder_attention = self.attn_dropout[layer](encoder_attention, training=is_training)
            # do residuals
            encoder_attention = encoder_attention + self_attention
            if self.use_norm:
                encoder_attention = self.attn_norm[layer](encoder_attention)

            transition = self.transition_layer[layer](encoder_attention)
            if self.use_dropout:
                transition = self.transition_dropout[layer](transition, training=is_training)
            # do residuals
            transition = transition + encoder_attention
            if self.use_norm:
                transition = self.transition_norm[layer](transition)

            layer_input = transition

        return layer_input

class SelfAttentiveSelfRecurrent_Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 units=[[20, 20], [20,20]],
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_hw=True,
                 use_norm=True,
                 use_quasi_dense=True,
                 return_state=True,
                 attention_hidden=False,
                 only_last_layer_output=True):
        super(SelfAttentiveSelfRecurrent_Decoder, self).__init()

        self.self_attention = multihead_attentive_layer(num_heads=len(units[0]),
                                                        num_units_per_head=units[0][0],
                                                        num_units_projection=False)
        self.non_self_attention = multihead_attentive_layer(num_heads=len(units[0]),
                                                        num_units_per_head=units[0][0],
                                                        num_units_projection=False)
        ml_LSTM_params = {'units': units,
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_norm': use_norm,
                        'use_hw': use_hw,
                        'use_quasi_dense': use_quasi_dense
                        }
        self.LSTM = MultiLayer_LSTM(**ml_LSTM_params)

    def __build(self, inp_shape):
        pass

    def call(self, decoder_input, decoder_init_state, attention_query=None, attention_value=None):
        self_attention_out = self.self_attention(attention_query, value=attention_query)
        attention_out = self.non_self_attention(self_attention_out, value=attention_value)
        decoder_input = tf.concat([decoder_input, attention_out], axis=1)

        out_values, out_state = self.LSTM()
        pass

class WeightNormalization(tf.keras.layers.Wrapper):
    """This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    This speeds up convergence by improving the
    conditioning of the optimization problem.
    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)
    WeightNormalization wrapper works for keras and tf layers.
    ```python
      net = WeightNormalization(
          tf.keras.layers.Conv2D(2, 2, activation='relu'),
          input_shape=(32, 32, 3),
          data_init=True)(x)
      net = WeightNormalization(
          tf.keras.layers.Conv2D(16, 5, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(120, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(n_classes),
          data_init=True)(net)
    ```
    Arguments:
      layer: a layer instance.
      data_init: If `True` use data dependent variable initialization
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    """

    def __init__(self, layer, data_init=True, **kwargs):
        super(WeightNormalization, self).__init__(layer, **kwargs)
        self.data_init = data_init
        self._track_trackable(layer, name='layer')

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

        if not hasattr(self.layer, 'kernel'):
            raise ValueError('`WeightNormalization` must wrap a layer that'
                             ' contains a `kernel` for weights')

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(self.layer.kernel.shape[-1])
        self.kernel_norm_axes = list(range(self.layer.kernel.shape.rank - 1))

        self.g = self.add_variable(
            name='g',
            shape=(self.layer_depth,),
            initializer='ones',
            dtype=self.layer.kernel.dtype,
            trainable=True)
        self.v = self.layer.kernel

        self._initialized = self.add_variable(
            name='initialized',
            shape=None,
            initializer='zeros',
            dtype=tf.dtypes.bool,
            trainable=False)

        if self.data_init:
            # Used for data initialization in self._data_dep_init.
            layer_config = tf.keras.layers.serialize(self.layer)
            layer_config['config']['trainable'] = False
            self._naked_clone_layer = tf.keras.layers.deserialize(layer_config)
            self._naked_clone_layer.build(input_shape)
            self._naked_clone_layer.set_weights(self.layer.get_weights())
            self._naked_clone_layer.activation = None

        self.built = True

    def call(self, inputs):
        """Call `Layer`"""

        def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            # Ensure we read `self.g` after _update_weights.
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = tf.cond(self._initialized, _do_nothing, _update_weights)

        with tf.name_scope('compute_weights'):
            # Replace kernel by normalized weight variable.
            self.layer.kernel = tf.nn.l2_normalize(
                self.v, axis=self.kernel_norm_axes) * g

            # Ensure we calculate result after updating kernel.
            update_kernel = tf.identity(self.layer.kernel)
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

    def _initialize_weights(self, inputs):
        """Initialize weight g.
        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        with tf.control_dependencies([
                tf.debugging.assert_equal(  # pylint: disable=bad-continuation
                    self._initialized,
                    False,
                    message='The layer has been initialized.')
        ]):
            if self.data_init:
                assign_tensors = self._data_dep_init(inputs)
            else:
                assign_tensors = self._init_norm()
            assign_tensors.append(self._initialized.assign(True))
            return assign_tensors

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope('init_norm'):
            v_flat = tf.reshape(self.v, [-1, self.layer_depth])
            v_norm = tf.linalg.norm(v_flat, axis=0)
            g_tensor = self.g.assign(tf.reshape(v_norm, (self.layer_depth,)))
            return [g_tensor]

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""
        with tf.name_scope('data_dep_init'):
            # Generate data dependent init values
            x_init = self._naked_clone_layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1. / tf.math.sqrt(v_init + 1e-10)

            # Assign data dependent init values
            g_tensor = self.g.assign(self.g * scale_init)
            if hasattr(self.layer, 'bias'):
                bias_tensor = self.layer.bias.assign(-m_init * scale_init)
                return [g_tensor, bias_tensor]
            else:
                return [g_tensor]

    def get_config(self):
        config = {'data_init': self.data_init}
        base_config = super(WeightNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

