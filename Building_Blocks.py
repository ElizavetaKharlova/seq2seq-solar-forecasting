# contains all the building blocks necessary to build the architectures we will be working on
# In a sense, this will be the main modiefied file


import tensorflow as tf
import numpy as np
import random
########################################################################################################################
'''General help functions, lowest level'''
initializer = tf.keras.initializers.glorot_uniform()
activation = tf.nn.relu


def randomize_input_lengths(signal, random_degree):
    rand = tf.random.uniform(shape=(), minval=1 - random_degree, maxval=1)
    new_signal_length = max(int(rand * signal.shape[1]), 1)
    new_signal_length = 30
    return signal[:, -new_signal_length:, :]

def drop_features_of_signal(input, dropout_rate):
    batch_dim = tf.shape(input)[0]
    # if batch_dim is None:
    #     batch_dim = 128
    return tf.keras.backend.in_train_phase(
                tf.nn.dropout(input,
                              dropout_rate,
                              noise_shape=(batch_dim, 1, input.shape[2]),
                              ),
                alt=input,
                training=tf.keras.backend.learning_phase())

def drop_timesteps_of_signal(input, dropout_rate):
    batch_dim = tf.shape(input)[0]
    # if batch_dim is None:
    #     batch_dim = 128
    return tf.keras.backend.in_train_phase(
                tf.nn.dropout(input, dropout_rate, noise_shape=(batch_dim, input.shape[1], 1)),
                alt=input,
                training=tf.keras.backend.learning_phase())

class Cropping_layer(tf.keras.layers.Layer):
    def __init__(self, noise_rate):
        super(Cropping_layer, self).__init__()
        self.noise_rate = noise_rate

    def call(self, signal):
        return tf.keras.backend.in_train_phase(self.randomize_input_lengths(signal),
                                                    alt=signal,
                                                    training=tf.keras.backend.learning_phase())

    def randomize_input_lengths(self, signal):
        rand = tf.random.uniform(shape=(), minval=0 , maxval=int(self.noise_rate*signal.shape[1]), dtype=tf.int32)
        signal = tf.slice(signal, begin=[0, rand, 0], size=[-1,-1,-1])
        # signal = signal[:, -new_signal_length:, :]
        return signal
########################################################################################################################
'''Wrappers for feedforward architectures'''
# important to note that all wrappers have to be compatible with both, attention(query, value) and self_attention(query)

class Dropout_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, dropout_rate, units=None):
        super(Dropout_wrapper, self).__init__(layer)
        # self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, noise_shape=(1, units))
        self.units = units
        self.dropout_rate = dropout_rate
    def call(self, inputs, value=None):
        if value is None:
            return drop_features_of_signal(self.layer(inputs), self.dropout_rate)
        else:
            return drop_features_of_signal(self.layer(inputs, value), self.dropout_rate)
        # if value is None:
        #     dropped_inputs = drop_features_of_signal(inputs, self.dropout_rate)
        #     return self.layer(dropped_inputs)
        # else:
        #     dropped_inputs = drop_features_of_signal(inputs, self.dropout_rate)
        #     dropped_values = drop_features_of_signal(value, self.dropout_rate)
        #     return self.layer(dropped_inputs, dropped_values)

class Norm_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, norm='layer'):
        super(Norm_wrapper, self).__init__(layer)

        if norm == 'layer':
            self.norm = tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True, )
        elif norm == 'batch':
            self.norm = tf.keras.layers.BatchNormalization(axis=-1,
                                                           center=True,
                                                           scale=True)
        else:
            print('norm type', norm, 'not found, check the code to see if it is a typo')

    def call(self, inputs, value=None):
        inputs = self.norm(inputs)
        if value is None:
            return self.layer(inputs)
        else:
            value = self.norm(inputs)
            return self.layer(inputs, value)

class Residual_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer):
        super(Residual_wrapper, self).__init__(layer)

    def call(self, inputs, value=None):
        if value is None:
            return inputs + self.layer(inputs)
        else:
            return inputs + self.layer(inputs, value=value)

########################################################################################################################
'''LSTM wrappers'''
#
class LSTM_DropoutWrapper(tf.keras.layers.Wrapper):
    '''
    Application of feed-forward & recurrent dropout on encoder & decoder hidden states.
    '''
    def __init__(self, layer, dropout_prob, units):
        super(LSTM_DropoutWrapper, self).__init__(layer)
        self.dropout_prob = dropout_prob

    def call(self, inputs, initial_state):
        inputs = drop_features_of_signal(inputs, self.dropout_prob)
        # get the output and hidden states in one layer of the network
        output, state_h, state_c = self.layer(inputs, initial_state)
        outputs = drop_features_of_signal(output)
        # apply dropout to each state of an LSTM layer
        # state_h = self.state_dropout_layer(state_h, training=istraining)
        # state_c = self.state_dropout_layer(state_c, training=istraining)
        return output, state_h, state_c

class ZoneoutWrapper(tf.keras.layers.Wrapper):
    '''
    Application of Zoneout on hidden encoder & decoder states.
    '''
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

class LSTM_Residual_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer):
        super(LSTM_Residual_wrapper, self).__init__(layer)
        # self.transform_gate = tf.keras.layers.TimeDistributed(self.transform_gate )

    def call(self, input, initial_states=None):
        output, states = self.layer(input, initial_states)
        output[-1] = output[-1]+input
        return output, states

class LSTM_Highway_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, units, iniital_bias):
        super(LSTM_Highway_wrapper, self).__init__(layer)
        self.transform_gate  = tf.keras.layers.Dense(units,
                                                     activation='sigmoid',
                                                     kernel_initializer=initializer,
                                                     bias_initializer=tf.initializers.constant(iniital_bias))
        # self.transform_gate = tf.keras.layers.TimeDistributed(self.transform_gate )

    def call(self, input, initial_state):
        H_x, state_h, state_c = self.layer(input, initial_state)
        T_x = self.transform_gate(input)
        hw_out = tf.add(tf.multiply(H_x, T_x), tf.multiply((1.0-T_x), input))
        return hw_out, state_h, state_c

class LSTM_Norm_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, norm_type='Layer norm'):
        super(LSTM_Norm_wrapper, self).__init__(layer)
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

########################################################################################################################

class MultiLayer_LSTM(tf.keras.layers.Layer):
    '''
    this builds the wrapper class for the multilayer LSTM
    in addition to the multiple LSTM layers it adds dropout
    layer norm (or maybe recurrent layer norm, whatever feels best)
    inits are the num_layers, num_units per layer and dropout rate
    call needs the inputs and the initial state
    it gives the outputs of all RNN layers (or should it be just one?)
    and the states at the last step
    '''
    def __init__(self, units=[[20, 20], [20,20]],
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_norm=False,
                 use_hw=False,
                 use_residuals = False,
                 return_all_states = True,
                 L1=0.0,
                 L2=0.0):
        super(MultiLayer_LSTM, self).__init__()

        self.units = units
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.use_hw = use_hw
        self.use_residuals = use_residuals
        self.return_all_states = return_all_states

        self.LSTM_list_o_lists = []  # Multiple layers work easiest as a list of layers, so here we start

        for block in range(len(self.units)):
            LSTM_list = []
            for layer in range(len(self.units[block])):
                # get one LSTM layer per layer we speciefied, with the units we need
                # Do we need to initialize the layer in the loop?
                one_LSTM_layer = tf.keras.layers.LSTM(units=self.units[block][layer],
                                                      activation='tanh',
                                                      recurrent_activation='sigmoid',
                                                      recurrent_dropout=dropout_rate,
                                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                      dropout=dropout_rate,
                                                      unroll=False,
                                                      use_bias=True,
                                                      return_sequences=True,
                                                      implementation=2,
                                                      return_state=True,
                                                      kernel_initializer=initializer,
                                                      recurrent_initializer='orthogonal',
                                                      bias_initializer='zeros')

                # do we wanna highway
                if self.use_hw:
                    one_LSTM_layer = LSTM_Highway_wrapper(one_LSTM_layer)
                elif self.use_residuals:
                    one_LSTM_layer = LSTM_Residual_wrapper(one_LSTM_layer)
                # do we wanna norm
                if self.use_norm:
                    one_LSTM_layer = LSTM_Norm_wrapper(one_LSTM_layer)

                LSTM_list.append(one_LSTM_layer) # will be len(layers_in_block)
            self.LSTM_list_o_lists.append(LSTM_list)


    def call(self, signal, initial_states=None):

        # Initialize as zero states if No state is given
        if initial_states is None:
            initial_states = []
            for block in range(len(self.units)):
                initial_states_block = []
                for layer in range(len(self.units[block])):
                    state_h = tf.zeros([tf.shape(signal)[0], self.units[block][layer]])
                    state_c = tf.zeros([tf.shape(signal)[0], self.units[block][layer]])
                    initial_states_block.append([state_h, state_c])
                initial_states.append(initial_states_block)

        all_states = []
        # just a simple trick to prevent usage of one if_else loop for the first layer
        for block in range(len(self.units)):
            block_states = []

            for layer in range(len(self.units[block])):

                signal, state_h, state_c = self.LSTM_list_o_lists[block][layer](signal, initial_state=initial_states[block][layer])

                block_states.append([state_h, state_c])

            all_states.append(block_states)
        return signal, all_states

########################################################################################################################

class Attention(tf.keras.layers.Layer):
    '''
    Implements an attention layer.
    Implemented are Bahdanau and Transformer style attention.
    If query/key kernel and strides are supplied this mimics the proposed convolutional attention mechanism.
    '''
    def __init__(self, units,
                 mode='Transformer',
                 causal_attention=False,
                 only_context=True,
                 query_kernel=1,
                 query_stride=1,
                 key_kernel=1,
                 key_stride=1,
                 use_dropout=True,
                 dropout_rate=0.2,
                 L1=0.0, L2=0.0,
                 ):
        super(Attention, self).__init__()
        self.mode = mode
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.only_context = only_context
        self.causal_attention = causal_attention
        self.W_query = tf.keras.layers.Conv1D(units,
                                         #activation=tf.nn.sigmoid,
                                         strides=query_stride,
                                         kernel_size=query_kernel,
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                         padding='causal',
                                         kernel_initializer=initializer,
                                         use_bias=False)
        if use_dropout:
            self.W_query = Dropout_wrapper(self.W_query, dropout_rate=dropout_rate)

        self.W_value = tf.keras.layers.Conv1D(units,
                                         #activation=tf.nn.sigmoid,
                                         strides=1,
                                         kernel_size=1,
                                         padding='causal',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                         kernel_initializer=initializer,
                                         use_bias=False)
        if use_dropout:
            self.W_value = Dropout_wrapper(self.W_value, dropout_rate=dropout_rate)

        if mode == 'Bahdanau':
            self.V = tf.keras.layers.Dense(1)
        elif mode == 'Transformer':
            self.W_key = tf.keras.layers.Conv1D(units,
                                         #activation=tf.nn.sigmoid,
                                         strides=key_stride,
                                         kernel_size=key_kernel,
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                         padding='causal',
                                         kernel_initializer=initializer,
                                         use_bias=False)
            if use_dropout:
                self.W_key = Dropout_wrapper(self.W_key, dropout_rate=dropout_rate)

    def dropout_mask(self, score):
        dropout_mask = tf.random.uniform(shape=tf.shape(score), minval=0.0, maxval=1.0)
        dropout_mask = tf.where(dropout_mask <= self.dropout_rate/2, x=-1e12, y=0.0)
        return dropout_mask

    def build_mask(self, input_tensor):
        mask = tf.linalg.band_part(tf.ones_like(input_tensor)*-1e12, 0, -1) #upper triangular ones, INCLUDING diagonal
        return mask

    def call(self, query, value=None, key=None):
        # If value is none, we assume self attention and set value = query
        # if key is none we assume standard usage and set key = value

        if value is None:
            value = query
            self_attention = True
        else:
            self_attention = False
        if key is None:
            key = value

        if self.mode == 'Bahdanau':
            # Bahdaau style attention: score = tanh(Q + V)
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(self.V(tf.nn.tanh(self.W_query(query) + self.W_value(key))), axis=1)
            context_vector = attention_weights * value

        elif self.mode == 'Transformer':

            # Trasformer style attention: score=Q*K/sqrt(dk)
            # in this case keys = values
            if len(query.shape) < len(key.shape):
                query = tf.expand_dims(query,1)

            score_pre_softmax = tf.matmul(self.W_query(query), self.W_value(key), transpose_b=True)
            score_pre_softmax = score_pre_softmax / tf.math.sqrt(tf.cast(tf.shape(value)[-1], tf.float32))

            if self_attention:
                score_pre_softmax += self.build_mask(score_pre_softmax)

            # score_pre_softmax = tf.keras.backend.in_train_phase(x=score_pre_softmax + self.dropout_mask(score_pre_softmax),
            #                                         alt=score_pre_softmax,
            #                                         training=tf.keras.backend.learning_phase())

            score = tf.nn.softmax(score_pre_softmax, axis=-1)

            context_vector = tf.matmul(score, self.W_key(value))

        if self.only_context:
            return context_vector
        else:
            return context_vector, attention_weights

class multihead_attentive_layer(tf.keras.layers.Layer):
    def __init__(self, units_per_head=[80, 80], kernel_size=None,
                 project_to_units=None,
                 use_norm=True,
                 norm_type='layer',
                 L1=0.0, L2=0.0,
                 use_dropout=True, dropout_rate=0.2):
        super(multihead_attentive_layer, self).__init__()
        # units is a list of lists, containing the numbers of units in the attention head
        self.num_heads = len(units_per_head)
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.use_dropout = use_dropout
        if project_to_units is None:
            self.projection = False
        else:
            self.projection = True
            self.projection_layer = tf.keras.layers.Conv1D(project_to_units,
                                         activation=tf.nn.sigmoid,
                                         strides=1,
                                         kernel_size=1,
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                         padding='causal',
                                         kernel_initializer=initializer,
                                         use_bias=False)
            if self.use_dropout:
                self.projection_layer = Dropout_wrapper(self.projection_layer, dropout_rate=dropout_rate)

        self.multihead_attention = []

        for head in range(self.num_heads):
            attention = Attention(units_per_head[head],
                                  mode='Transformer',
                                  query_kernel= 1 if kernel_size==None else kernel_size[head],
                                  key_kernel=1 if kernel_size==None else kernel_size[head],
                                  use_dropout=use_dropout,
                                  dropout_rate=dropout_rate,
                                  L2=L2, L1=L1,
                                  only_context=True)
            # if self.use_dropout:
            #     attention = Dropout_wrapper(attention, dropout_rate=dropout_rate)

            if self.use_norm:
                attention = Norm_wrapper(attention, norm=norm_type)
            self.multihead_attention.append(attention)

    def call(self, query, value=None):
        if value is None: #Self Attention
            multihead_out = [head(query) for head in self.multihead_attention]
        else:
            multihead_out = [head(query, value) for head in self.multihead_attention]

        multihead_out = tf.concat(multihead_out, axis=-1)

        if self.projection:

            multihead_out = self.projection_layer(multihead_out)

        return multihead_out

########################################################################################################################

class FFW_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=[96,96],
                 use_dropout=True,
                 dropout_rate=0.2,
                 use_norm=True,
                 ):
        super(FFW_block, self).__init__()

        self.layers = []
        self.residuals=False
        for layer_num in range(len(units)):
            layer = tf.keras.layers.Dense(units[layer_num],
                                            activation=activation,
                                            use_bias=True,
                                            kernel_initializer=initializer,
                                            bias_initializer='zeros',
                                            kernel_regularizer=None,
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            bias_constraint=None)
            if use_dropout:
                layer = Dropout_wrapper(layer, dropout_rate, units[layer_num])

            # if layer_num > 0: #can we even residuals
            #     if self.residuals and units[layer_num-1] == units[layer_num]:
            #         layer = Residual_wrapper(layer)

            if use_norm:
                layer = Norm_wrapper(layer)

            self.layers.append(layer)

    def call(self, input):
        out = input
        for layer in self.layers:
            out = layer(out)
        return out

class preactivated_CNN(tf.keras.layers.Layer):
    def __init__(self, num_features, kernel_size, dilation_rate, stride=1, L1=0.0, L2=0.0):
        super(preactivated_CNN, self).__init__()
        self.layer =  tf.keras.layers.Conv1D(filters=num_features,
                                            kernel_size=kernel_size,
                                            activation=activation,
                                            dilation_rate=dilation_rate,
                                            kernel_initializer=initializer,
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                            padding='causal')
    def call(self, signal):
        return self.layer(signal)

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
    def __init__(self, units=[[12,12,12]],
                growth_rate=12,
                squeeze_factor=0.7,
                use_dropout=False,
                dropout_rate=0.0,
                use_norm=False,
                 L1=0.0, L2=0.0,
                downsample_rate=1,
                kernel_sizes=[2, 4],
                residual=False,
                project_to_features=None):
        super(DenseTCN, self).__init__()

        self.units = units
        self.L1 = L1
        self.L2 = L2
        self.num_blocks = len(units)
        self.growth_rate = growth_rate
        # bottleneck features are the 1x1 conv size for each layer
        self.bottleneck_features = 4*growth_rate
        # squeeze factor is the percentage of features passed from one block to the next
        self.squeeze_factor = squeeze_factor
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.downsample_rate = downsample_rate
        self.kernel_sizes = kernel_sizes
        self.use_norm = use_norm
        self.residual = residual
        self.project_to_features = project_to_features
        # build flag
        self.already_built = False

        if self.use_norm:
            self.norm = []
        # create a stack of squeeze layers to be used between blocks (is not used after the last block)
        self.squeeze = []

    # build a stack for one layer of the network
    def build_layer(self, num_layer, kernel_size):
        layer = {}
        #Bottleneck features
        bottleneck_layer = preactivated_CNN(self.bottleneck_features, kernel_size=1, dilation_rate=1, L2=self.L2, L1=self.L1)
        if self.use_norm:
            bottleneck_layer = Norm_wrapper(bottleneck_layer, norm='batch')
        if self.use_dropout:
            bottleneck_layer = Dropout_wrapper(bottleneck_layer, self.dropout_rate)
        layer['bottleneck_layer'] = bottleneck_layer

        # Growth feature layer
        growth_layer = preactivated_CNN(self.growth_rate, kernel_size=kernel_size, dilation_rate=2**num_layer, L2=self.L2, L1=self.L1)
        if self.use_norm:
            growth_layer = Norm_wrapper(growth_layer, norm='batch')
        if self.use_dropout:
            growth_layer = Dropout_wrapper(growth_layer, self.dropout_rate)
        layer['growth_layer'] = growth_layer
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
            self.num_layers_per_block = len(self.units[block])
            all_streams = []
            for kernel_size in self.kernel_sizes:
                all_streams.append(self.build_stream(kernel_size))
            self.blocks.append(all_streams)
            # calculate the number of features in the squeeze layer
            # shape of input to the block + growth rate for all layers in the block
            # if there are residual connections, the shape of output is equal to the shape of input (only at the last block)
            if block == 0 and not self.residual:
                num_features = int(self.squeeze_factor*(in_shape+self.num_layers_per_block*self.growth_rate))
            # If Dense block is first, we don't want the same shape as input, so an additional parameter is required
            elif block == self.num_blocks-1 and self.residual:
                if self.project_to_features is None:
                    num_features = in_shape
                else:
                    num_features = self.project_to_features
            else:
                num_features = int(self.squeeze_factor*(num_features + self.num_layers_per_block*self.growth_rate))
            # create stack of squeeze layers
            squeeze_layer = preactivated_CNN(num_features, kernel_size=1, dilation_rate=1, stride=1, L2=self.L2, L1=self.L1)
            if self.use_norm:
                squeeze_layer = Norm_wrapper(squeeze_layer, norm='batch')
            if self.use_dropout:
                squeeze_layer = Dropout_wrapper(squeeze_layer, self.dropout_rate)
            self.squeeze.append(squeeze_layer)
            if self.downsample_rate > 1:
                self.pool = tf.keras.layers.AvgPool1D(pool_size=self.downsample_rate,
                                                      strides=self.downsample_rate)
        # update build flag
        self.already_built = True
        return None

    # run one basic dense network layer
    # return output concatednated with input
    def run_layer(self, block, num_layer, num_stream, inputs):
        out = inputs

        #bottleneck section
        if inputs.shape[-1] > self.bottleneck_features:
            out = self.blocks[block][num_stream][num_layer]['bottleneck_layer'](out)

        # growth section
        out = self.blocks[block][num_stream][num_layer]['growth_layer'](out)
        # print(out.shape)
        return tf.concat([out, inputs], axis=-1)

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
                out = tf.concat([out, self.run_stream(block, num_stream, inputs)], -1)
        # return output concatednated with input
        return out


    def call(self, inputs):
        out = inputs
        # call the init function upon first call
        if not self.already_built:
            self.build_block(inputs.shape[-1])
        # iterate through blocks
        for block in range(self.num_blocks):
            out = self.run_block(block, out)
            if self.use_norm:
                out = self.norm[block](out)
            out = self.squeeze[block](out)
            if self.downsample_rate >1:
                out = self.pool(out)
        return out

class project_input_to_4k_layer(tf.keras.layers.Layer):
    def __init__(self,
                 growth_rate=12,
                 kernel=3,
                 use_dropout=True,
                 dropout_rate=0.2,
                 L1=0.0,
                 L2=0.0,
                 use_norm=False,
                 ):
        super(project_input_to_4k_layer, self).__init__()
        self.input_to_4k_features =  preactivated_CNN(num_features=4*growth_rate,
                                                      kernel_size=2,
                                                      dilation_rate=1,
                                                      L2=L2, L1=L1)
        if use_norm:
            self.input_to_4k_features = Norm_wrapper(self.input_to_4k_features, norm='batch')
        if use_dropout:
            self.input_to_4k_features = Dropout_wrapper(self.input_to_4k_features, dropout_rate)

    def call(self, signal):
        signal = self.input_to_4k_features(signal)
        return signal

########################################################################################################################

class block_LSTM(tf.keras.layers.Layer):
    def __init__(self, units=[[20, 20], [20,20]],
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_norm=False,
                 use_hw=False, use_residual=False,
                 L1=0.0, L2=0.0,
                 return_state=True):
        super(block_LSTM, self).__init__()
        self.return_state = return_state
        self.encoder_blocks = []
        self.cropping_layer = Cropping_layer(0.6)
        for block in units:
            self.encoder_blocks.append(MultiLayer_LSTM(units=[block],
                                                use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                use_norm=use_norm,
                                                use_hw=use_hw, use_residuals=use_residual,
                                                L1=L1, L2=L2))

    def call(self, signal, initial_states=None):
        signal = self.cropping_layer(signal)
        encoder_outs = []
        encoder_last_states = []
        
        for block in range(len(self.encoder_blocks)):
            signal, block_states = self.encoder_blocks[block](signal, initial_states=initial_states)
            encoder_outs.append(signal)
            encoder_last_states.append(block_states[0])
            
        if self.return_state:
            return encoder_outs, encoder_last_states
        else:
            return encoder_outs

class decoder_LSTM_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=[[20, 20], [20,20]],
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_hw=False, use_residual=False,
                 use_norm=False,
                 L1=0.0, L2=0.0,
                 use_attention=False, attention_heads=3,
                 projection_layer=None):
        super(decoder_LSTM_block, self).__init__()
        self.units = units
        self.use_attention = use_attention
        self.projection_layer=projection_layer

        self.cropping_layer = Cropping_layer(0.6)

        if self.use_attention:
            self.attention_blocks = []
        self.decoder_blocks = []

        for block in units:
            self.decoder_blocks.append(MultiLayer_LSTM(units=[block],
                                                use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                use_norm=use_norm,
                                                use_hw=use_hw, use_residuals=use_residual,
                                                L1=L1, L2=L2))
            if self.use_attention:
                attention_units = int(block[-1] / 2)
                if attention_heads == 1:
                    #equivalent to the dimensionality of attention heads
                    attention = Attention(attention_units,
                                        mode='Transformer',
                                        query_kernel= 1,
                                        key_kernel=1,
                                        use_dropout=use_dropout,
                                        dropout_rate=dropout_rate,
                                        L2=L2, L1=L1,
                                        only_context=True)

                else:
                    units_per_head =[attention_units]*attention_heads
                    attention = multihead_attentive_layer(units_per_head=units_per_head,
                                                          kernel_size=None,
                                                          project_to_units=int(np.sum(units_per_head)/2),
                                                          dropout_rate=dropout_rate,
                                                          use_dropout=use_dropout,
                                                          use_norm=use_norm,
                                                          L1=L1,
                                                          L2=L2, )
                self.attention_blocks.append(attention)

    def call(self, prev_history, teacher, attention_value, timesteps, decoder_init_state=None):

        # decoder state is supposed to be the storage for the states
        # block state is the temporal buffer, because the layout suxx
        forecast = tf.keras.backend.in_train_phase(x=self.teacher_call(prev_history, teacher,
                                                                     decoder_init_state,
                                                                     attention_value,
                                                                     timesteps),
                                                   alt=self.self_recurrent_call(prev_history,
                                                                                decoder_init_state,
                                                                                attention_value,
                                                                                timesteps),
                                                   training=tf.keras.backend.learning_phase())

        return forecast

    def teacher_call(self, prev_history, teacher, decoder_init_state, attention_value, timesteps):
        prev_history = self.cropping_layer(prev_history)
        full_history = tf.concat([prev_history, teacher], axis=1)
        unprojected_forecast = self.decode_layerwise(full_history, decoder_init_state, attention_value)
        forecast = self.projection_layer(unprojected_forecast[:,-timesteps:,:])
        return forecast

    def decode_layerwise(self, input, decoder_state, attention_value):
        signal=input
        for num_block in range(len(self.decoder_blocks)):
            block_state = [decoder_state[num_block]] if decoder_state else None

            signal, block_state = self.decoder_blocks[num_block](signal, initial_states=block_state)

            if self.use_attention:
                attention_context = self.attention_blocks[num_block](signal, value=attention_value[num_block])
                signal = tf.concat([signal, attention_context], axis=-1)

            # #ToDO: Tensorflow ragged tensors
            # for step in range(num_steps):
            #     signal_step = signal[:,step,:]
            #     signal_step = tf.expand_dims(signal_step, axis=1)
            #     signal_step, block_state = self.decoder_blocks[num_block](signal_step, initial_states=block_state)
            #     print('.')
            #
            #     concat_states = tf.concat(block_state[0][-1], axis=-1)
            #     concat_states = tf.expand_dims(concat_states, axis=1)
            #     states_history = tf.concat([states_history, concat_states], axis=1)
            #     out = tf.concat([out, signal_step], axis=1)
            #
            # signal = out
            # if self.use_attention:
            #     # attention value is the output of the encoder with no states
            #     attention_context = self.attention_blocks[num_block](states_history, value=attention_value[num_block])
            #     signal = tf.concat([signal, attention_context], axis=-1)

        return signal

    def self_recurrent_call(self, prev_history, decoder_init_state, attention_value, timesteps):
        decoder_state = decoder_init_state  # [[[state_h, state_c]]] #(blocks, layers_in_block, 2)

        for timestep in range(timesteps + prev_history.shape[1] - 1):
            if timestep < prev_history.shape[1]:
                input = tf.expand_dims(prev_history[:,timestep,:], axis=1)
                decoder_out, decoder_state = self.decode_stepwise(input, decoder_state, attention_value)
                if timestep == prev_history.shape[1] - 1:
                    forecast = self.projection_layer(decoder_out)
            else:
                input = tf.expand_dims(forecast[:, -1, :], axis=1)
                decoder_out, decoder_state = self.decode_stepwise(input, decoder_state, attention_value)
                forecast = tf.concat([forecast, self.projection_layer(decoder_out)], axis=1)

        return forecast

    def decode_stepwise(self, input, decoder_state, attention_value):
            # attention layers and assign those to the decoder states later.
            signal = input

            block_states = []
            for num_block in range(len(self.decoder_blocks)):
                    
                signal, block_state = self.decoder_blocks[num_block](signal, 
                                                                    initial_states=[decoder_state[num_block]] if decoder_state else None,
                                                                    )
                block_states.append(block_state[0])

                if self.use_attention:
                    query = tf.concat(block_state[0][-1] , axis=-1)
                    query = tf.expand_dims(query, axis=1)
                    attention_context = self.attention_blocks[num_block](signal, value=attention_value[num_block])
                    signal = tf.concat([signal, attention_context], axis=-1)

            decoder_state = block_states
            return signal, decoder_state


class classic_attn_decoder_LSTM_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=[[20, 20], [20,20]],
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_hw=False, use_residual=False,
                 use_norm=False,
                 L1=0.0, L2=0.0,
                 use_attention=False, attention_heads=3,
                 projection_layer=None):
        super(classic_attn_decoder_LSTM_block, self).__init__()
        self.units = units
        self.use_attention = use_attention
        self.projection_layer=projection_layer

        if self.use_attention:
            self.attention_blocks = []
        self.decoder_blocks = []

        if self.use_attention:
            self.attention = Attention(units[-1][-1],
                                    mode='Transformer',
                                    query_kernel= 1,
                                    key_kernel=1,
                                    use_dropout=use_dropout,
                                    dropout_rate=dropout_rate,
                                    L2=L2, L1=L1,
                                    only_context=True)

        for block in units:
            self.decoder_blocks.append(MultiLayer_LSTM(units=[block],
                                                use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                use_norm=use_norm,
                                                use_hw=use_hw, use_residuals=use_residual,
                                                L1=L1, L2=L2))

    def call(self, zeroth_step, decoder_init_state, attention_value=None, timesteps=12):
        signal = zeroth_step
        decoder_state = decoder_init_state # [[[state_h, state_c]]] #(blocks, layers_in_block, 2)
        # decoder state is supposed to be the storage for the states
        # block state is the temporal buffer, because the layout suxx
        # ToDo: this is throwing errors for anything with
        for timestep in range(timesteps):

            # Create an empty list to get states on the LSTM from between
            # attention layers and assign those to the decoder states later.
            block_states = []
            # if self.use_attention:
            #     context = self.attention(signal, value=attention_value)
            #     signal = tf.concat([signal, context], axis=-1)

            for num_block in range(len(self.decoder_blocks)):
                signal, block_state = self.decoder_blocks[num_block](signal, initial_states=[decoder_state[num_block]])
                block_states.append(block_state[0])
            decoder_state = block_states

            if timestep == 0:
                signal = self.projection_layer(signal)
                decoder_out = signal
            else:
                signal = self.projection_layer(signal)
                decoder_out = tf.concat([decoder_out, signal], axis=1)

        return decoder_out

########################################################################################################################

class attentive_TCN(tf.keras.layers.Layer):
    '''
    Encoder consisting of alternating self-attention and Desne TCN blocks
    requires units in shape of [[attention block units], [dense tcn units] ... ]
    the first block is always self-attention in order to get the most information from the input
    outputs after each block (dense tcn and self-attention) are concatenated
    '''
    def __init__(self,
                 units=None,
                 use_dropout=False,
                 dropout_rate=0.0,
                 attention_heads=3,
                 L1=0.0, L2=0.0,
                 use_norm=False,):
        super(attentive_TCN, self).__init__()
        self.use_norm = use_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        #ToDo: Figure out how we want to do the parameter growth thingie
        kernel_size = [2]
        self.input_projection = project_input_to_4k_layer(growth_rate=units[0][0],
                                                          kernel=kernel_size,
                                                          use_dropout=use_dropout,
                                                          dropout_rate=dropout_rate,
                                                          L1=L1, L2=L2,
                                                          use_norm=use_norm)
        self.transform = []
        self.self_attention = []
        self.self_attention_context = [[]]*len(units)
        for block in range(len(units)):
            Dense_params = {'units': [units[block][:-1]],
                            'growth_rate': units[block][0],
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'L1':L1,
                            'L2':L2,
                            'use_norm': use_norm,
                            'kernel_sizes': kernel_size,
                            'residual':True,
                            'project_to_features': units[block][-1]
                             }
            transform = DenseTCN(**Dense_params)
            if self.use_norm:
                transform = Norm_wrapper(transform, norm='batch')
            self.transform.append(transform)

            attn_units_per_head = [int(attention_heads / (attention_heads + 1) * units[block][-1])] * attention_heads
            self_attention = multihead_attentive_layer(units_per_head=attn_units_per_head,
                                                       use_norm=use_norm,
                                                       norm_type='batch',
                                                       L1=L1, L2=L2,
                                                       project_to_units=units[block][-1],
                                                       use_dropout=use_dropout, dropout_rate=dropout_rate)
            self_attention = Residual_wrapper(self_attention)
            if use_norm:
                self_attention=Norm_wrapper(self_attention)
            self.self_attention.append(self_attention)

    def call(self, inputs):
        # Input to first transformation layer, assign last state and out
        out = self.input_projection(inputs)
        for block in range(len(self.transform)):
            out = self.transform[block](out)
            out = self.self_attention[block](out)

        return out

class generator_Dense_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=None,
                 use_dropout=False,
                 dropout_rate=0.0,
                 attention_heads=3,
                 L1=0.0, L2=0.0,
                 use_norm=False,
                 projection=tf.keras.layers.Dense(20)):
        super(generator_Dense_block, self).__init__()
        self.use_norm = use_norm
        self.projection = projection

        self.input_projection = preactivated_CNN(units[0][0], kernel_size=2,dilation_rate=1)

        if self.use_norm:
            self.input_projection = Norm_wrapper(self.input_projection)

        self.transform = []
        self.self_attention = []
        self.attention = []
        # initialize the transform and attention blocks
        for block in range(len(units)):
            transform = preactivated_CNN(units[block][-1], kernel_size=2, dilation_rate=2**block)

            if self.use_norm:
                transform = Norm_wrapper(transform, norm='batch')
            self.transform.append(transform)

            self_attn = Attention(units[block][-1])
            if self.use_norm:
                self_attn = Norm_wrapper(self_attn)
            self.self_attention.append(self_attn)

            attn = Attention(units[block][-1])
            if self.use_norm:
                attn = Norm_wrapper(attn)
            self.attention.append(attn)

    # zero padding
    def zero_pad(self, thing_to_pad, input_shape):
        if  not thing_to_pad.shape[1] < self.history_length + self.forecast_timesteps -1:
            padded_out = thing_to_pad
        else:
            steps_to_pad = self.history_length + self.forecast_timesteps-1 - thing_to_pad.shape[1]
            zero_pad = tf.zeros(shape=[input_shape[0], steps_to_pad, thing_to_pad.shape[2]])
            padded_out = tf.concat([zero_pad, thing_to_pad], axis=1)
        return padded_out

    # in case of teacher forcing or histoy processing
    def process_one_step(self, data):
        self.buffer = data
        out = self.input_projection(data)
        out = self.zero_pad(out, tf.shape(data))
        for block in range(len(self.transform)):
            out = self.transform[block](out)
            out = self.self_attention[block](out)
            out = self.attention[block](out, value=self.attention_value)
        self.forecast = self.projection(out)
        return None

    # use when validating and forecasting step by step
    def unroll(self, data):
        self.buffer = tf.concat([self.buffer, data], axis=1)
        out = self.input_projection(self.buffer)
        out = self.zero_pad(out, tf.shape(data))
        for block in range(len(self.transform)):
            out = self.transform[block](out)
            out = self.self_attention[block](out)
            out = self.attention[block](out, value=self.attention_value)
        out = self.projection(out)
        self.forecast = tf.concat([self.forecast, out[:,-1:,:]], axis=1)
        return None

    def training_call(self, history_and_teacher):
        self.process_one_step(history_and_teacher)
        forecast = self.forecast[:,-self.forecast_timesteps:,:]
        return forecast

    def validation_call(self, history):
        self.process_one_step(history)
        for step in range(self.forecast_timesteps-1):
            self.unroll(self.forecast[:,-1:,:])
        forecast = self.forecast[:,-self.forecast_timesteps:,:]
        return forecast


    def call(self, history, attention_value, teacher, timesteps=12):
        self.attention_value = attention_value
        self.forecast_timesteps = timesteps
        self.history_length = history.shape[1]

        forecast = tf.keras.backend.in_train_phase(self.training_call(tf.concat([history, teacher], axis=1)),
                                        alt=self.validation_call(history),
                                        training=tf.keras.backend.learning_phase())

        return forecast

########################################################################################################################

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


########################################################################################################################
