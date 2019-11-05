# contains all the building blocks necessary to build the architectures we will be working on
# In a sense, this will be the main modiefied file


import tensorflow as tf
import numpy as np
########################################################################################################################
'''General help functions, lowest level'''

def drop_features_of_signal(input, dropout_rate):
    batch_dim = tf.shape(input)[0]
    # if batch_dim is None:
    #     batch_dim = 128
    return tf.keras.backend.in_train_phase(
                tf.nn.dropout(input, dropout_rate, noise_shape=(batch_dim, 1, input.shape[2])),
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
########################################################################################################################
'''Wrappers for feedforward architectures'''
# important to note that all wrappers have to be compatible with both, attention(query, value) and self_attention(query)

class Dropout_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, dropout_rate, units):
        super(Dropout_wrapper, self).__init__(layer)
        # self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, noise_shape=(1, units))
        self.dropout_rate = dropout_rate
    def call(self, inputs, value=None):

        if value is None:
            return drop_features_of_signal(self.layer(inputs), self.dropout_rate)
        else:
            return drop_features_of_signal(self.layer(inputs, value), self.dropout_rate)

class Norm_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, norm='layer'):
        super(Norm_wrapper, self).__init__(layer)

        if norm == 'layer':
            self.norm = tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True, )
        else:
            print('norm type', norm, 'not found, check the code to see if it is a typo')

    def call(self, inputs, value=None):
        if value is None:
            return self.norm(self.layer(inputs))
        else:
            return self.norm(self.layer(inputs, value))

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

class Residual_LSTM_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer):
        super(Residual_LSTM_wrapper, self).__init__(layer)
        # self.transform_gate = tf.keras.layers.TimeDistributed(self.transform_gate )

    def call(self, input, initial_states=None):
        output, states = self.layer(input, initial_states)
        output[-1] = output[-1]+input
        return output, states

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
    def __init__(self, units=[[20, 20], [20,20]], use_dropout=True, dropout_rate=0.0, use_norm=True, use_hw=True, use_quasi_dense=True):
        super(MultiLayer_LSTM, self).__init__()

        self.units = units
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.use_hw = use_hw
        self.use_quasi_dense = use_quasi_dense

        self.LSTM_list_o_lists = []  # Multiple layers work easiest as a list of layers, so here we start
        if self.use_norm:
            self.LSTM_norm_list_o_lists = []
        if self.use_dropout:
            self.LSTM_dropout_list_o_lists = []
        if self.use_hw:
            self.T_x_list_o_lists = []
        if self.use_quasi_dense:
            self.dense_drop = tf.keras.layers.Dropout(rate=dropout_rate)

        for block in range(len(self.units)):
            LSTM_list = []
            LSTM_norm_list = []
            LSTM_dropout_list = []
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
                # norm if norming and not the very last layer
                if block < (len(self.units) + 1) or layer < (len(self.units[block])):
                    very_last_layer = True
                else:
                    very_last_layer = False

                # do we wanna norm
                if self.use_norm and not very_last_layer:
                    LSTM_norm_list.append(tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True))

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

                # norm if norming and not the very last layer
                if block < (len(self.units) + 1) or layer < (len(self.units[block])):
                    very_last_layer = True
                else:
                    very_last_layer = False

                if self.use_dropout:
                    layer_out = drop_features_of_signal(layer_out, self.dropout_rate)

                if self.use_norm and not very_last_layer:
                    print('norming')
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
                 ):
        super(Attention, self).__init__()
        self.mode = mode
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.only_context = only_context
        self.causal_attention = causal_attention
        self.W_query = tf.keras.layers.Conv1D(units,
                                         activation=tf.nn.sigmoid,
                                         strides=query_stride,
                                         kernel_size=query_kernel,
                                         padding='causal',
                                         kernel_initializer='glorot_uniform',
                                         use_bias=False)
        if use_dropout:
            self.W_query = Dropout_wrapper(self.W_query, dropout_rate=dropout_rate, units=units)

        self.W_value = tf.keras.layers.Conv1D(units,
                                         activation=tf.nn.sigmoid,
                                         strides=1,
                                         kernel_size=1,
                                         padding='causal',
                                         kernel_initializer='glorot_uniform',
                                         use_bias=False)
        if use_dropout:
            self.W_value = Dropout_wrapper(self.W_value, dropout_rate=dropout_rate, units=units)

        if mode == 'Bahdanau':
            self.V = tf.keras.layers.Dense(1)
        elif mode == 'Transformer':
            self.W_key = tf.keras.layers.Conv1D(units,
                                         activation=tf.nn.sigmoid,
                                         strides=key_stride,
                                         kernel_size=key_kernel,
                                         padding='causal',
                                         kernel_initializer='glorot_uniform',
                                         use_bias=False)
            if use_dropout:
                self.W_key = Dropout_wrapper(self.W_key, dropout_rate=dropout_rate, units=units)

    def build_mask(self, input_shape):
        # create mask to wipe out the future timesteps
        triangular_mask = tf.ones(shape=[input_shape[1],input_shape[1]], dtype=tf.float32)
        # upper_triangular_ones = tf.linalg.band_part(ones, 0, -1) #this is where we want to pass information through
        # return tf.where(upper_triangular_ones==1.0, x=0.0, y=-np.inf) #replace strictly lower part with -infs, upper part and diag with 0s
        #alternative, strictlu upper_triangular:
        upper_triangular_mask = tf.linalg.band_part(triangular_mask, 0, -1) #upper triangular ones, INCLUDING diagonal
        mask = tf.where(upper_triangular_mask==1.0,
                        x=-np.inf,
                        y=0.0)
        return mask

    def call(self, query, value=None, key=None):
        # If value is none, we assume self attention and set value = query
        # if key is none we assume standard usage and set key = value

        if value is None:
            value = query
            mask = self.build_mask(query.shape)
            self_attention = True
        else:
            self_attention = False
        if key is None:
            key = value
        #
        # if self.causal_attention and not self.mask_built:
        #     self.__build_causal_mask(len_value=query.shape[1])
        # query = self.W_query(query)

        # if self.causal_attention:
        #     query = query * self.causal_mask

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
            # ToDo: add mask matrix?
            # (len_query, dims_query) matmul (dims_value, len_value) = (len_query, len_value)
            score = tf.matmul(self.W_query(query), self.W_value(key), transpose_b=True)
            score = score / tf.math.sqrt(tf.cast(tf.shape(value)[-1], tf.float32))

            if self_attention:
                score = tf.add(score, mask)
                #print(score.shape)
                zeroth_row_replacement = tf.zeros(shape=[tf.shape(score)[0], 1, score.shape[-1]])
                rest_context = tf.nn.softmax(score[:,1:,:], axis=-1)
                score = tf.concat([zeroth_row_replacement, rest_context], axis=1)

                # score =tf.nn.softmax(score, axis=-1)
                # score = tf.where(tf.math.is_nan(score),
                #                  x=0.0,
                #                  y=score)
            else:
                score = tf.nn.softmax(score, axis=-1)

            if self.use_dropout:
                score = drop_timesteps_of_signal(score, self.dropout_rate)

            context_vector = tf.matmul(score, self.W_key(value))

        if self.only_context:
            return context_vector
        else:
            return context_vector, attention_weights

class multihead_attentive_layer(tf.keras.layers.Layer):
    def __init__(self, units_per_head=[80, 80], kernel_size=None,
                 project_to_units=None,
                 use_norm=True,
                 use_dropout=True, dropout_rate=0.2):
        super(multihead_attentive_layer, self).__init__()
        # units is a list of lists, containing the numbers of units in the attention head
        self.num_heads = len(units_per_head)

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
                                         padding='causal',
                                         kernel_initializer='glorot_uniform',
                                         use_bias=False)
            if self.use_dropout:
                self.projection_layer = Dropout_wrapper(self.projection_layer, dropout_rate=dropout_rate, units=project_to_units)

        self.multihead_attention = []
        if self.use_norm:
            self.norm = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=False)
        if self.use_dropout:
            self.dropout = tf.keras.layers.Dropout(dropout_rate, noise_shape=(1, int(np.sum(units_per_head))))

        for head in range(self.num_heads):
            attention = Attention(units_per_head[head],
                                  mode='Transformer',
                                  query_kernel= 1 if kernel_size==None else kernel_size[head],
                                  key_kernel=1 if kernel_size==None else kernel_size[head],
                                  use_dropout=use_dropout,
                                  dropout_rate=dropout_rate,
                                  only_context=True)

            self.multihead_attention.append(attention)

    def call(self, query, value=None):
        if value is None: #Self Attention
            multihead_out = [head(query) for head in self.multihead_attention]
        else:
            multihead_out = [head(query, value) for head in self.multihead_attention]

        multihead_out = tf.concat(multihead_out, axis=-1)

        if self.projection:
            if self.use_norm:
                multihead_out = self.norm(multihead_out)
            if self.use_dropout:
                multihead_out = self.dropout(multihead_out, training=tf.keras.backend.learning_phase())

            multihead_out = self.projection_layer(multihead_out)

        return multihead_out

########################################################################################################################

class FFW_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=[[96,96]],
                 use_dropout=True,
                 dropout_rate=0.2,
                 use_norm=True,
                 ):
        super(FFW_block, self).__init__()

        self.layers = []
        self.residuals=False
        for block in range(len(units)):
            for layer_num in range(len(units[block])):
                layer = tf.keras.layers.Dense(units[block][layer_num],
                                              activation=tf.nn.elu,
                                              use_bias=True,
                                              kernel_initializer='glorot_uniform',
                                              bias_initializer='zeros',
                                              kernel_regularizer=None,
                                              bias_regularizer=None,
                                              activity_regularizer=None,
                                              kernel_constraint=None,
                                              bias_constraint=None)
                if use_dropout:
                    layer = Dropout_wrapper(layer, dropout_rate, units[block][layer_num])

                # if layer_num > 0: #can we even residuals
                #     if self.residuals and units[block][layer_num-1] == units[block][layer_num]:
                #         layer = Residual_wrapper(layer)

                # norm if norming and not the very last layer
                if block < (len(units) + 1) or layer < (len(units[block])):
                    very_last_layer = True
                else:
                    very_last_layer = False

                if use_norm and not very_last_layer:
                    layer = Norm_wrapper(layer)

                self.layers.append(layer)

    def call(self, input):
        out = input
        for layer in self.layers:
            out = layer(out)
        return out

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
                squeeze_factor=0.5,
                use_dropout=False,
                dropout_rate=0.0,
                use_norm=False,
                downsample_rate=1,
                kernel_sizes=[3, 5],
                residual=False,
                project_to_features=None):
        super(DenseTCN, self).__init__()

        self.units = units
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

    # initialize one CNN layer with arbitrary parameters
    def cnn_layer(self, num_features, kernel_size, dilation_rate, stride=1):
        return tf.keras.layers.Conv1D(filters=num_features,
                                        kernel_size=kernel_size,
                                        dilation_rate=dilation_rate,
                                        strides=stride,
                                        padding='causal')

    # build a stack for one layer of the network
    def build_layer(self, num_layer, kernel_size):
        layer = []
        #Bottleneck features
        if self.use_norm:
            layer.append(tf.keras.layers.BatchNormalization(axis=-1, center=True))
        else:
            layer.append([])
        layer.append(self.cnn_layer(self.bottleneck_features, kernel_size=1, dilation_rate=1))
        if self.use_dropout:
            layer.append(tf.keras.layers.Dropout(rate=self.dropout_rate, noise_shape=(1,self.bottleneck_features)))
        else:
            layer.append([])
        # Growth feature layer
        if self.use_norm:
            layer.append(tf.keras.layers.BatchNormalization(axis=-1, center=True))
        else:
            layer.append([])
        #ToDo: Fix this thing pltz!!
        layer.append(self.cnn_layer(self.growth_rate, kernel_size=kernel_size, dilation_rate=1))
        if self.use_dropout:
            layer.append(tf.keras.layers.Dropout(rate=self.dropout_rate, noise_shape=(1,self.growth_rate)))
        else:
            layer.append([])
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
            if self.use_norm:
                self.norm.append(tf.keras.layers.BatchNormalization(axis=-1, center=True))
            self.squeeze.append(self.cnn_layer(num_features, kernel_size=1, dilation_rate=1, stride=1))
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
        # self.blocks[block][num_stream][num_layer] = [Bottleneck_norm, Bottleneck_features, bottleneck_dropout,
        #                                              growth_norm, growth_layer, growth_dropout]
        # Bottleneck section
        if self.use_norm:
            out = self.blocks[block][num_stream][num_layer][0](out)
        out = tf.keras.activations.relu(out)
        out = self.blocks[block][num_stream][num_layer][1](out)
        if self.use_dropout:
            out = self.blocks[block][num_stream][num_layer][2](out, training=tf.keras.backend.learning_phase())
        # growth section
        if self.use_norm:
            out = self.blocks[block][num_stream][num_layer][3](out)
        out = tf.keras.activations.relu(out)
        out = self.blocks[block][num_stream][num_layer][4](out)
        if self.use_dropout:
            out = self.blocks[block][num_stream][num_layer][5](out, training=tf.keras.backend.learning_phase())
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

########################################################################################################################

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
        if use_norm:
            self.use_norm = use_norm
            self.last_layer_norm = tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True)

    def call(self, encoder_inputs, initial_states=None):
        block_output, block_states = self.multilayer_LSTMs(encoder_inputs, initial_states=initial_states)

        if self.use_norm:
            block_output[-1] = self.last_layer_norm(block_output[-1])
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
                units = units[-1][-1]
                heads=2
                self.attention = multihead_attentive_layer(units_per_head=[int(units/heads)]*heads,
                                                           kernel_size=[1,1],
                                                           use_norm=use_norm,
                                                           ) # choose the attention style (Bahdanau, Transformer)
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

class generator_LSTM_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=[[20, 20], [20,20]],
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_hw=True,
                 use_norm=True,
                 use_quasi_dense=True,
                 use_attention=False,
                 only_last_layer_output= True,
                 projection=tf.keras.layers.Dense(20)):
        super(generator_LSTM_block, self).__init__()
        self.use_norm = use_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        #ToDo: Figure out how we want to do the parameter growth thingie

        # LSTMs / transformation layers
        self.ml_LSTM_params = {'units': units,
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_norm': use_norm,
                        'use_hw': use_hw,
                        'use_quasi_dense': use_quasi_dense,
                        }
        self.LSTM_history = MultiLayer_LSTM(**self.ml_LSTM_params)
        self.projection = projection

        # Self attention and attention layers
        units = units[-1][-1]
        heads = 2
        kernel_sizes = [1, 1]
        self.max_kernel_size = np.amax(kernel_sizes)

        self.attention = multihead_attentive_layer(units_per_head=[int((2/3)*units)] * heads,
                                                   use_norm=False,
                                                   project_to_units=units,
                                                   use_dropout=use_dropout, dropout_rate=dropout_rate)  # choose the attention style (Bahdanau, Transformer
        self.attention = Residual_wrapper(self.attention)

        self.LSTM_post_attn = MultiLayer_LSTM(**self.ml_LSTM_params)
        self.LSTM_post_attn = Residual_LSTM_wrapper(self.LSTM_post_attn)

        # Norm layers for everything, so we can do residuals and then norm
        if self.use_norm:
            self.attn_norm = tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True)
            self.self_attn_norm = tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True)
            self.LSTM1_output_norm = tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True)
            self.LSTM2_output_norm = tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True)


    def process(self, input_steps):

        # Input to first transformation layer, assign last state and out
        LSTM1_out, self.LSTM1_last_states = self.LSTM_history(input_steps, initial_states=self.LSTM1_last_states)
        LSTM1_out = LSTM1_out[-1] #LSTM specific bcyz multilayerlstm
        if self.use_norm:
            LSTM1_out = self.LSTM1_output_norm(LSTM1_out)

        attention_in = LSTM1_out
        attention_features = self.attention(attention_in, value=self.attention_value)
        if self.use_norm:
            attention_features = self.attn_norm(attention_features)

        LSTM2_out, self.LSTM2_last_states = self.LSTM_post_attn(attention_features, initial_states=self.LSTM2_last_states)
        LSTM2_out = LSTM2_out[-1]
        if self.use_norm:
            LSTM2_out = self.LSTM2_output_norm(LSTM2_out)

        #log the history
        if not self.started:
            self.LSTM2_history = LSTM2_out
        else:
            self.LSTM2_history = tf.concat([self.LSTM2_history, LSTM2_out], axis=1)

    def call(self, history, attention_value, forecast_timesteps=12, teacher=None):
        self.attention_value = attention_value

        forecast = tf.keras.backend.in_train_phase(self.training_call(tf.concat([history, teacher], axis=1), forecast_timesteps),
                                        alt=self.validation_call(history, forecast_timesteps),
                                        training=tf.keras.backend.learning_phase())

        return forecast

    def training_call(self, history_and_teacher, forecast_timesteps):
        self.LSTM1_last_states = None
        self.LSTM2_last_states = None
        self.started = False
        if self.use_dropout:
            history_and_teacher = drop_timesteps_of_signal(history_and_teacher, self.dropout_rate)
        self.process(history_and_teacher)
        self.started = True
        return self.projection(self.LSTM2_history[:,-forecast_timesteps:,:])

    def validation_call(self, history, forecast_timesteps):
        self.LSTM1_last_states = None
        self.LSTM2_last_states = None
        self.started = False
        self.process(history)
        self.started = True

        for step in range(forecast_timesteps - 1):
            self.process(self.projection(self.LSTM2_history[:, -1:, :]))

        return self.projection(self.LSTM2_history[:,-forecast_timesteps:,:])

########################################################################################################################

class FFW_encoder(tf.keras.layers.Layer):
    def __init__(self,
                 units=[[128], [128]],
                 use_dropout=True,
                 dropout_rate=0.2,
                 use_norm=True,
                 ):
        super(FFW_encoder, self).__init__()
        block_kwargs = {'units': units,
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_norm': use_norm,
                        }

        attention_kwargs = {'units_per_head': [int(units[-1][-1]/3)] * 2,
                            'project_to_units': units[-1][-1],
                            'use_norm': use_norm,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate}
        self.tranform_in = FFW_block(**block_kwargs)

        self.self_attention = []
        self.transform = []

        for block in units:
            self_attention=multihead_attentive_layer(**attention_kwargs)
            self_attention=Residual_wrapper(self_attention)
            if use_norm:
                self_attention=Norm_wrapper(self_attention)
            self.self_attention.append(self_attention)
            self.transform.append(FFW_block(**block_kwargs))

    def call(self, inputs):
        outputs = self.tranform_in(inputs)
        for block in range(len(self.self_attention)):
            outputs = self.self_attention[block](outputs)
            outputs = self.transform[block](outputs)

        return outputs

class FFW_generator(tf.keras.layers.Layer):

    def __init__(self,
                 units=[[128], [128]],
                 use_dropout=True,
                 dropout_rate=0.2,
                 use_norm=True,
                 projection=tf.keras.layers.Dense(20)
                 ):
        super(FFW_generator, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        block_kwargs = {'units': units,
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_norm': use_norm,
                        }

        attention_kwargs = {'units_per_head': [int(units[-1][-1]/3)] * 2,
                            'project_to_units': units[-1][-1],
                            'use_norm': use_norm,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate}

        self.after_transform_in_history = []
        self.self_attention = []
        self.attention = []
        self.transform_out = []

        self.transform_in = FFW_block(**block_kwargs)
        if use_norm:
            self.transform_in = Norm_wrapper(self.transform_in)
        self.after_transform_in_history = None
        for block in units:
            self_attention = multihead_attentive_layer(**attention_kwargs)
            self_attention = Residual_wrapper(self_attention)
            if use_norm:
                self_attention=Norm_wrapper(self_attention)
            self.self_attention.append(self_attention)

            attention = multihead_attentive_layer(**attention_kwargs)
            attention = Residual_wrapper(attention)
            if use_norm:
                attention=Norm_wrapper(attention)
            self.attention.append(attention)

            transform_out = FFW_block(**block_kwargs)
            transform_out = Residual_wrapper(transform_out)
            if use_norm:
                transform_out=Norm_wrapper(transform_out)
            self.transform_out.append(transform_out)

        self.pre_projection_history = None

        self.projection = projection


    def call(self, history, attention_value, forecast_timesteps=12, teacher=None):

        self.attention_value = attention_value
        forecast = tf.keras.backend.in_train_phase(self.training_call(tf.concat([history, teacher], axis=1), forecast_timesteps),
                                        alt=self.validation_call(history, forecast_timesteps),
                                        training=tf.keras.backend.learning_phase())
        return forecast

    # information flow for a single processing step,
    def process_inputs(self, inputs):
        # transform the raw input signal
        after_transform_in = self.transform_in(inputs)
        if not self.previous_history_exists:
            self.after_transform_in_history = after_transform_in
        else:
            self.after_transform_in_history = tf.concat([self.after_transform_in_history, after_transform_in], axis=1)
        out = self.after_transform_in_history
        # create temporal context, residual, norm
        for block in range(len(self.self_attention)):
            out = self.self_attention[block](out)
            #relevant_self_attention_steps = after_self_attention[:,-inputs.shape[1]:,:]
            # create context with encoder thing, residual, norm
            out = self.attention[block](out, value=self.attention_value)
            out = self.transform_out[block](out)

        if not self.previous_history_exists:
            self.pre_projection_history = out
        else:
            self.pre_projection_history = tf.concat([self.pre_projection_history, out[:,-inputs.shape[1]:,:]], axis=1)

    def training_call(self, history_and_teacher, forecast_timesteps):
        # if self.use_dropout:
        #    history_and_teacher = drop_timesteps_of_signal(history_and_teacher, self.dropout_rate)
        #    history_and_teacher = drop_features_of_signal(history_and_teacher, self.dropout_rate)

        self.previous_history_exists = False
        self.process_inputs(history_and_teacher)
        self.previous_history_exists = True
         #technically not needed, since we dont have any followup calls anymore
        return self.projection(self.pre_projection_history[:,-forecast_timesteps:,:])

    def validation_call(self, history, forecast_timesteps):
        self.previous_history_exists = False
        self.process_inputs(history)
        self.previous_history_exists = True

        for step in range(forecast_timesteps - 1):
            next_input = self.projection(self.pre_projection_history[:, -1:, :])
            self.process_inputs(next_input)

        return self.projection(self.pre_projection_history[:,-forecast_timesteps:,:])

########################################################################################################################

class Dense_generator(tf.keras.layers.Layer):

    def __init__(self,
                 units=[[96,96]],
                 use_dropout=True,
                 dropout_rate=0.2,
                 use_norm=True,
                 projection=tf.keras.layers.Dense(20)
                 ):
        super(Dense_generator, self).__init__()
        block_kwargs = {'units': units,
                        'growth_rate': 12,
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_norm': use_norm,
                        'kernel_sizes': [3],
                        'residual': True,
                        }

        attention_kwargs = {'units_per_head': [int(units[-1][-1]/3)] * 2,
                            'project_to_units': units[-1][-1],
                            'use_norm': use_norm,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate}

        block_kwargs['project_to_features']=units[-1][-1]
        self.transform_in = DenseTCN(**block_kwargs)
        if use_norm:
            self.transform_in = Norm_wrapper(self.transform_in)
        self.after_transform_in_history = None

        self.self_attention=multihead_attentive_layer(**attention_kwargs)
        self.self_attention=Residual_wrapper(self.self_attention)
        if use_norm:
            self.self_attention=Norm_wrapper(self.self_attention)

        self.attention = multihead_attentive_layer(**attention_kwargs)
        self.attention = Residual_wrapper(self.attention)
        if use_norm:
            self.attention=Norm_wrapper(self.attention)

        block_kwargs['project_to_features']=None
        self.transform_out = DenseTCN(**block_kwargs)
        self.transform_out = Residual_wrapper(self.transform_out)
        if use_norm:
            self.transform_out=Norm_wrapper(self.transform_out)
        self.pre_projection_history = None

        self.projection = projection


    def call(self, history, attention_value, forecast_timesteps=12, teacher=None):
        self.attention_value = attention_value
        self.previous_history_exists = False

        forecast = tf.keras.backend.in_train_phase(self.training_call(tf.concat([history, teacher], axis=1), forecast_timesteps),
                                        alt=self.validation_call(history, forecast_timesteps),
                                        training=tf.keras.backend.learning_phase())

        return forecast

    # helper function to keep history if signals that we need to attend to
    def keep_history(self, addendum, history):
        if not self.previous_history_exists:
            return addendum
        else:
            return tf.concat([history, addendum], axis=1)

    # information flow for a single processing step,
    def process_inputs(self, inputs):
        # transform the raw input signal
        after_transform_in = self.transform_in(inputs)
        if not self.previous_history_exists:
            self.after_transform_in_history = after_transform_in
        else:
            self.after_transform_in_history = tf.concat([self.after_transform_in_history, after_transform_in], axis=1)

        # create temporal context, residual, norm
        after_self_attention = self.self_attention(self.after_transform_in_history)
        relevant_self_attention_steps = after_self_attention[:,-inputs.shape[1]:,:]

        # create context with encoder thing, residual, norm
        after_attention = self.attention(relevant_self_attention_steps, value=self.attention_value)

        after_transform_out = self.transform_out(after_attention)
        if not self.previous_history_exists:
            self.pre_projection_history = after_transform_out
        else:
            self.pre_projection_history = tf.concat([self.pre_projection_history, after_transform_out], axis=1)

    def training_call(self, history_and_teacher, forecast_timesteps):
        self.process_inputs(history_and_teacher)
        self.previous_history_exists = True
         #technically not needed, since we dont have any followup calls anymore
        return self.projection(self.pre_projection_history[:,-forecast_timesteps:,:])

    def validation_call(self, history, forecast_timesteps):
        self.process_inputs(history)
        self.previous_history_exists = True

        for step in range(forecast_timesteps - 1):
            next_input = self.projection(self.pre_projection_history[:, -1:, :])
            self.process_inputs(next_input)

        return self.projection(self.pre_projection_history[:,-forecast_timesteps:,:])

########################################################################################################################

class attentive_TCN(tf.keras.layers.Layer):
    '''
    Encoder consisting of alternating self-attention and Desne TCN blocks
    requires units in shape of [[attention block units], [dense tcn units] ... ]
    the first block is always self-attention in order to get the most information from the input
    outputs after each block (dense tcn and self-attention) are concatenated
    '''
    def __init__(self, units = None, 
                use_dropout=False, 
                dropout_rate=0.15, 
                use_norm=False, 
                mode='encoder', 
                use_attention=True, 
                self_recurrent=False):
        # self_recurrent is a quick fix, change to exclude later
        super(attentive_TCN, self).__init__()

        #TODO: change units so it's flexible but not too much
        if units is None:
            units = [[12,12,12,12,12],[20,20],[12,12,12]]
        self.units = units
        self.mode = mode
        self.use_attention = use_attention
        self.use_norm = use_norm
        self.block_stack = []
        for block in range(len(self.units)):
            # first block is always attention
            if block%2 == 1:
                # can accommodate several self-attention layers
                attn_block = []
                for layer in range(1):
                    attn_block.append(multihead_attentive_layer())
                self.block_stack.append(attn_block)
            else:
                self.block_stack.append(DenseTCN(units=[self.units[block]], 
                                            growth_rate=self.units[block][0],
                                            squeeze_factor=0.5, 
                                            use_dropout = use_dropout,
                                            dropout_rate=dropout_rate,
                                            use_norm=use_norm,
                                            kernel_sizes=[3],
                                            residual=False))
        if self.mode == 'decoder' and self.use_attention:
            self.attention_layer = Attention(self.units[-1][-1], mode='Transformer')


    def call(self, inputs, decoder_init_state=None, attention_query=None, attention_value=None):
        if self.mode=='decoder' and self.use_attention:
            context_vector = self.attention_layer(attention_query, value=attention_value)
            inputs = tf.concat([inputs,context_vector], axis=-1)
        for block in range(len(self.units)):
            # for the first and later attention layers, concatenate the outputs
            if block%2 == 1:
                for layer in range(len(self.block_stack[block])):
                    out = self.block_stack[block][layer](inputs)
                    # out = tf.concat([out,inputs],-1)
                    inputs = out
            else:
                # dense tcn block concatenates the outputs automatically
                out = self.block_stack[block](inputs)
                inputs = out
        return out

class TCN_Transformer(tf.keras.layers.Layer):
    def __init__(self,
                 units=[[20, 20], [10, 10, 10],    # first si the attention layer units, then the TCN units
                                  [30, 30], [10, 10, 10],
                                  [45, 45], [10, 10, 10]],
                 mode='Encoder',
                 use_dropout=True, dropout_rate=0.2, use_norm=True, **kwargs):
        super(TCN_Transformer, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.mode = mode

        self.all_layers = []
        if self.mode == 'decoder' or self.mode =='Decoder' or self.mode=='D':
            downsample_rate = 1
        else:
            downsample_rate = 2

        for stage_in_block in range(len(units)):
            if stage_in_block%2 == 0: # attention stuffs
                block_layers = []
                # self attention
                num_heads = len(units[stage_in_block])
                num_units = units[stage_in_block][0]
                block_layers.append(multihead_attentive_layer(num_heads=num_heads, num_units_per_head=num_units) )
                if self.use_norm:
                    block_layers.append(tf.keras.layers.BatchNormalization(axis=-1, center=True, scale=True))
                else:
                    block_layers.append([])
                if self.use_dropout:
                    block_layers.append( tf.keras.layers.Dropout(dropout_rate, noise_shape=(1, num_heads*num_units) ) )
                else:
                    block_layers.append([])

                # Wohoo, do the whole thing again, yay
                if self.mode == 'decoder' or self.mode =='Decoder' or self.mode=='D':

                    block_layers.append(multihead_attentive_layer(num_heads=num_heads, num_units_per_head=num_units))
                    if self.use_norm:
                        block_layers.append(tf.keras.layers.BatchNormalization(axis=-1, center=True, scale=True))
                    else:
                        block_layers.append([])
                    if self.use_dropout:
                        block_layers.append(
                            tf.keras.layers.Dropout(dropout_rate, noise_shape=(1, num_heads * num_units)))
                    else:
                        block_layers.append([])

            elif stage_in_block%2 == 1 and units[stage_in_block] != []:
                block_layers.append( DenseTCN(num_blocks=1,
                                            num_layers_per_block=len(units[stage_in_block]),
                                            growth_rate=units[stage_in_block][0],
                                            squeeze_factor=0.7,
                                            use_dropout = use_dropout,
                                            dropout_rate=dropout_rate,
                                              downsample_rate=downsample_rate,
                                            use_norm=use_norm,
                                            kernel_sizes=[3]))
                self.all_layers.append(block_layers)
            elif units[stage_in_block] == []:
                block_layers.append([])
                self.all_layers.append(block_layers)

    def call(self, input, attention_query=None, attention_value=None):
        out = input
        for num_block in range(len(self.all_layers)):
            # block_layer in encoder case: [attention, norm, dropout, tcn]
            # block_layer in decoder case [attention, norm, dropout, attention, norm, dropout, tcn]

            # self attention stage
            out = self.all_layers[num_block][0](out, value=out)
            if self.use_norm:
                out = self.all_layers[num_block][1](out)
            if self.use_dropout:
                out = self.all_layers[num_block][2](out, training=tf.keras.backend.learning_phase())

            # context attention stage
            if self.mode == 'decoder' or self.mode =='Decoder' or self.mode=='D':
                out = self.all_layers[num_block][3](out, value=attention_value)

                if self.use_norm:
                    out = self.all_layers[num_block][4](out)
                if self.use_dropout:
                    out = self.all_layers[num_block][5](out, training=tf.keras.backend.learning_phase())

            if self.all_layers[num_block][-1] != []:
                out = self.all_layers[num_block][-1](out)

        return out

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