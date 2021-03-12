# contains all the building blocks necessary to build the architectures we will be working on
# In a sense, this will be the main modiefied file


import tensorflow as tf
import numpy as np
import random
########################################################################################################################
'''General help functions, lowest level'''
initializer = tf.keras.initializers.glorot_uniform
activation = tf.nn.relu

def test_attention_equivalence():
    query = tf.random.uniform(shape=[1, 3, 5])
    value = tf.random.uniform(shape=[1, 3, 5])

    sa_test = attention_equivalence_test(query, value, self_attention=True)
    sa_test = tf.reduce_sum(tf.abs(sa_test))
    if sa_test.numpy() > 0.0:
        print('Self attention not equivalent')
    else:
        print('self attention equivalent')


    a_test = attention_equivalence_test(query, value, self_attention=False)
    a_test = tf.reduce_sum(tf.abs(a_test))
    if a_test.numpy() > 0.0:
        print('attention not equivalent')
    else:
        print('attention equivalent')

def attention_equivalence_test(query, value, self_attention=True):
    # If value is none, we assume self attention and set value = query
    # if key is none we assume standard usage and set key =
    def build_mask(input_tensor):
        mask = tf.linalg.band_part(tf.ones_like(input_tensor)*-1e12, 0, -1) #upper triangular ones, INCLUDING diagonal
        return mask

    def our_attention(query, value, self_attention = True):
        # Changed the beginning slightly, so we guarantee the same inputs to our attention as the tensorflow attention
        key = value

        score_pre_softmax = tf.matmul(query, key, transpose_b=True)

        score_pre_softmax = score_pre_softmax / tf.math.sqrt(tf.cast(tf.shape(value)[1], tf.float32))

        if self_attention:
            score_pre_softmax += build_mask(score_pre_softmax)

        score = tf.nn.softmax(score_pre_softmax, axis=1)

        context_vector = tf.matmul(score, value)

        return context_vector

    if self_attention:
        tf_attention = tf.keras.layers.Attention(use_scale=True, causal=True)
    else:
        tf_attention = tf.keras.layers.Attention(use_scale=True, causal=False)

    our_attention_context = our_attention(query, value, self_attention)
    tf_attention_context = tf_attention([query, value])
    return our_attention_context - tf_attention_context

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
                alt=input,)

def drop_timesteps_of_signal(input, dropout_rate):
    batch_dim = tf.shape(input)[0]
    # if batch_dim is None:
    #     batch_dim = 128
    return tf.keras.backend.in_train_phase(
                tf.nn.dropout(input, dropout_rate, noise_shape=(batch_dim, input.shape[1], 1)),
                alt=input,)

class Cropping_layer(tf.keras.layers.Layer):
    def __init__(self, noise_rate=0.2):
        super(Cropping_layer, self).__init__()
        self.noise_rate = noise_rate

    def call(self, signal):
        return tf.keras.backend.in_train_phase(self.randomize_input_lengths(signal),
                                                    alt=signal,)

    def randomize_input_lengths(self, signal):

        max_val=self.noise_rate*tf.cast(tf.shape(signal)[1], dtype=tf.float32)
        slice_beginning = tf.random.uniform(shape=(), minval=0 , maxval=max_val)
        slice_beginning = tf.cast(slice_beginning, dtype=tf.int32)

        signal = tf.slice(signal, begin=[0, 0, 0], size=[tf.shape(signal)[0],tf.shape(signal)[1] -slice_beginning,tf.shape(signal)[2]])
        return signal

from tensor2tensor.layers.common_attention import add_positional_embedding_nd, add_timing_signal_1d, get_timing_signal_1d

class Positional_embedding_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, max_length, name=None):
        super(Positional_embedding_wrapper, self).__init__(layer)
        self.max_length = max_length
        self.embedding_name = name

    def call(self, input):
        signal = self.layer(input)
        return  add_timing_signal_1d(signal, max_timescale=self.max_length)


    # def call(self, input):
    #     layer_out = self.layer(input)
    #     layer_out = add_positional_embedding_nd(layer_out, max_length=self.max_length, name=self.embedding_name)
    #
    #     return layer_out
########################################################################################################################
'''Wrappers for feedforward architectures'''
# important to note that all wrappers have to be compatible with both, attention(query, value) and self_attention(query)

class Dropout_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, dropout_rate, units=None):
        super(Dropout_wrapper, self).__init__(layer)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
    def call(self, inputs, value=None):
        if value is None:
            return self.dropout(self.layer(inputs))
        else:
            return self.dropout(self.layer(inputs, value))

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

class Dense_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer):
        super(Dense_wrapper, self).__init__(layer)

    def call(self, inputs, value=None):
        if value is None:
            return tf.concat([inputs, self.layer(inputs)], axis=-1)
        else:
            return tf.concat([inputs, self.layer(inputs, value=value)], axis=-1)
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

    def call(self, input, initial_state=None):
        output, state_h, state_c = self.layer(input, initial_state=initial_state)
        output = output+input
        return output, state_h, state_c

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

    def call(self, input, initial_state=None):
        layer_out, state_h, state_c = self.layer(input, initial_state=initial_state)

        return self.norm(layer_out) , state_h, state_c

class GRU_Norm_wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, norm_type='Layer norm'):
        super(GRU_Norm_wrapper, self).__init__(layer)
        if norm_type == 'Layer norm':
            self.norm = tf.keras.layers.LayerNormalization(axis=-1,
                                                           center=True,
                                                           scale=True)
        else:
            print('wrong norm type supplied to NormWrapper, supplied', norm_type)

    def call(self, input, initial_state=None):
        layer_out, state_f = self.layer(input, initial_state=initial_state)

        return self.norm(layer_out) , state_f

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

        self.multilayer_lstm = []  # Multiple layers work easiest as a list of layers, so here we start

        for block in range(len(self.units)):
            lstm_block = []
            for layer in range(len(self.units[block])):
                # get one LSTM layer per layer we speciefied, with the units we need
                # Do we need to initialize the layer in the loop?
                one_lstm = tf.keras.layers.LSTM(units=self.units[block][layer],
                                                      activation='tanh',
                                                      recurrent_activation='sigmoid',
                                                      recurrent_dropout=dropout_rate,
                                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                      dropout=dropout_rate,
                                                      unroll=False,
                                                      use_bias=True,
                                                      return_sequences=True,
                                                      implementation=1,
                                                      return_state=True,
                                                      kernel_initializer=initializer,
                                                      recurrent_initializer='orthogonal',
                                                      bias_initializer='zeros')

                # do we wanna highway
                # if self.use_hw:
                #     one_lstm = LSTM_Highway_wrapper(one_lstm)
                if self.use_residuals:
                    one_lstm = LSTM_Residual_wrapper(one_lstm)
                # do we wanna norm
                if self.use_norm:
                    one_lstm = LSTM_Norm_wrapper(one_lstm)

                lstm_block.append(one_lstm) # will be len(layers_in_block)
            self.multilayer_lstm.append(lstm_block)

    def call(self, signal, initial_states=None):
        out_states = []
        # Initialize as zero states if No state is given
        if initial_states is None:
            for lstm_block in self.multilayer_lstm:
                out_states_block = []
                for lstm in lstm_block:
                    signal, state_h, state_c = lstm(signal, initial_state=None)
                    out_states_block.append([state_h, state_c])
                out_states.append(out_states_block)

        else:
            for lstm_block, states_block in zip(self.multilayer_lstm, initial_states):
                out_states_block = []
                for lstm, states in zip(lstm_block, states_block):
                    signal, state_h, state_c = lstm(signal, initial_state=states)
                    out_states_block.append([state_h, state_c])

                out_states.append(out_states_block)

        return signal, out_states

class MultiLayer_GRU(tf.keras.layers.Layer):
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
        super(MultiLayer_GRU, self).__init__()

        self.units = units
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.use_hw = use_hw
        self.use_residuals = use_residuals
        self.return_all_states = return_all_states

        self.multilayer_lstm = []  # Multiple layers work easiest as a list of layers, so here we start

        for block in range(len(self.units)):
            lstm_block = []
            for layer in range(len(self.units[block])):
                # get one LSTM layer per layer we speciefied, with the units we need
                # Do we need to initialize the layer in the loop?
                one_lstm = tf.keras.layers.GRU(units=self.units[block][layer],
                                                      activation='tanh',
                                                      recurrent_activation='sigmoid',
                                                      recurrent_dropout=dropout_rate,
                                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                      dropout=dropout_rate,
                                                      unroll=False,
                                                      use_bias=True,
                                                      return_sequences=True,
                                                      implementation=1,
                                                      return_state=True,
                                                      kernel_initializer=initializer,
                                                      recurrent_initializer='orthogonal',
                                                      bias_initializer='zeros')

                # do we wanna highway
                # if self.use_hw:
                #     one_lstm = LSTM_Highway_wrapper(one_lstm)
                if self.use_residuals:
                    one_lstm = LSTM_Residual_wrapper(one_lstm)
                # do we wanna norm
                if self.use_norm:
                    one_lstm = GRU_Norm_wrapper(one_lstm)

                lstm_block.append(one_lstm) # will be len(layers_in_block)
            self.multilayer_lstm.append(lstm_block)

    def call(self, signal, initial_states=None):
        out_states = []
        # Initialize as zero states if No state is given
        if initial_states is None:
            for lstm_block in self.multilayer_lstm:
                out_states_block = []
                for lstm in lstm_block:
                    signal, state_f = lstm(signal, initial_state=None)
                    out_states_block.append([state_f])
                out_states.append(out_states_block)

        else:
            for lstm_block, states_block in zip(self.multilayer_lstm, initial_states):
                out_states_block = []
                for lstm, states in zip(lstm_block, states_block):
                    signal, state_f = lstm(signal, initial_state=states)
                    out_states_block.append([state_f])

                out_states.append(out_states_block)

        return signal, out_states

########################################################################################################################

class new_bad_Attention(tf.keras.layers.Layer):
    '''
    Implements an attention layer.
    Implemented are Bahdanau and Transformer style attention.
    If query/key kernel and strides are supplied this mimics the proposed convolutional attention mechanism.
    '''
    def __init__(self, units,
                 mode='Transformer',
                 use_dropout=True,
                 dropout_rate=0.2,
                 L1=0.0, L2=0.0,
                 ):
        super(Attention, self).__init__()
        self.mode = mode
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.W_query = tf.keras.layers.Dense(units,
                              activation=None,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                              kernel_initializer=initializer,
                              use_bias=False)
        if use_dropout:
            self.W_query = Dropout_wrapper(self.W_query, dropout_rate=dropout_rate)

        self.W_value = tf.keras.layers.Dense(units,
                              activation=None,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                              kernel_initializer=initializer,
                              use_bias=False)
        if use_dropout:
            self.W_value = Dropout_wrapper(self.W_value, dropout_rate=dropout_rate)

        self.W_key = tf.keras.layers.Dense(units,
                          activation=None,
                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                          kernel_initializer=initializer,
                          use_bias=False)
        if use_dropout:
            self.W_key = Dropout_wrapper(self.W_key, dropout_rate=dropout_rate)


    def dropout_mask(self, score):
        dropout_mask = tf.random.uniform(shape=tf.shape(score), minval=0.0, maxval=1.0)
        dropout_mask = tf.where(dropout_mask <= self.dropout_rate/2, x=-1e12, y=0.0)
        return dropout_mask

    def add_causal_mask(self, input_tensor):
        causal_mask = -1e12* (tf.ones_like(input_tensor) - tf.linalg.band_part(tf.ones_like(input_tensor), -1, 0)) #strictly upper triagonal -inf

        return input_tensor + causal_mask


    def call(self, query, value=None, key=None):
        # If value is none, we assume self attention and set value = query
        # if key is none we assume standard usage and set key = value
        if value is None:
            value = query
            self_attention = True
        else:
            self_attention = False

        if key is None:
            key= value

        query = self.W_query(query)
        key = self.W_key(key)
        value = self.W_value(value)

        score_pre_softmax = tf.matmul(query, key, transpose_b=True)

        score_pre_softmax = score_pre_softmax / tf.math.sqrt(tf.cast(tf.shape(value)[1], tf.float32))

        if self_attention:
            score_pre_softmax = self.add_causal_mask(score_pre_softmax)

        score = tf.nn.softmax(score_pre_softmax, axis=-1)

        context_vector = tf.matmul(score, value)

        return context_vector

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

    # def build_mask(self, input_tensor):
    #     mask = tf.linalg.band_part(tf.ones_like(input_tensor)*-1e12, 0, -1) #upper triangular ones, INCLUDING diagonal
    #     return mask
    def add_causal_mask(self, input_tensor):
        causal_mask = -1e12* (tf.ones_like(input_tensor) - tf.linalg.band_part(tf.ones_like(input_tensor), -1, 0)) #strictly upper triagonal -inf

        return input_tensor + causal_mask

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
                score_pre_softmax = self.add_causal_mask(score_pre_softmax)

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
    def __init__(self, num_heads=3,
                 output_units=20,
                 L1=0.0, L2=0.0,
                 use_dropout=True, dropout_rate=0.2,
                 max_length_sequence_query=None,
                 max_length_sequence_value=None,
                 layer_name=None):
        super(multihead_attentive_layer, self).__init__()
        # units is a list of lists, containing the numbers of units in the attention head
        self.num_heads = num_heads
        # TODO: DANIEL CHECK THE ATTENTION FEATURE SIZE 
        # we rewrite units_per_head without getting the attention features size
        self.units_per_head = int(output_units*2/self.num_heads)
        self.attention_features = self.units_per_head*self.num_heads
        self.layer_name = layer_name

        if max_length_sequence_query is not None:
            self.positional_embed = True
            self.max_length_sequence_query = max_length_sequence_query
            if max_length_sequence_value is None:
                self.max_length_sequence_value = max_length_sequence_query
            else:
                self.max_length_sequence_value = max_length_sequence_value
        else:
            self.positional_embed = False

        self.W_query = tf.keras.layers.Dense(self.attention_features,
                                              activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                              kernel_initializer=initializer,
                                              use_bias=True)
        if use_dropout:
            self.W_query = Dropout_wrapper(self.W_query, dropout_rate=dropout_rate)

        self.W_value = tf.keras.layers.Dense(self.attention_features,
                                              activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                              kernel_initializer=initializer,
                                              use_bias=True)
        if use_dropout:
            self.W_value = Dropout_wrapper(self.W_value, dropout_rate=dropout_rate)

        self.W_key = tf.keras.layers.Dense(self.attention_features,
                                          activation=None,
                                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                          kernel_initializer=initializer,
                                          use_bias=True)
        if use_dropout:
            self.W_key = Dropout_wrapper(self.W_key, dropout_rate=dropout_rate)

        self.output_projection = tf.keras.layers.Dense(output_units,
                                                      activation=None,
                                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                      kernel_initializer=initializer,
                                                      use_bias=True)
        if use_dropout:
            self.output_projection = Dropout_wrapper(self.output_projection, dropout_rate=dropout_rate)

    def get_causal_mask(self, query):
        query_shape = tf.shape(query)
        tensor_of_ones = tf.ones(shape=[query_shape[0], query_shape[1], query_shape[1]])
        causal_mask = -1e12* (tensor_of_ones - tf.linalg.band_part(tensor_of_ones, -1, 0)) #strictly upper triagonal -inf

        return causal_mask

    def calculate_attention(self, query, value, key, self_attention_mask=None):

        score_pre_softmax = tf.matmul(query, key, transpose_b=True)
        score_pre_softmax = score_pre_softmax / tf.math.sqrt(tf.cast(tf.shape(value)[1], tf.float32))
        if self_attention_mask is not None:
            score_pre_softmax = score_pre_softmax + self_attention_mask

        score = tf.nn.softmax(score_pre_softmax, axis=-1)
        context_vector = tf.matmul(score, value)

        return context_vector

    def slit_into_heads(self, signal):
        signal_shape = tf.shape(signal)
        return tf.reshape(signal, shape=[signal_shape[0], self.num_heads, signal_shape[1], self.units_per_head])
    def merge_heads(self, signal):
        signal_shape = tf.shape(signal)
        return tf.reshape(signal, shape=[signal_shape[0], signal_shape[2], self.attention_features])

    def call(self, query, value=None):

        if value is None:
            value = query
            self_attention_mask = self.get_causal_mask(query)
        else:
            self_attention_mask = None

        # if self.positional_embed:
        #     # query = add_timing_signal_1d(query, self.max_length_sequence_query)
        #     # value = add_timing_signal_1d(value, self.max_length_sequence_value)
        #     query = add_positional_embedding_nd(query, self.max_length_sequence_query, self.name+'_query')
        #     value = add_positional_embedding_nd(value, self.max_length_sequence_value, self.name+'_value')

        full_query = self.W_query(query)
        full_key = self.W_key(value)
        full_value = self.W_value(value)

        multihead_attention = []
        for head in range(self.num_heads):
            start_feature = head*self.units_per_head
            end_feature = start_feature+self.units_per_head
            head_attention = self.calculate_attention(full_query[:,:,start_feature:end_feature],
                                                      full_value[:,:,start_feature:end_feature],
                                                      full_key[:,:,start_feature:end_feature],
                                                      self_attention_mask)
            multihead_attention.append(head_attention)

        multihead_attention = tf.concat(multihead_attention, axis=-1)

        multihead_out = self.output_projection(multihead_attention)

        return multihead_out

class transformation_layer(tf.keras.layers.Layer):
    def __init__(self, output_units=128,
                 L1=0.0, L2=0.0,
                 use_dropout=True, dropout_rate=0.2):
        super(transformation_layer, self).__init__()

        self.transform = tf.keras.layers.Dense(int(4*output_units),
                                                          activation=activation,
                                                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                          kernel_initializer=initializer,)
        self.project = tf.keras.layers.Dense(output_units,
                                                          activation=None,
                                                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                          kernel_initializer=initializer,)

    def call(self, signal):
        signal = self.transform(signal)
        signal = self.project(signal)
        return signal
########################################################################################################################

class preactivated_CNN(tf.keras.layers.Layer):
    def __init__(self, num_features, kernel_size, dilation_rate, stride=1, L1=0.0, L2=0.0):
        super(preactivated_CNN, self).__init__()
        self.layer =  tf.keras.layers.Conv1D(filters=num_features,
                                            kernel_size=kernel_size,
                                            activation=activation,
                                            dilation_rate=1,
                                            kernel_initializer=initializer,
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                            padding='causal')
    def call(self, signal):
        return self.layer(signal)

class wavenet_CNN(tf.keras.layers.Layer):
    def __init__(self,
                 num_channels=5*7,
                 max_length_sequence=288,
                 length_receptive_window=288,
                 dropout_rate=0.0,
                 L1=0.0, L2=0.0,
                 use_residual=True,
                 use_norm=False,
                 use_dense=False,
                 use_dropout=False,
                 pass_input=False,
                 add_positional_embedding=False,
                 embedding_name=None,
                 ):
        super(wavenet_CNN, self).__init__()

        top = np.log(length_receptive_window)
        base = np.log(2)
        num_layers = top / base
        num_layers = np.ceil(num_layers)

        self.use_residual = use_residual
        self.pass_input = pass_input

        self.wavenet = []
        dilation_rates = []
        for layer in range(int(num_layers)):
            dilation_rates.append(2**layer)

        self.num_channels = 0
        for num_layer in range(int(num_layers)):
            if use_dense:
                filters = int(4*num_channels) if num_layer == 0 else int(num_channels)
                self.num_channels = self.num_channels +  filters
            else:
                filters = num_channels
                self.num_channels = num_channels
            cnn = tf.keras.layers.Conv1D(filters=filters,
                                            kernel_size=2,
                                            activation=activation,
                                            dilation_rate=int(2**num_layer),
                                            kernel_initializer=initializer,
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                            padding='causal')

            if self.use_residual and num_layer != 0:
                cnn = Residual_wrapper(cnn)
            if use_dense:
                cnn = Dense_wrapper(cnn)
            if  num_layer == num_layers-1 and add_positional_embedding:
                cnn = Positional_embedding_wrapper(cnn, max_length=max_length_sequence, name=embedding_name)

            if use_norm:
                cnn = Norm_wrapper(cnn, norm='batch')

            self.wavenet.append(cnn)


    def get_num_channels(self):
        return int(self.num_channels)

    def call(self, input):
        signal = input

        for layer in self.wavenet:
            signal = layer(signal)

        return signal

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
    def __init__(self, units=20,
                 num_encoder_blocks=1,
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_norm=False,
                 use_hw=False, use_residual=False,
                 L1=0.0, L2=0.0,
                 return_state=True):
        super(block_LSTM, self).__init__()
        self.return_state = return_state
        self.encoder_blocks = []
        self.cropping_layer = Cropping_layer(0.2)
        for block in range(num_encoder_blocks):
            self.encoder_blocks.append(MultiLayer_LSTM(units=[[units]],
                                                use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                use_norm=use_norm,
                                                use_hw=use_hw, use_residuals=use_residual,
                                                L1=L1, L2=L2))
        self.transform_layer = tf.keras.layers.Dense(units,
                                                    activation=activation,
                                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                    kernel_initializer=initializer,
                                                    use_bias=True)
        if use_norm:
            self.transform_layer = Norm_wrapper(self.transform_layer)


    def call(self, signal, initial_states=None):
        # signal = self.cropping_layer(signal)
        encoder_outs = []
        encoder_last_states = []
        signal = self.transform_layer(signal)

        for block in range(len(self.encoder_blocks)):
            signal, block_states = self.encoder_blocks[block](signal, initial_states=initial_states)
            encoder_outs.append(signal)
            encoder_last_states.append(block_states[0])
            
        if self.return_state:
            return encoder_outs, encoder_last_states
        else:
            return tf.concat(encoder_outs, axis=-1)

class decoder_LSTM_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=20,
                 num_decoder_blocks=1,
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_hw=False, use_residual=False,
                 use_norm=False,
                 L1=0.0, L2=0.0,
                 attention_squeeze=0.5,
                 use_attention=False, attention_heads=5,
                 projection_layer=None):
        super(decoder_LSTM_block, self).__init__()
        self.num_decoder_blocks = num_decoder_blocks
        self.use_attention = use_attention
        self.projection_layer=projection_layer

        self.cropping_layer = Cropping_layer(0.8)

        self.transform_layer = tf.keras.layers.Dense(units,
                                                    activation=activation,
                                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                    kernel_initializer=initializer,
                                                    use_bias=True)
        if use_norm:
            self.transform_layer = Norm_wrapper(self.transform_layer)

        if self.use_attention:
            self.attention_blocks = []
        self.decoder_blocks = []

        for layer in range(num_decoder_blocks):
            self.decoder_blocks.append(MultiLayer_LSTM(units=[[units]],
                                                use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                use_norm=use_norm,
                                                use_hw=use_hw, use_residuals=use_residual,
                                                L1=L1, L2=L2))
            if self.use_attention:
                if attention_heads == 1:
                    #equivalent to the dimensionality of attention heads
                    attention = Attention(units,
                                        mode='Transformer',
                                        use_dropout=use_dropout,
                                        dropout_rate=dropout_rate,
                                        L2=L2, L1=L1,)

                else:
                    units_per_head =[int(units*attention_squeeze)]*attention_heads
                    attention = multihead_attentive_layer(units_per_head=units_per_head,
                                                          output_units=units,
                                                          dropout_rate=dropout_rate,
                                                          use_dropout=use_dropout,
                                                          L1=L1,
                                                          L2=L2, )
                    attention = Residual_wrapper(attention)
                if use_norm:
                    attention = Norm_wrapper(attention)
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
                                                                                timesteps),)

        return forecast

    def teacher_call(self, prev_history, teacher, decoder_init_state, attention_value, timesteps):
        if prev_history.shape[-2] > 1:
            prev_history = self.cropping_layer(prev_history)
        full_history = tf.concat([prev_history, teacher], axis=1)
        full_history = self.transform_layer(full_history) # to get feature size for residuals
        unprojected_forecast = self.decode_layerwise(full_history, decoder_init_state, attention_value)
        forecast = self.projection_layer(unprojected_forecast[:,-timesteps:,:])
        return forecast

    def decode_layerwise(self, input, decoder_state, attention_values):
        signal=input
        for num_block in range(len(self.decoder_blocks)):
            block_state = [decoder_state[num_block]] if decoder_state else None

            signal, block_state = self.decoder_blocks[num_block](signal, initial_states=block_state)

            if self.use_attention:
                signal = self.attention_blocks[num_block](signal, value=attention_values[-1])
                # signal = tf.concat([signal, attention_context], axis=-1)

        return signal

    def self_recurrent_call(self, prev_history, decoder_init_state, attention_values, timesteps):
        decoder_state = decoder_init_state  # [[[state_h, state_c]]] #(blocks, layers_in_block, 2)
        for timestep in range(timesteps + prev_history.shape[1] - 1):
            if timestep < prev_history.shape[1]:
                input = tf.expand_dims(prev_history[:,timestep,:], axis=1)
                decoder_out, decoder_state = self.decode_stepwise(input, decoder_state, attention_values)
                if timestep == prev_history.shape[1] - 1:
                    forecast = self.projection_layer(decoder_out) #self.projection_layer(decoder_out)
            else:
                input = tf.expand_dims(forecast[:, -1, :], axis=1)
                # input = forecast[:,-1:,:]
                decoder_out, decoder_state = self.decode_stepwise(input, decoder_state, attention_values)
                forecast = tf.concat([forecast, self.projection_layer(decoder_out)], axis=1) #self.projection_layer(decoder_out)], axis=1)

        # forecast = self.projection_layer(forecast)
        return forecast

    def decode_stepwise(self, input, last_step_states, attention_values):
        # attention layers and assign those to the decoder states later.
        signal = input
        signal = self.transform_layer(signal)
        this_step_states = []
        for num_block in range(len(self.decoder_blocks)):
            block_state = [last_step_states[num_block]] if last_step_states else None
            signal, block_state = self.decoder_blocks[num_block](signal, 
                                                                initial_states=block_state,
                                                                )
            this_step_states.append(block_state[0])

            if self.use_attention:
                signal = self.attention_blocks[num_block](signal, value=attention_values[-1])
                    # signal = tf.concat([signal, attention_context], axis=-1)

        last_step_states = this_step_states
        return signal, last_step_states

class classic_attn_decoder_LSTM_block():
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

class CNN_encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_initial_features,
                 length_receptive_window,
                 max_length_sequence,
                 attention_heads=5,
                 attention_squeeze=0.5,
                 use_residual=True,
                 use_dense=False,
                 use_norm=True,
                 L2=0.0, L1=0.0,
                positional_embedding=True,
                 force_relevant_context=False,
                 use_self_attention=False,
                 transformer_blocks=1,
                 ):
        super(CNN_encoder, self).__init__()

        # self.crop = Cropping_layer(0.4)
        self.force_relevant_context = force_relevant_context

        self.pseudo_embedding = wavenet_CNN(num_channels=num_initial_features,
                                           max_length_sequence=max_length_sequence,
                                           length_receptive_window=length_receptive_window,
                                           embedding_name='Encoder',
                                           use_residual=use_residual,
                                           use_norm=use_norm,
                                           add_positional_embedding=positional_embedding,
                                           pass_input=False,
                                           use_dense=use_dense,
                                           L2=L2, L1=L1,
                                           )
        self.use_self_attention = use_self_attention
        if use_self_attention:
            attention_features = self.pseudo_embedding.get_num_channels() * attention_squeeze
            self_attention = multihead_attentive_layer(units_per_head=[int(attention_features)] * attention_heads,
                                                  causal=False,
                                                  use_dropout=False,
                                                  L2=L2, L1=L1,
                                                  output_units=int(self.pseudo_embedding.get_num_channels()))
            self_attention = Residual_wrapper(self_attention)
            self.self_attention = self_attention

            transform = tf.keras.layers.Conv1D(filters=num_initial_features,
                                               kernel_size=1,
                                               activation=activation,
                                               dilation_rate=1,
                                               kernel_initializer=initializer,
                                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                               padding='causal')
            if use_residual:
                transform = Residual_wrapper(transform)
            if use_norm:
                transform = Norm_wrapper(transform, norm='layer')
            self.transform_out = transform

    def call(self, input):
        signal = input

        signal = self.pseudo_embedding(signal)

        if self.use_self_attention:
            signal = self.self_attention(signal)
            signal = self.transform_out(signal)

        if self.force_relevant_context:
            signal = signal[:,-24*4:,:]

        return signal

class CNN_decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_initial_features=4*7,
                 max_length_sequence=24,
                 length_receptive_window=500,
                 attention_heads=5,
                 attention_squeeze=0.5,
                 projection_layer=None,
                 use_residual=True,
                 use_dense=False,
                 use_norm=True,
                 L2=0.0, L1=0.0,
                 positional_embedding=True,
                 force_relevant_context=False,
                transformer_blocks=1,
                 use_self_attention=False,
                 ):
        super(CNN_decoder, self).__init__()
        self.projection_layer = projection_layer

        self.pseudo_embedding = wavenet_CNN(num_channels=num_initial_features,
                                           max_length_sequence=max_length_sequence,
                                           length_receptive_window=length_receptive_window,
                                            embedding_name='Decoder',
                                          use_residual=use_residual,
                                          use_norm=use_norm,
                                          add_positional_embedding=positional_embedding,
                                          use_dense=use_dense,
                                          L2=L2, L1=L1,)

        attention_features = self.pseudo_embedding.get_num_channels() * attention_squeeze
        width = int(self.pseudo_embedding.get_num_channels())
        self.use_self_attention = use_self_attention
        self.transformer = []

        for block in range(transformer_blocks):
            block = {}
            if use_self_attention:
                self_attention = multihead_attentive_layer(units_per_head=[int(attention_features)] * attention_heads,
                                                      causal=False,
                                                      use_dropout=False,
                                                      L2=L2, L1=L1,
                                                      output_units=width)
                self_attention = Residual_wrapper(self_attention)
                block['self_attention'] = self_attention

            attention = multihead_attentive_layer(units_per_head=[int(attention_features)] * attention_heads,
                                                  causal=False,
                                                  use_dropout=False,
                                                  L2=L2, L1=L1,
                                                  output_units=width)
            attention = Residual_wrapper(attention)
            block['attention'] = attention

            block['transform'] = tf.keras.layers.Conv1D(filters=width,
                                         kernel_size=1,
                                         activation=activation,
                                         dilation_rate=1,
                                         kernel_initializer=initializer,
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                         padding='causal')

            self.transformer.append(block)


    def call(self, history_input, teacher, attention_value, timesteps):
        return tf.keras.backend.in_train_phase(self.training_call(history_input, teacher, attention_value, timesteps),
                                               alt=self.inference_call(history_input, teacher, attention_value,
                                                                       timesteps),)

    def training_call(self, history_input, teacher, attention_value, timesteps):
        self.max_length = history_input.shape[1] + teacher.shape[1]
        signal = tf.concat([history_input, teacher], axis=1)

        signal = self.pseudo_embedding(signal) #Pseudo Embeddings
        # signal = self.crop(signal)
        # Put Transformer Decoder

        for block in self.transformer:
            if 'self_attention' in block:
                signal = block['self_attention'](signal)
            if 'attention' in block:
                signal = block['attention'](signal, attention_value)
            if 'transform' in block:
                signal = block['transform'](signal)

        signal = signal[:, -timesteps:, :]
        forecast = self.projection_layer(signal)
        return forecast

    def inference_call(self, history_input, teacher, attention_value, timesteps):
        offset = self.max_length - history_input.shape[1]

        for step in range(timesteps):
            # make sure the signal we're feeding to the CNN always has the same length, because TF sucks
            # projected_history = self.projection(history_input)
            padded_projected_history_input = tf.concat([history_input,
                                               tf.zeros(shape=[tf.shape(history_input)[0],
                                                               offset - step,
                                                               history_input.shape[2]]
                                                        )], axis=1)

            padded_post_cnn = self.pseudo_embedding(padded_projected_history_input)

            signal = padded_post_cnn[:, :-(offset - step), :] if offset - step > 0 else padded_post_cnn

            if step == 0:
                buffers = []

            for num_block in range(len(self.transformer)):
                block = self.transformer[num_block]

                if 'self_attention' in block:

                    if step == 0:
                        buffers.append(signal)
                        signal = block['self_attention'](signal)
                    else:
                        query = signal[:,-1:,:]
                        buffers[num_block] = tf.concat([buffers[num_block], query], axis=1)
                        signal = block['self_attention'](query, buffers[num_block])

                if 'attention' in block:
                    signal = block['attention'](signal, attention_value)
                if 'transform' in block:
                    signal = block['transform'](signal)

            forecast_step = self.projection_layer(signal[:,-1:,:])
            history_input = tf.concat([history_input, forecast_step], axis=1)


        return history_input[:,-timesteps:,:]
########################################################################################################################

class FFNN_encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_initial_features,
                 max_length_sequence_supplement,
                 attention_heads=5,
                 attention_squeeze=0.5,
                 use_residual=True,
                 use_dense=False,
                 use_norm=True,
                 L2=0.0, L1=0.0,
                 positional_embedding=True,
                 force_relevant_context=False,
                 use_self_attention=False,
                 transformer_blocks=1,
                 ):
        super(FFNN_encoder, self).__init__()

        self.crop = Cropping_layer(0.2)
        self.force_relevant_context = force_relevant_context
        self.pseudo_embedding = tf.keras.layers.Dense(num_initial_features,
                              activation=activation,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                              kernel_initializer=initializer,
                              use_bias=True)
        # self.pseudo_embedding = Dropout_wrapper(self.pseudo_embedding, dropout_rate=0.1)
        if positional_embedding:
             self.pseudo_embedding = Positional_embedding_wrapper(self.pseudo_embedding, max_length=max_length_sequence_supplement)
        if use_norm:
            self.pseudo_embedding = Norm_wrapper(self.pseudo_embedding, norm='layer')

        self.use_self_attention = use_self_attention
        if use_self_attention:
            # TODO: DANIEL CHECK THE ATTENTION FEATURE SIZE
            attention_features = num_initial_features * attention_squeeze
            self_attention = multihead_attentive_layer(num_heads= attention_heads,
                                                  use_dropout=False,
                                                  L2=L2, L1=L1,
                                                  output_units=num_initial_features,
                                                  max_length_sequence_query=max_length_sequence_supplement,
                                                       layer_name='encoder_sa'
                                                       )
            if use_residual:
                self_attention = Residual_wrapper(self_attention)
            if use_norm:
                self_attention = Norm_wrapper(self_attention, norm='layer')
            self.self_attention = self_attention

            transform = transformation_layer(num_initial_features, L1=L1, L2=L2, use_dropout=False)
            if use_residual:
                transform = Residual_wrapper(transform)
            if use_norm:
                transform = Norm_wrapper(transform, norm='layer')
            # if positional_embedding:
            #     transform = Positional_embedding_wrapper(transform, max_length_sequence)
            self.transform_out = transform


    def call(self, input):
        signal = input

        signal = self.pseudo_embedding(signal)

        if self.use_self_attention:
            signal = self.self_attention(signal)
            signal = self.transform_out(signal)
        if self.force_relevant_context:
            signal = signal[:,-24*4:,:]


        return signal

class FFNN_decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_initial_features=4*7,
                 max_length_sequence_history=24,
                 max_length_sequence_supplement=24*4,
                 attention_heads=5,
                 attention_squeeze=0.5,
                 projection_layer=None,
                 use_residual=True,
                 use_dense=False,
                 use_norm=True,
                 L2=0.0, L1=0.0,
                 positional_embedding=True,
                 force_relevant_context=False,
                 transformer_blocks=1,
                 use_self_attention=False,
                 use_attention=True,
                 full_targets=True,
                 ):
        super(FFNN_decoder, self).__init__()
        self.full_targets = full_targets
        self.projection_layer = projection_layer

        self.pseudo_embedding = tf.keras.layers.Dense(num_initial_features,
                              activation=activation,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                              kernel_initializer=initializer,
                              use_bias=True)
        # self.pseudo_embedding = Dropout_wrapper(self.pseudo_embedding, dropout_rate=0.1)
        if positional_embedding:
            self.pseudo_embedding = Positional_embedding_wrapper(self.pseudo_embedding, max_length=max_length_sequence_history)
        if use_norm:
            self.pseudo_embedding = Norm_wrapper(self.pseudo_embedding, norm='layer')

        # TODO: DANIEL CHECK THE ATTENTION FEATURE SIZE
        attention_features = num_initial_features * attention_squeeze
        self.use_self_attention = use_self_attention
        self.transformer = []

        for num_block in range(transformer_blocks):
            block = {}
            if use_self_attention:
                self_attention = multihead_attentive_layer(num_heads= attention_heads,
                                                      use_dropout=False,
                                                      L2=L2, L1=L1,
                                                      output_units=num_initial_features,
                                                      max_length_sequence_query=max_length_sequence_history,
                                                           layer_name='decoder_sa'+str(num_block))
                if use_residual:
                    self_attention = Residual_wrapper(self_attention)
                if use_norm:
                    self_attention = Norm_wrapper(self_attention, norm='layer')
                block['self_attention'] = self_attention

            if use_attention:
                attention = multihead_attentive_layer(units_per_head=[int(attention_features)] * attention_heads,
                                                    use_dropout=False,
                                                    L2=L2, L1=L1,
                                                    output_units=num_initial_features,
                                                    max_length_sequence_query=max_length_sequence_history,
                                                    max_length_sequence_value=max_length_sequence_supplement,
                                                    layer_name='decoder_a'+str(num_block))

                if use_residual:
                    attention = Residual_wrapper(attention)
                if use_norm:
                    attention = Norm_wrapper(attention, norm='layer')
                block['attention'] = attention

            transform = transformation_layer(num_initial_features, L1=L1, L2=L2, use_dropout=False)
            if use_residual:
                transform = Residual_wrapper(transform)
            if use_norm:
                transform = Norm_wrapper(transform, norm='layer')
            # if positional_embedding and num_block < transformer_blocks-1:
            #     transform = Positional_embedding_wrapper(transform, max_length_sequence)
            block['transform'] = transform

            self.transformer.append(block)


    def call(self, history_input, attention_value, timesteps):
        forecast = tf.keras.backend.in_train_phase(self.training_call(history_input, attention_value, timesteps),
                                                   alt=self.inference_call(history_input, attention_value, timesteps),
                                                   )
        return forecast

    def training_call(self, history_input, attention_value, timesteps):
        self.max_length = history_input.shape[1]
        signal = history_input

        signal = self.pseudo_embedding(signal) #Pseudo Embeddings
        # signal = self.crop(signal)
        # Put Transformer Decoder

        for block in self.transformer:
            if 'self_attention' in block:
                signal = block['self_attention'](signal)
            if 'attention' in block:
                signal = block['attention'](signal, attention_value)
            if 'transform' in block:
                signal = block['transform'](signal)

        # For predicting full targets vs. last 24 steps
        if not self.full_targets:
            signal = signal[:, -timesteps:, :]
        else:
            signal = signal

        # signal = signal[:, 24:,:]
        return self.projection_layer(signal)

    def inference_call(self, history_input, attention_value, timesteps):

        input_signal = history_input
        for step in range(timesteps):
            # make sure the signal we're feeding to the CNN always has the same length, because TF sucks
            # projected_history = self.projection(history_input)
            signal = self.pseudo_embedding(input_signal)

            if step == 0:
                buffers = []

            for num_block in range(len(self.transformer)):
                block = self.transformer[num_block]

                if 'self_attention' in block:

                    if step == 0:
                        buffers.append(signal)
                        signal = block['self_attention'](signal)
                    else:
                        query = signal[:,-1:,:]
                        buffers[num_block] = tf.concat([buffers[num_block], query], axis=1)
                        signal = block['self_attention'](query, buffers[num_block])

                if 'attention' in block:
                    signal = block['attention'](signal, attention_value)
                if 'transform' in block:
                    signal = block['transform'](signal)

            forecast_step = self.projection_layer(signal[:,-1:,:])
            input_signal = tf.concat([input_signal, forecast_step], axis=1)


        return input_signal[:,-timesteps:,:]

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
                                                       L1=L1, L2=L2,
                                                       output_units=units[block][-1],
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
                                        alt=self.validation_call(history),)

        return forecast

########################################################################################################################

class Transformer_encoder(tf.keras.layers.Layer):
    def __init__(self,
                num_initial_features,
                max_length_sequence_supplement,
                transformer_blocks=1,
                attention_heads=5,
                attention_squeeze=0.5,
                L1=0.0, L2=0.0,
                positional_embedding=True,
                use_norm=False,
                use_residual=False,
                ):
        super(Transformer_encoder, self).__init__()
        self.transformer_blocks = transformer_blocks

        self.pseudo_embedding = tf.keras.layers.Dense(num_initial_features,
                                                    activation=activation,
                                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                    kernel_initializer=initializer,
                                                    use_bias=True)
        if positional_embedding:
            self.pseudo_embedding = Positional_embedding_wrapper(self.pseudo_embedding, max_length=max_length_sequence_supplement)
        if use_norm:
            self.pseudo_embedding = Norm_wrapper(self.pseudo_embedding, norm='layer')

        attention_features = num_initial_features * attention_squeeze

        self.self_attention = []
        self.transform = []
        for block in range(transformer_blocks):
            self_attention = multihead_attentive_layer(units_per_head=[int(attention_features)] * attention_heads,
                                                        output_units=num_initial_features,
                                                        L1=0.0, L2=0.0,
                                                        use_dropout=False,
                                                        dropout_rate=0.0,
                                                        max_length_sequence_query=max_length_sequence_supplement)
            if use_residual:
                self_attention = Residual_wrapper(self_attention)
            if use_norm:
                self_attention = Norm_wrapper(self_attention, norm='layer')
            self.self_attention.append(self_attention)
            
            transform = transformation_layer(num_initial_features, L1=L1, L2=L2, use_dropout=False)
            if use_residual:
                transform = Residual_wrapper(transform)
            if use_norm:
                transform = Norm_wrapper(transform, norm='layer')
            self.transform.append(transform)

    def call(self, inputs):
        signal = inputs
        signal = self.pseudo_embedding(signal)
        for block in range(self.transformer_blocks):
            signal = self.self_attention[block](signal)
            signal = self.transform[block](signal)
        return signal

class FFNN_LSTM_decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_initial_features=4*7,
                 max_length_sequence_history=24,
                 max_length_sequence_supplement=24*4,
                 attention_heads=5,
                 attention_squeeze=0.5,
                 projection_layer=None,
                 use_residual=True,
                 use_dense=False,
                 use_norm=True,
                 L2=0.0, L1=0.0,
                 positional_embedding=True,
                 force_relevant_context=False,
                 transformer_blocks=1,
                 use_self_attention=False,
                 use_attention=True,
                 full_targets=True
                 ):
        super(FFNN_LSTM_decoder, self).__init__()
        self.full_targets = full_targets
        self.projection_layer = projection_layer

        self.pseudo_embedding = tf.keras.layers.Dense(num_initial_features,
                              activation=activation,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                              kernel_initializer=initializer,
                              use_bias=True)
        # self.pseudo_embedding = Dropout_wrapper(self.pseudo_embedding, dropout_rate=0.1)
        if positional_embedding:
            self.pseudo_embedding = Positional_embedding_wrapper(self.pseudo_embedding, max_length=max_length_sequence_history)
        if use_norm:
            self.pseudo_embedding = Norm_wrapper(self.pseudo_embedding, norm='layer')

        attention_features = num_initial_features * attention_squeeze
        self.use_self_attention = use_self_attention
        self.transformer = []

        for num_block in range(transformer_blocks):
            block = {}
            if use_self_attention:
                lstm_layer = MultiLayer_LSTM(units=[[num_initial_features]],
                                            use_norm=use_norm,
                                            use_residuals=use_residual,
                                            L1=L1, L2=L2,)
                block['lstm'] = lstm_layer

            if use_attention:
                attention = multihead_attentive_layer(units_per_head=[int(attention_features)] * attention_heads,
                                                    use_dropout=False,
                                                    L2=L2, L1=L1,
                                                    output_units=num_initial_features,
                                                    max_length_sequence_query=max_length_sequence_history,
                                                    max_length_sequence_value=max_length_sequence_supplement,
                                                    layer_name='decoder_a'+str(num_block))

                if use_residual:
                    attention = Residual_wrapper(attention)
                if use_norm:
                    attention = Norm_wrapper(attention, norm='layer')
                block['attention'] = attention

            transform = transformation_layer(num_initial_features, L1=L1, L2=L2, use_dropout=False)
            if use_residual:
                transform = Residual_wrapper(transform)
            if use_norm:
                transform = Norm_wrapper(transform, norm='layer')
            # if positional_embedding and num_block < transformer_blocks-1:
            #     transform = Positional_embedding_wrapper(transform, max_length_sequence)
            block['transform'] = transform

            self.transformer.append(block)


    def call(self, history_input, teacher, attention_value, timesteps):
        return tf.keras.backend.in_train_phase(self.training_call(history_input, teacher, attention_value, timesteps),
                                               alt=self.inference_call(history_input, teacher, attention_value,
                                                                       timesteps),)

    def training_call(self, history_input, teacher, attention_value, timesteps):
        self.max_length = history_input.shape[1] + teacher.shape[1]
        signal = tf.concat([history_input, teacher], axis=1)

        signal = self.pseudo_embedding(signal) #Pseudo Embeddings
        # signal = self.crop(signal)
        # Put Transformer Decoder
        for block in self.transformer:
            if 'lstm' in block:
                signal, states = block['lstm'](signal)
            if 'attention' in block:
                signal = block['attention'](signal, attention_value)
            if 'transform' in block:
                signal = block['transform'](signal)

        # For predicting full targets vs. last 24 steps
        if not self.full_targets:
            signal = signal[:, -timesteps:, :]
        forecast = self.projection_layer(signal)
        return forecast

    def inference_call(self, history_input, teacher, attention_value, timesteps):

        input_signal = history_input
        for step in range(timesteps):
            # make sure the signal we're feeding to the CNN always has the same length, because TF sucks
            # projected_history = self.projection(history_input)
            signal = self.pseudo_embedding(input_signal)

            if step == 0:
                buffers = [] # history + forecast

            for num_block in range(len(self.transformer)):
                block = self.transformer[num_block]

                if 'lstm' in block:

                    if step == 0:
                        buffers.append(signal)
                        signal, states = block['lstm'](signal)
                    else:
                        query = signal[:,-1:,:]
                        buffers[num_block] = tf.concat([buffers[num_block], query], axis=1)
                        signal, states = block['lstm'](signal) #(buffers[num_block]) # try just passing signal to it

                if 'attention' in block:
                    signal = block['attention'](signal, attention_value)
                if 'transform' in block:
                    signal = block['transform'](signal)

            forecast_step = self.projection_layer(signal[:,-1:,:])
            input_signal = tf.concat([input_signal, forecast_step], axis=1)


        return input_signal[:,-timesteps:,:]

class FFNN_block(tf.keras.layers.Layer):
    def __init__(self,
                 num_initial_features,
                 num_layers,
                 use_residual=True,
                 use_norm=True,
                 L2=0.0, L1=0.0,):
        super(FFNN_block, self).__init__()

        self.num_layers = num_layers

        self.ffnn = []
        for layer in range(num_layers):
            one_ffnn = tf.keras.layers.Dense(num_initial_features,
                                            activation=activation,
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                            kernel_initializer=initializer,
                                            use_bias=True)
            if use_norm:
                one_ffnn = Norm_wrapper(one_ffnn, norm='layer')
            if use_residual:
                one_ffnn = Residual_wrapper(one_ffnn)
            self.ffnn.append(one_ffnn)

    def call(self, inputs, timesteps):
        signal = inputs
        for layer in range(self.num_layers):
            signal = self.ffnn[layer](signal)
        signal = signal[:,-timesteps:,:]
        return signal

#################################################################################################################################################
# DELETE AFTER S2S EXPERIMENTS DELETE AFTER S2S EXPERIMENTS DELETE AFTER S2S EXPERIMENTS DELETE AFTER S2S EXPERIMENTS DELETE AFTER S2S EXPERIMENTS 

class old_block_LSTM(tf.keras.layers.Layer):
    def __init__(self, units=20,
                num_encoder_blocks=1,
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_norm=False,
                 use_hw=False, use_residual=False,
                 L1=0.0, L2=0.0,
                 return_state=True,
                 gru=True):
        super(old_block_LSTM, self).__init__()
        self.return_state = return_state
        self.encoder_blocks = []
        for block in range(num_encoder_blocks):
            if not gru:
                self.encoder_blocks.append(MultiLayer_LSTM(units=[[units]],
                                                    use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                    use_norm=use_norm,
                                                    use_hw=use_hw, use_residuals=use_residual,
                                                    L1=L1, L2=L2))
            else:
                self.encoder_blocks.append(MultiLayer_GRU(units=[[units]],
                                                    use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                    use_norm=use_norm,
                                                    use_hw=use_hw, use_residuals=use_residual,
                                                    L1=L1, L2=L2))

    def call(self, signal, initial_states=None):
        
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

class old_decoder_LSTM_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=20,
                 num_decoder_blocks=1,
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_hw=False, use_residual=False,
                 use_norm=False,
                 L1=0.0, L2=0.0,
                 use_attention=False, attention_heads=3,
                 projection_layer=None,
                 gru=True):
        super(old_decoder_LSTM_block, self).__init__()
        self.units = units
        self.use_attention = use_attention
        self.projection_layer=projection_layer

        if self.use_attention:
            self.attention_blocks = []
            # self.input_attention = Attention(int(units[0][0]/2),
            #                         mode='Transformer',
            #                         query_kernel= 1,
            #                         key_kernel=1,
            #                         use_dropout=use_dropout,
            #                         dropout_rate=dropout_rate,
            #                         L2=L2, L1=L1,
            #                         only_context=True)
        self.decoder_blocks = []

        for block in range(num_decoder_blocks):
            if not gru:
                self.decoder_blocks.append(MultiLayer_LSTM(units=[[units]],
                                                    use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                    use_norm=use_norm,
                                                    use_hw=use_hw, use_residuals=use_residual,
                                                    L1=L1, L2=L2))
            else:
                self.decoder_blocks.append(MultiLayer_GRU(units=[[units]],
                                                use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                use_norm=use_norm,
                                                use_hw=use_hw, use_residuals=use_residual,
                                                L1=L1, L2=L2))
            if self.use_attention:
                attention_units = int(units/2) #equivalent to the dimensionality of attention heads
                attention = Attention(attention_units,
                                    mode='Transformer',
                                    use_dropout=use_dropout,
                                    dropout_rate=dropout_rate,
                                    L2=L2, L1=L1,)
                # attention = Attention(attention_units,
                #                     mode='Transformer',
                #                     query_kernel= 1,
                #                     key_kernel=1,
                #                     use_dropout=use_dropout,
                #                     dropout_rate=dropout_rate,
                #                     L2=L2, L1=L1,
                #                     only_context=True)
                self.attention_blocks.append(attention)

    def call(self, prev_history, decoder_init_state, teacher=None, attention_value=None, timesteps=12):

        decoder_state = decoder_init_state # [[[state_h, state_c]]] #(blocks, layers_in_block, 2)
        # decoder state is supposed to be the storage for the states
        # block state is the temporal buffer, because the layout suxx
        # ToDo: this is throwing errors for anything with
        for timestep in range(timesteps + prev_history.shape[1] - 1):
            if timestep < prev_history.shape[1]:
                input = tf.expand_dims(prev_history[:,timestep,:], axis=1)
                decoder_out, decoder_state = self.decoder_step(input, decoder_state, attention_value)
                if timestep == prev_history.shape[1] - 1:
                    forecast = self.project_and_concat(decoder_out, None, first_step=True)
            else:

                if teacher is not None:
                    input = tf.keras.backend.in_train_phase(tf.expand_dims(teacher[:,timestep - prev_history.shape[1],:], axis=1),
                                                             alt=tf.expand_dims(forecast[:,-1,:], axis=1),
                                                             training=tf.keras.backend.learning_phase())
                else:
                    input = tf.expand_dims(forecast[:, -1, :], axis=1)
                decoder_out, decoder_state = self.decoder_step(input, decoder_state, attention_value)
                forecast = self.project_and_concat(decoder_out, forecast, first_step=False)

        return forecast


    def decoder_step(self, input, decoder_state, attention_value):
            # attention layers and assign those to the decoder states later.
            signal = input

            block_states = []
            for num_block in range(len(self.decoder_blocks)):

                if self.use_attention:
                    query = tf.concat(decoder_state[num_block][-1] , axis=-1)
                    query = tf.expand_dims(query, axis=1)
                    if num_block == 0:
                        query = tf.concat([query, signal], axis=-1)
                    attention_context = self.attention_blocks[num_block](query, value=attention_value[num_block])
                    signal = tf.concat([signal, attention_context], axis=-1)
                    
                signal, block_state = self.decoder_blocks[num_block](signal, initial_states=[decoder_state[num_block]])
                block_states.append(block_state[0])

            decoder_state = block_states
            return signal, decoder_state

    def project_and_concat(self, unprocessed_timestep, previous_processed_timesteps, first_step=True):
            if first_step:
                return self.projection_layer(unprocessed_timestep)
            else:
                return tf.concat([previous_processed_timesteps, self.projection_layer(unprocessed_timestep)], axis=1)

class luong_decoder_LSTM_block(tf.keras.layers.Layer):
    def __init__(self,
                 units=20,
                 num_decoder_blocks=1,
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_hw=False, use_residual=False,
                 use_norm=False,
                 L1=0.0, L2=0.0,
                 use_attention=False, attention_heads=3,
                 projection_layer=None,
                 gru=True):
        super(luong_decoder_LSTM_block, self).__init__()
        self.units = units
        self.use_attention = use_attention
        self.projection_layer=projection_layer

        if self.use_attention:
            self.attention_blocks = []
        self.decoder_blocks = []

        self.input_projection = tf.keras.layers.Dense(units=50+units)

        for block in range(num_decoder_blocks):
            if not gru:
                self.decoder_blocks.append(MultiLayer_LSTM(units=[[units]],
                                                    use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                    use_norm=use_norm,
                                                    use_hw=use_hw, use_residuals=use_residual,
                                                    L1=L1, L2=L2))
            else:
                self.decoder_blocks.append(MultiLayer_GRU(units=[[units]],
                                                use_dropout=use_dropout, dropout_rate=dropout_rate,
                                                use_norm=use_norm,
                                                use_hw=use_hw, use_residuals=use_residual,
                                                L1=L1, L2=L2))
        if self.use_attention:
            attention_units = int(units/2) #equivalent to the dimensionality of attention heads
            attention = Attention(attention_units,
                                    mode='Transformer',
                                    use_dropout=use_dropout,
                                    dropout_rate=dropout_rate,
                                    L2=L2, L1=L1,)
            self.attention_blocks.append(attention)
            
            self.attn_proj = tf.keras.layers.Dense(units)

    def call(self, prev_history, decoder_init_state, teacher=None, attention_value=None, timesteps=12):

        decoder_state = decoder_init_state # [[[state_h, state_c]]] #(blocks, layers_in_block, 2)
        # decoder state is supposed to be the storage for the states
        # block state is the temporal buffer, because the layout suxx
        # ToDo: this is throwing errors for anything with
        for timestep in range(timesteps + prev_history.shape[1] - 1):
            if timestep < prev_history.shape[1]:
                input = tf.expand_dims(prev_history[:,timestep,:], axis=1)
                input = self.input_projection(input)
                decoder_out, decoder_state, attention_context = self.decoder_step(input, decoder_state, attention_value)
                if timestep == prev_history.shape[1] - 1:
                    forecast = self.project_and_concat(decoder_out, None, first_step=True)
            else:

                if teacher is not None:
                    input = tf.keras.backend.in_train_phase(tf.expand_dims(teacher[:,timestep - prev_history.shape[1],:], axis=1),
                                                             alt=tf.expand_dims(forecast[:,-1,:], axis=1),
                                                             training=tf.keras.backend.learning_phase())
                else:
                    input = tf.expand_dims(forecast[:, -1, :], axis=1)

                input = tf.concat([input,attention_context], axis=-1)
                decoder_out, decoder_state, attention_context = self.decoder_step(input, decoder_state, attention_value)
                forecast = self.project_and_concat(decoder_out, forecast, first_step=False)

        return forecast


    def decoder_step(self, input, decoder_state, attention_value):
            # attention layers and assign those to the decoder states later.
            signal = input

            block_states = []
            for num_block in range(len(self.decoder_blocks)):

                # if self.use_attention:
                #     query = tf.concat(decoder_state[num_block][-1] , axis=-1)
                #     query = tf.expand_dims(query, axis=1)
                #     if num_block == 0:
                #         query = tf.concat([query, signal], axis=-1)
                #     attention_context = self.attention_blocks[num_block](query, value=attention_value[num_block])
                #     signal = tf.concat([signal, attention_context], axis=-1)
                    
                signal, block_state = self.decoder_blocks[num_block](signal, initial_states=[decoder_state[num_block]])
                block_states.append(block_state[0])

            if self.use_attention:
                # query = tf.concat(block_states[-1][-1] , axis=-1)
                # query = tf.expand_dims(query, axis=1)
                # value = tf.concat(attention_value, axis=-1)
                # attention_context = self.attention_blocks[0](query, value)
                # signal = tf.concat([signal, attention_context], axis=-1)
                query = signal
                # query = tf.expand_dims(query, axis=1)
                value = tf.concat(attention_value, axis=-1)
                attention_context = self.attention_blocks[0](query, value)

                attn_vector = tf.concat([attention_context, query], axis=-1)
                attn_vector = self.attn_proj(attn_vector)
                attn_vector = tf.tanh(attn_vector)

            decoder_state = block_states
            return attn_vector, decoder_state, attn_vector

    def project_and_concat(self, unprocessed_timestep, previous_processed_timesteps, first_step=True):
            if first_step:
                return self.projection_layer(unprocessed_timestep)
            else:
                return tf.concat([previous_processed_timesteps, self.projection_layer(unprocessed_timestep)], axis=1)

########################################################################################################################
class ForecasterModel(tf.keras.Model):
    def __init__(self,
                output_shape,
                encoder_specs,
                decoder_specs,
                model_type='LSTM',
                history_and_features = True,
                ):
        super(ForecasterModel, self).__init__()
        self.out_steps = output_shape[-2]
        self.history_and_features = history_and_features
        self.model_type = model_type

        # identify the model type and assemble accordingly
        if model_type == 'LSTM-Generator':
            self.encoder = block_LSTM(**encoder_specs)
            self.decoder = decoder_LSTM_block(**decoder_specs)
        elif model_type == 'CNN-Generator':
            self.encoder = CNN_encoder(**encoder_specs)
            self.decoder = CNN_decoder(**decoder_specs)
        elif model_type == 'FFNN-Generator' and history_and_features:
            self.encoder = FFNN_encoder(**encoder_specs)
            self.decoder = FFNN_decoder(**decoder_specs)
        elif model_type == 'FFNN-Generator' and not history_and_features:
            self.decoder = FFNN_decoder(**decoder_specs)
        elif model_type == 'FFNN-LSTM-Generator' and history_and_features:
            self.encoder = FFNN_encoder(**encoder_specs)
            self.decoder = FFNN_LSTM_decoder(**decoder_specs)
        elif model_type == 'FFNN-LSTM-Generator' and not history_and_features:
            self.decoder = FFNN_LSTM_decoder(**decoder_specs)
        elif model_type == 'Transformer-Generator':
            self.encoder = Transformer_encoder(**encoder_specs)
            self.decoder = FFNN_decoder(**decoder_specs)

    def call(self, inputs,):
        # undictionary inputs (TF doesn't let you have multiple inputs)
        support_input = inputs['support_input']
        history_input = inputs['history_input']
        # print(support_input, history_input, teacher)
        # run the model
        if self.history_and_features:
            encoder_features = self.encoder(support_input)
            forecast = self.decoder(history_input,
                                    attention_value=encoder_features,
                                    timesteps=self.out_steps)
        else:
            forecast = self.decoder(history_input,
                                    attention_value=None,
                                    timesteps=self.out_steps)
        return forecast

class S2SModel(tf.keras.Model):
    def __init__(self,
                output_shape,
                encoder_specs,
                decoder_specs,
                model_type,
                ):
        super().__init__()
        self.out_steps = output_shape[-2]
        self.model_type = model_type

        # if model_type == 'E-D':
        #     self.encoder = block_LSTM(**encoder_specs)
        #     self.decoder = decoder_LSTM_block(**decoder_specs)
        # DELETE AFTER EXPERIMENTS
        if model_type == 'E-D':
            self.encoder = old_block_LSTM(**encoder_specs)
            self.decoder = old_decoder_LSTM_block(**decoder_specs)
        elif model_type == 'Transformer':
            self.encoder = Transformer_encoder(**encoder_specs)
            self.decoder = FFNN_decoder(**decoder_specs)

    def call(self, inputs):
        # undictionary inputs (TF doesn't let you have multiple inputs)
        nwp_pv_input = inputs['nwp_pv_input']
        teacher = inputs['teacher']
        decoder_input = tf.expand_dims(teacher[:,0,:], axis=1)

        if self.model_type == 'E-D':
            encoder_outputs, encoder_states = self.encoder(nwp_pv_input)
            forecast = self.decoder(decoder_input,
                                    teacher=teacher[:,1:,:],
                                    decoder_init_state=encoder_states,
                                    attention_value=encoder_outputs,
                                    timesteps=self.out_steps)
        elif self.model_type == 'Transformer':
            encoder_outputs = self.encoder(nwp_pv_input)
            forecast = self.decoder(decoder_input, 
                                    attention_value=encoder_outputs,
                                    timesteps=self.out_steps,
                                    teacher=teacher[:,1:,:])

        return forecast

class MIMOModel(tf.keras.Model):
    def __init__(self,
                output_shape,
                encoder_specs,
                model_type,
                projection_block=None,
                ):
        super().__init__()

        self.out_steps = output_shape[-2]
        if model_type == 'MiMo-LSTM':
            self.encoder = old_block_LSTM(**encoder_specs)
        self.squeeze_time = tf.keras.layers.Dense(self.out_steps)
        self.projection_layer = projection_block

    def call(self, inputs):
        nwp_pv_input = inputs['nwp_pv_input']

        forecast, _ = self.encoder(nwp_pv_input)

        forecast = tf.transpose(forecast, perm=[0, 2, 1])
        forecast = self.squeeze_time(forecast)
        forecast = tf.transpose(forecast, perm=[0, 2, 1])

        forecast = self.projection_layer(forecast)
        return forecast


########################################################################################################################

########################################################################################################################
