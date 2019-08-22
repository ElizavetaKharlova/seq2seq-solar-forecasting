# contains all the building blocks necessary to build the architectures we will be working on
# In a sense, this will be the main modiefied file

# ToDO: clean up commentary
#ToDo: change in a way, that we can pass different encoders and decoders as arguments or something
# ToDO: change everything to tf.float32 for speeeeed


import tensorflow as tf
from tensorflow import keras
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


def build_model(E_D_layers, E_D_units,
                in_shape, out_shape,
                dropout_rate=0.2,
                use_dropout=False,
                CNN_encoder=False, LSTM_encoder=True,
                LSTM_decoder=True,
                use_attention=False):

    keras.backend.clear_session()  # make sure we are working clean

    layers = E_D_layers  # remember units and layers
    units = E_D_units
    #   Encoder init
#    encoder_inputs = tf.keras.layers.Input(shape=(in_shape[1], in_shape[2]))
    network_inputs = tf.keras.layers.Input(shape=(in_shape[1], in_shape[2]))
    network = TCN(layers, units, out_shape=out_shape[1:], dropout_rate=dropout_rate, use_dropout=True)
    
#    if CNN_encoder:
#        encoder = encoder_CNN(layers, units, use_dropout=use_dropout, dropout_rate=dropout_rate)
#    if LSTM_encoder:
#        encoder = encoder_LSTM(layers, units, use_dropout=use_dropout, dropout_rate=dropout_rate)
    # ------------------------------------------------------------------------------------------
    #   Decoder init
#    decoder_inputs = tf.keras.layers.Input(shape=(out_shape[1], out_shape[2]))
#    blend_factor_input = tf.keras.layers.Input(shape=(1))
#    if LSTM_decoder:
#        decoder = decoder_LSTM(layers, units,
#                               use_dropout=use_dropout, dropout_rate=dropout_rate,
#                               out_shape=out_shape[1:],
#                               use_attention=use_attention)
#    else:
#        print('whoops, no other decoders available atm!!')
    # ------------------------------------------------------------------------------------------
    # Run
    network_outputs = network(network_inputs)
#    if CNN_encoder:
#        encoder_outputs = encoder(encoder_inputs)
#        encoder_states=None
#    if LSTM_encoder:
#        encoder_outputs, encoder_states = encoder(encoder_inputs)
#
#    if LSTM_decoder:
#        decoder_output = decoder(decoder_inputs, encoder_outputs=encoder_outputs, initial_states=encoder_states,
#                                 blend_factor_input=blend_factor_input)


    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
#    model = tf.keras.Model([encoder_inputs, decoder_inputs, blend_factor_input], decoder_output)
    model = tf.keras.Model(network_inputs, network_outputs)
    return model

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
            return tf.nn.dropout(input_to_drop_from, self.dropout_prob)

        inputs = tf.keras.backend.in_train_phase(x=drop_stuff(inputs), alt=inputs, training=istraining)

        # get the output and hidden states in one layer of the network
        output, state_h, state_c = self._layer(inputs, initial_state)
        # apply dropout to each state of an LSTM layer
        # see which states to use


        state_h = tf.keras.backend.in_train_phase(x=drop_stuff(state_h), alt=state_h, training=istraining)
        state_c = tf.keras.backend.in_train_phase(x=drop_stuff(state_c), alt=state_c, training=istraining)

        # state_h = self.dropout_layer(state_h, istraining)
        # state_c = self.dropout_layer(state_c, istraining)

        # def drop_state_c():
        #     return tf.nn.dropout(state_c, self.dropout_prob)
        #
        #
        # ToDo: GRU functionality?
        # else:  # test/validation time
        #     output, state_h, state_c = self._layer(inputs, initial_state)
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

        self.num_layers = num_layers  # simple, just save all the things we might need
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.use_norm = use_norm
        self.isTraining = isTraining

        self.LSTM = []  # Multiple layers work easiest as a list of layers, so here we start
        self.norm = []
        for layer in range(self.num_layers):
            # get one LSTM layer per layer we speciefied, with the units we need

            # Do we need to initialize the layer in the loop?
            one_LSTM_layer = tf.keras.layers.LSTM(self.num_units,
                                                  activation=tf.nn.tanh,
                                                  return_sequences=True,
                                                  return_state=True,
                                                  recurrent_initializer='glorot_uniform')

            # if it is the first layer of we do not want to use dropout, we won't
            # otherwise add the zoneout wrapper

            if layer == 0 or self.use_dropout == False:
                self.LSTM.append(one_LSTM_layer)
            else:
                self.LSTM.append(
                    DropoutWrapper(one_LSTM_layer, dropout_prob=self.dropout_rate)
                            )

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

        all_out = []  # again, we are working with lists, so we might just as well do the same for the outputs
        states = []

        out = inputs  # just a simple trick to prevent usage of one if_else loop for the first layer
        for layer in range(self.num_layers):
            out, state_h, state_c = self.LSTM[layer](out, initial_state=initial_states[layer])
            if self.use_norm:
                out = self.norm[layer](out)
            all_out.append(out)
            states.append([state_h, state_c])

        return all_out, states

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

class FFN(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, out_shape):
        super(FFN, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.out_shape = out_shape # shape [timesteps, variables]
        self.ffn = []
        for layer in range(self.num_layers):
            ff_layer = tf.keras.layers.Dense(self.num_units)
            self.ffn.append(ff_layer)
        self.ffn.append(tf.keras.layers.Dense(self.out_shape[0]))
        # figure out how the datasets in the paper compare to mine and daniel's
        # then decide on the sizes of benchmarks based on what they have

    def call(self, inputs):
        out = inputs
        # same problem as in TCN, input shape [None, 14, 8] -> [None, 14, 6] does not work with the output [None, 6, 1]
        for layer in range(self.num_layers+1):
            out = self.ffn[layer](out)
        return out


class LSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, out_shape):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.out_shape = out_shape
        self.reshape_to_out = tf.keras.layers.Reshape((self.out_shape[0], self.out_shape[1]))
        
        self.LSTM_net = []
        
        # create a network stack
        for layer in range(self.num_layers):
            one_LSTM_layer = tf.keras.layers.LSTM(self.num_units,
                                                  activation=tf.nn.tanh,
                                                  return_sequences=True,
                                                  return_state=True,
                                                  recurrent_initializer='glorot_uniform')
            self.LSTM_net.append(one_LSTM_layer)
        # last layer, the shape is the whole forecast (MIMO according to the paper)
        self.output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.out_shape[1]))

    def call(self, inputs):
        # maybe add dropout later
        out = inputs
        # add the recursive output later (need input data variables for future predicted (or tf) load values (performs better, need that)
        for layer in range(self.num_layers):
            out = self.LSTM_net[layer](out)
        out = self.reshape_to_out(out)
        return out

class TCN(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, out_shape, dropout_rate, use_dropout, kernel_size=3, strides=1):
        super(TCN, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.kernel_size = kernel_size
        self.strides = strides
        self.out_shape = out_shape
        
        self.CNN = []
        
        # for output calculation according to the paper
        self.conv1D_input = tf.keras.layers.Conv1D(filters=self.num_units,
                                                   kernel_size=self.kernel_size,
                                                   strides=self.strides,
                                                   data_format='channels_last',
                                                   padding='causal')
                                             
        self.conv1D_output = tf.keras.layers.Conv1D(filters=out_shape[0],
                                                    kernel_size=1,
                                                    strides=self.strides,
                                                    data_format='channels_last',
                                                    padding='causal')
        
        # create a network stack
        for layer in range(self.num_layers):
            one_CNN_layer = tf.keras.layers.Conv1D(self.num_units,
                                                    kernel_size=self.kernel_size,
                                                    strides=self.strides,
                                                    data_format='channels_last',
                                                    padding='causal',
                                                    dilation_rate=2**layer) # increase dilation rate each layer (1,2,4,8...)
                                                                            # do layer mod 6 so that the dilation doesn't go above 2**5
            self.CNN.append(one_CNN_layer)

    # create one residual block with 1 LSTM, 1 dropout and an activation layers
    def residual_block(self, inputs, layer, isTraining):
        print(layer)
        out = inputs
        out = self.CNN[layer](out)
        print('OUT SHAPE', out.shape)
        # check if training
        if self.use_dropout: # and isTraining
            out = tf.nn.dropout(out, self.dropout_rate)
        out = tf.keras.activations.relu(out)
        # add the residual connection
        # could add skip connections later
        res_out = tf.keras.layers.add([out, inputs])
        print('RES OUT SHAPE', res_out.shape)
        return res_out

    def call(self, inputs):
        print('SHAPES', inputs.shape)
        # don't know how to change the shape of the input such that the convolution produces [None, 6, 1] not [None, 14, 6]
        inputs = self.conv1D_input(inputs)
        print('SHAPES', inputs.shape)
        res_out = inputs
        isTraining = tf.keras.backend.learning_phase()
        # in the paper they use two inputs, one of historical features and load and one of only the features for the prediction window
        for layer in range(self.num_layers):
            res_out = self.residual_block(res_out, layer, isTraining)
        out = self.conv1D_output(res_out)
        return out

class seq2seq(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units):
        super(seq2seq, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.LSTM_layer = tf.keras.layers.LSTM(self.num_units,
                                               activation=tf.nn.tanh,
                                               return_sequences=True,
                                               return_state=True,
                                               recurrent_initializer='glorot_uniform')
        self.encoder = []
        self.decoder = []
        for layer in range(self.num_layers):
            self.encoder.append(self.LSTM_layer)
            self.decoder.append(self.LSTM_layer)
        self.decoder.append(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.out_shape[1])))
    
#    def encoder():


    def call(inputs):
        out = inputs
        return out
        # implemet teacher forcing for the decoder


# This builds the Encoder_LSTM
# basically just a simple pseudowrapper for the MultiLayer_LSTM layer
# ToDO: add a projection layer, that might be important for the attention mechs later on
# this will reduce the size of attention needed :-)
class encoder_LSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, use_dropout=False, dropout_rate=0.0):
        super(encoder_LSTM, self).__init__()
        self.layers = num_layers
        self.units = num_units
        self.encoder = MultiLayer_LSTM(num_layers, num_units, use_dropout, dropout_rate)

    def call(self, encoder_inputs, initial_states=None):
        encoder_outputs, encoder_states = self.encoder(encoder_inputs, initial_states=initial_states)

        return encoder_outputs, encoder_states

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
# using the MultiLayer_LSTM, all we really need to do is do stepwise unrolling
# the decoder needs additional inputs compared to the decoder ...
# ToDO: same thing here, add an isTraining thingie, where the blend factor will be used, and if isTraining is false then the blend_factor will be multiplied by 0

class decoder_LSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units,
                 out_shape=1,
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_attention=False,
                 attention_hidden=False):
        super(decoder_LSTM, self).__init__()
        self.layers = num_layers
        self.units = num_units
        self.out_shape = out_shape
        self.use_attention = use_attention
        self.attention_hidden = attention_hidden

        self.reshape_to_1ts = tf.keras.layers.Reshape(
            (1, self.out_shape[1]))  # Keras is finicky with dimensions, this is to make sure the dim is specified
        self.reshape_to_one = tf.keras.layers.Reshape( (1, 1))
        self.concat_timestep = tf.keras.layers.Concatenate(axis=1)  # concatenates the timeteps together

        self.attention = Attention(num_units, bahdanau=False) # choose the attention style (Bahdanau, Transformer)
        self.decoder = MultiLayer_LSTM(num_layers, num_units, use_dropout, dropout_rate)

        self.decoder_wrap = []
        if self.units > 512:
            self.decoder_wrap.append(tf.keras.layers.TimeDistributed(
                                                tf.keras.layers.Dense(units=256, activation='relu')
                                                                    )
                                    )

        self.decoder_wrap.append(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.out_shape[1])))

        if self.out_shape[1] > 1:
            self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, decoder_inputs, encoder_outputs, initial_states, blend_factor_input):

        istraining = tf.keras.backend.learning_phase()

        dec_states_t = initial_states  # get the initial states for the first timestep
        output_decoder_LSTM = encoder_outputs  # for the first time step for attention

        for t in range(self.out_shape[0]):  # for loop over all timesteps

            target_support_t = self.reshape_to_1ts(decoder_inputs[:, t, :])  # slice a timestep of the support
            if t == 0:  # if no previous output we cannot blend
                dec_in_t = target_support_t
            else:  # otherwise blend during training and no blend during va
                if istraining:
                    blend_factor = self.reshape_to_one(
                        blend_factor_input)  # because, otherwise keras will be acting wierd as the decoder_output is (1,1) and this would be (1)
                    dec_in_t = (1.0 - blend_factor) * dec_out_t + blend_factor * target_support_t
                else:
                    dec_in_t = dec_out_t

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
            dec_out_t_not_projected = output_decoder_LSTM[-1]  # get the lat layer output of the decoder
            for layer in range(len(self.decoder_wrap)):
                dec_out_t_not_projected = self.decoder_wrap[layer](dec_out_t_not_projected)
            dec_out_t = self.reshape_to_1ts(dec_out_t_not_projected)

            if self.out_shape[-1] > 1:
                dec_out_t = self.softmax(dec_out_t)

            # construct the output time-series
            if t == 0:
                decoder_output = dec_out_t
            else:  # else we append
                decoder_output = self.concat_timestep([decoder_output, dec_out_t])


        return decoder_output
