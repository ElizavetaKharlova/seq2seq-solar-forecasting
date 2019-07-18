# contains all the building blocks necessary to build the architectures we will be working on
# In a sense, this will be the main modiefied file

# ToDO: clean up commentary

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

#ToDo: change in a way, that we can pass different encoders and decoders as arguments or something
def build_model(E_D_layers, E_D_units, in_shape, out_shape, dropout_rate=0.3):

    keras.backend.clear_session()  # make sure we are working clean

    layers = E_D_layers  # remember units and layers
    units = E_D_units
    #   Encoder init
    encoder_inputs = tf.keras.layers.Input(shape=(in_shape[1], in_shape[2]))
    encoder = encoder_LSTM(layers, units, use_dropout=True, dropout_rate=dropout_rate)
    # ------------------------------------------------------------------------------------------
    #   Decoder init
    decoder_inputs = tf.keras.layers.Input(shape=(out_shape[1], out_shape[2]))
    blend_factor_input = tf.keras.layers.Input(shape=(1))
    decoder = decoder_LSTM(layers, units, use_dropout=True, dropout_rate=dropout_rate, out_shape=out_shape[1:])
    # ------------------------------------------------------------------------------------------
    # Run
    encoder_outputs, encoder_states = encoder(encoder_inputs)
    decoder_output = decoder(decoder_inputs, encoder_outputs=encoder_outputs, initial_states=encoder_states,
                             blend_factor_input=blend_factor_input)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([encoder_inputs, decoder_inputs, blend_factor_input], decoder_output)
    return model

# Application of feed-forward & recurrent dropout on encoder & decoder hidden states.
# ToDO: explain to Daniel, why is_training is in the Init, and how we change isTraining during the validation process?
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
        else:  # test/validation time
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
        if self.isTraining:  # apply dropout at the training stage
            # add feed forward dropout
            inputs = tf.nn.dropout(inputs, self.zoneout_prob)
            # get outputs & hidden states in one layer of the network
            output, state_h, state_c = self._layer(inputs, initial_state)
            # apply zoneout to each state of an LSTM
            state_h = (1 - self.zoneout_prob) * tf.nn.dropout((state_h - initial_state[0]), self.zoneout_prob) + state_h
            state_c = (1 - self.zoneout_prob) * tf.nn.dropout((state_c - initial_state[1]), self.zoneout_prob) + state_c
            # ToDo: add functionality for GRU?
        else:  # test/validation time
            output, state_h, state_c = self._layer(inputs, initial_state)
            state_h = (1 - self.zoneout_prob) * state_h + self.zoneout_prob * initial_state[0]
            state_c = (1 - self.zoneout_prob) * state_c + self.zoneout_prob * initial_state[1]
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
                                                  return_sequences=True,
                                                  return_state=True,
                                                  recurrent_initializer='glorot_uniform')

            # if it is the first layer of we do not want to use dropout, we won't
            # otherwise add the zoneout wrapper

            if layer == 0 or self.use_dropout == False:
                self.LSTM.append(one_LSTM_layer)
            else:
                self.LSTM.append(
                    ZoneoutWrapper(one_LSTM_layer, zoneout_prob=self.dropout_rate, isTraining=self.isTraining))

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


class Attention(tf.keras.Model):
    def __init__(self, units, bahdanau):
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
# ToDO: same thing here, add an isTraining thingie, where the blend factor will be used, and if isTraining is false then the blend_factor will be multiplied by 0

class decoder_LSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, out_shape, use_dropout=True, dropout_rate=0.0, use_attention=True,
                 attention_hidden=False):
        super(decoder_LSTM, self).__init__()
        self.layers = num_layers
        self.units = num_units
        self.out_shape = out_shape
        self.use_attention = use_attention
        self.attention_hidden = attention_hidden

        self.reshape_to_1ts = tf.keras.layers.Reshape(
            (1, self.out_shape[1]))  # Keras is finicky with dimensions, this is to make sure the dim is specified
        self.concat_timestep = tf.keras.layers.Concatenate(axis=1)  # concatenates the timeteps together

        self.attention = Attention(num_units, bahdanau=False) # choose the attention style (Bahdanau, Transformer)
        self.decoder = MultiLayer_LSTM(num_layers, num_units, use_dropout, dropout_rate)

        self.decoder_wrap = []
        if self.units > 512:
            self.decoder_wrap.append(tf.keras.layers.Dense(units=256,
                                                           activation='relu'))
        self.decoder_wrap.append(tf.keras.layers.Dense(units=self.out_shape[1]))

    def call(self, decoder_inputs, encoder_outputs, initial_states, blend_factor_input):

        blend_factor = self.reshape_to_1ts(
            blend_factor_input)  # because, otherwise keras will be acting wierd as the decoder_output is (1,1) and this would be (1)

        dec_states_t = initial_states  # get the initial states for the first timestep
        output_decoder_LSTM = encoder_outputs  # for the first time step for attention

        for t in range(self.out_shape[0]):  # for loop over all timesteps

            # determine the input to the decoder
            target_support_t = self.reshape_to_1ts(decoder_inputs[:, t, :])  # slice a timestep of the support
            if t == 0:  # if no previous output we cannot blend
                dec_in_t = target_support_t
            else:  # otherwise blend as desired
                dec_in_t = (1.0 - blend_factor) * dec_out_t + blend_factor * target_support_t

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

            # construct the output time-series
            if t == 0:
                decoder_output = dec_out_t
            else:  # else we append
                decoder_output = self.concat_timestep([decoder_output, dec_out_t])

        return decoder_output
