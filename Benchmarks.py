# contains all the building blocks necessary to build the architectures we will be working on
# In a sense, this will be the main modiefied file

# ToDO: clean up commentary
#ToDo: change in a way, that we can pass different encoders and decoders as arguments or something
# ToDO: change everything to tf.float32 for speeeeed


import tensorflow as tf
from tensorflow import keras

# choose and build a baseline model
# FNN, LSTM, TCN require inputs in shape of [timesteps, features]
# seq2seq requires [encoder_inputs, decoder_inputs] in shape [timesteps, features]
# the output is in shape [timesteps, dimensions]
# in a current form can be used with Train_Eval function without change other than (import from benchmarks) 

def build_model(E_D_layers, E_D_units,
                in_shape, out_shape,
                Dense_blocks=4,
                dropout_rate=0.2,
                use_dropout=True,
                CNN_encoder=False, LSTM_encoder=True,
                LSTM_decoder=True,
                use_attention=False,
                network='dense'):

    keras.backend.clear_session()  # make sure we are working clean

    layers = E_D_layers  # remember units and layers
    units = E_D_units

    if network=='TCN':
        network_inputs = tf.keras.layers.Input(shape=(in_shape[1], in_shape[2]))
        network = TCN(layers, units, out_shape=out_shape[1:], dropout_rate=dropout_rate, use_dropout=True)
    elif network=='LSTM':
        network_inputs = tf.keras.layers.Input(shape=(in_shape[1], in_shape[2]))
        network = LSTM(layers, units, in_shape=in_shape[1:], out_shape=out_shape[1:])
    elif network=='FFN':
        network_inputs = tf.keras.layers.Input(shape=(in_shape[1], in_shape[2]))
        network = FFN(layers, units, out_shape=out_shape[1:])
    elif network=='seq2seq':
        encoder_inputs = tf.keras.layers.Input(shape=(in_shape[1], in_shape[2]))
        decoder_inputs = tf.keras.layers.Input(shape=(out_shape[1], out_shape[2]))
        network_inputs = [encoder_inputs, decoder_inputs]
        network = seq2seq(layers, units, out_shape=out_shape[1:], use_norm=False)
    elif network=='dense':
        network_inputs = tf.keras.layers.Input(shape=(in_shape[0], in_shape[1]))
        network = DenseTCN(Dense_blocks, layers, in_shape, growth_rate=12, squeeze_factor=0.5, use_dropout=False, dropout_rate=dropout_rate, use_norm=False, kernel_sizes=[3, 5])

    # Run
    network_outputs = network(network_inputs)
    model = tf.keras.Model(network_inputs, network_outputs)
    return model

#TODO: clean up dropout wrapper, remove zoneout wrapper
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


# basic fully connected network
#TODO: make it perform better by adding regularization (WiP)
class FFN(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, out_shape):
        super(FFN, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.out_shape = out_shape # shape [timesteps, variables]
        self.reshape_to_out = tf.keras.layers.Reshape((self.out_shape[0], self.out_shape[1]))
        self.ffn = []
        # initialize the layer stack
        for layer in range(self.num_layers):
            ff_layer = tf.keras.layers.Dense(self.num_units)
            self.ffn.append(ff_layer)
        # add the output layer with required output shape
        self.ffn.append(tf.keras.layers.Dense(self.out_shape[0]))

    def call(self, inputs):
        out = inputs
        # run the network
        for layer in range(self.num_layers+1):
            out = self.ffn[layer](out)
        # take last time step as prediction
        out = out[:,-1,:]
        out = self.reshape_to_out(out) # reshape to be same as target
        return out

# basic multilayer LSTM network built in a recursive manner 
# such that it predicts one step ahead and uses the prediction as part of the target for the next step
# requires number of layers, units, and input and output shapes to initialize
# requires network inputs to call
# outputs the prediction in shape of target
class LSTM(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, in_shape, out_shape):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.out_shape = out_shape
        self.in_shape = in_shape
        # convert a list of tensors into an array
        self.reshape_to_out = tf.keras.layers.Reshape((self.out_shape[0], self.out_shape[1]))
        # concatenate inputs with an output slice to provide recursion
        self.concat_timestep = tf.keras.layers.Concatenate(axis=1)
        
        self.LSTM_net = []
        self.norm = []
        
        # create a network layer stack
        for layer in range(self.num_layers):
            one_LSTM_layer = tf.keras.layers.LSTM(self.num_units,
                                                  activation=tf.nn.tanh,
                                                  return_sequences=True,
                                                  return_state=True,
                                                  recurrent_initializer='glorot_uniform')
            self.LSTM_net.append(one_LSTM_layer)
            self.norm.append(tf.keras.layers.LayerNormalization())
        # layer used to reshape the number of features in order to concatenate the timesteps in a recursive loop
        self.reshape_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.in_shape[1]))
        # layer used to receive one data sample as a 1 step ahead prediction
        self.output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.out_shape[1]))

    def call(self, inputs):
        all_out = []
        for t in range(self.out_shape[0]): # recursive preciction
            out = inputs
            #TODO: change the slice parameter to accommodate for all input data
            slice = 1 # change to 4 for the 15 min sampling rate data (will remove later)
            # inerate through layers
            for layer in range(self.num_layers):
                out, _,_ = self.LSTM_net[layer](out)
                out = self.norm[layer](out)
            out = self.reshape_layer(out)
            # slice the input and append the predicted step 
            inputs = self.concat_timestep([inputs[:,slice:,:], out[:,-slice:,:]])
            out = self.output_layer(out)
            # create the output by appending all predictions
            if t == 0:
                all_out = tf.expand_dims(out[:,-1],2)
            else:
                all_out = tf.concat([all_out, tf.expand_dims(out[:,-1,:],2)], axis=-1) 
        return all_out


# Dense temporal convolutional network
# requires number of block and layers within a block to initialize
# can change growth rate, squeeze factor, kernel sizes, strides, dropout and norm parameters
# requires inputs to run
# can have an arbitrary number of streams determined by the size of the kernel array
# outputs the raw network output in size [Batches, Timesteps, Features]
# requires a MIMO or a recursive wrapper to output predictions
class DenseTCN(tf.keras.layers.Layer):
    def __init__(self, num_blocks, num_layers_per_block, in_shape, growth_rate=12, squeeze_factor=0.5, use_dropout=False, dropout_rate=0.0, use_norm=False, kernel_sizes=[3, 5]):
        super(DenseTCN, self).__init__()
        
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.in_shape = in_shape
        self.growth_rate = growth_rate
        # bottleneck features are the 1x1 conv size for each layer
        self.bottleneck_features = 4*growth_rate
        # squeeze factor is the percentage of features passed from one block to the next
        self.squeeze_factor = squeeze_factor
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.kernel_sizes = kernel_sizes
        self.use_norm = use_norm
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
    def build_block(self):
        self.blocks = []
        for block in range(self.num_blocks):
            all_streams = []
            for kernel_size in self.kernel_sizes:
                all_streams.append(self.build_stream(kernel_size))
            self.blocks.append(all_streams)
            # calculate the number of features in the squeeze layer
            # shape of input to the block + growth rate for all layers in the block
            if block == 0:
                num_features = int(self.squeeze_factor*(self.in_shape[1]+self.num_layers_per_block*self.growth_rate))
            else: 
                num_features = int(self.squeeze_factor*(num_features + self.num_layers_per_block*self.growth_rate))
            # create stack of squeeze layers
            self.squeeze.append(self.cnn_layer(num_features, kernel_size=1, dilation_rate=1))
        
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
            out = tf.nn.dropout(self.dropout_rate)
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
        return tf.concat([out, inputs], 2)


    def call(self, inputs):
        out = inputs
        # call the initialize function
        if not self.already_built:
            self.build_block()

        # iterate through blocks
        for block in range(self.num_blocks):
            out = self.run_block(block, out)
            # apply squeeze after each block except last
            if block < (self.num_blocks-1):
                out = self.squeeze[block](out)
        return out
        
        
# temporal convolutional network
# this implementation includes possibility to use residual blocks or dense blocks 
# TODO: possibly separate the 2 or remove the residual block
# requires number of layers and units, output shape, dropout info to initialize
# in addition may change growth rate, kernel size and strides parameters in init
# requires inputs to call
# outputs prediction in the shape of target
class TCN(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, out_shape, dropout_rate, use_dropout, growth_rate=12, kernel_size=3, strides=1):
        super(TCN, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.kernel_size = kernel_size
        self.strides = strides
        self.out_shape = out_shape
        self.growth_rate = growth_rate
        self.squeeze_factor = 4*growth_rate
        self.reshape_to_out = tf.keras.layers.Reshape((self.out_shape[0], self.out_shape[1]))
        # initialize the network layer stacks
        self.CNN_3 = []
        self.CNN1x1_3 = []
        self.CNN_5 = []
        self.CNN1x1_5 = []
        self.concat_outputs = tf.keras.layers.Concatenate(axis=2)
        
        # for reshaping the input layer in residual case
        self.conv1x1_input = tf.keras.layers.Conv1D(filters=self.num_units,
                                                   kernel_size=1,
                                                   strides=1,
                                                   data_format='channels_last',
                                                   padding='causal')
        # to reshape the number of features in the output layer according to the target
        self.conv1D_output = tf.keras.layers.Conv1D(filters=out_shape[0],
                                                    kernel_size=1,
                                                    strides=self.strides,
                                                    data_format='channels_last',
                                                    padding='causal')
        # to reshape the num features in the output, redundant with above 1x1 conv layer
        self.output_layer = tf.keras.layers.Dense(units=out_shape[0])
        
        # create a network stack
        for layer in range(self.num_layers):
            # 1x1 conv bottleneck layer for one stream of data in dense block
            one_1x1_layer = tf.keras.layers.Conv1D(self.num_units,
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='causal')
            # conv layer with kernel 3 for one stream of data in dense block
            one_CNN_layer = tf.keras.layers.Conv1D(self.num_units,
                                                    kernel_size=self.kernel_size,
                                                    strides=self.strides,
                                                    data_format='channels_last',
                                                    padding='causal',
                                                    dilation_rate=2**layer) # increase dilation rate each layer (1,2,4,8...)
                                                                            # do layer mod 6 so that the dilation doesn't go above 2**5
            # 1x1 bottleneck for the second stream of data in dense
            one_1x1_layer = tf.keras.layers.Conv1D(self.num_units,
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='causal')
            # conv layer for the second stream of data in dense
            one_CNN_layer = tf.keras.layers.Conv1D(self.num_units,
                                                    kernel_size=5,
                                                    strides=self.strides,
                                                    data_format='channels_last',
                                                    padding='causal',
                                                    dilation_rate=2**layer) # increase dilation rate each layer (1,2,4,8...)
                                                                            # do layer mod 6 so that the dilation doesn't go above 2**5

            self.CNN1x1_3.append(one_1x1_layer)
            self.CNN_3.append(one_CNN_layer)
            self.CNN1x1_5.append(one_1x1_layer)
            self.CNN_5.append(one_CNN_layer)

    # create one residual block with conv, dropout and res connection
    def residual_block(self, inputs, layer, isTraining):
        out = inputs
        out = self.CNN_3[layer](out)
        # check if training
        out = tf.keras.activations.relu(out)
        if self.use_dropout: # and isTraining
            out = tf.nn.dropout(out, self.dropout_rate)
        res_out = tf.keras.layers.add([out, inputs])
        return res_out

    # dense block with 2 streams of data with different kernel sizes for feature selection
    # that get concatenated
    # TODO: add recurrent batch norm
    def dense_block(self, inputs, layer):
        out = inputs
        out_3 = self.CNN1x1_3[layer](inputs)
        out_3 = self.CNN_3[layer](out_3)
        out_5 = self.CNN1x1_5[layer](inputs)
        out_5 = self.CNN_5[layer](out_5)
        out = self.concat_outputs([out_3, out_5])
        out = tf.keras.activations.relu(out)
        if self.use_dropout: # and isTraining
            out = tf.nn.dropout(out, self.dropout_rate)
        out = self.concat_outputs([inputs, out])
        return out

    def call(self, inputs):
        out = inputs
        isTraining = tf.keras.backend.learning_phase()
        # 1x1 convolution to create an appropriate feature size 
        out = self.conv1x1_input(out) # only needed for residual 
        # iterate through either dense or residual blocks 
        #TODO: as above, either separate or remove the residual block 
        for layer in range(self.num_layers):
            out = self.dense_block(out, layer)
            #out = self.residual_block(out, layer, isTraining)
        # change feature size to suit the output
        out = self.conv1D_output(out)
        # select the last timestep as the prediction
        out = tf.expand_dims(out[:,-1,:],2)
        return out

# basic LSTM encoder-decoder model 
# requires number of layers and units, output shape to initialize
# additionally takes parameter that decides the use of layer normalization
#TODO: add dropout
# reuqires inputs in shape [encoder_inputs, decoder_inputs] to call
# outputs the prediction in shape of target
class seq2seq(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, out_shape, use_norm=False):
        super(seq2seq, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.out_shape = out_shape
        self.use_norm = use_norm
        # concatenate predictions in the output
        self.concat_timestep = tf.keras.layers.Concatenate(axis=1)
        # currently unnecessary
        self.reshape_to_1ts = tf.keras.layers.Reshape((1, self.out_shape[1]))
        # initialize and append encoder network stack
        self.LSTM = []
        for layer in range(self.num_layers):
            LSTM_layer = tf.keras.layers.LSTM(self.num_units,
                                               activation=tf.nn.tanh,
                                               return_sequences=True,
                                               return_state=True,
                                               recurrent_initializer='glorot_uniform')
            self.LSTM.append(LSTM_layer)
        # initialize and append decoder network stack
        #TODO: maybe nicer to remove this and use a separate class for multilayer LSTM
        self.LSTM_dec = []
        for layer in range(self.num_layers):
            LSTM_layer = tf.keras.layers.LSTM(self.num_units,
                                               activation=tf.nn.tanh,
                                               return_sequences=True,
                                               return_state=True,
                                               recurrent_initializer='glorot_uniform')
            self.LSTM_dec.append(LSTM_layer)

        self.decoder_wrap = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.out_shape[1]))
    # define a multilayer LSTM function used for encoder and decoder
    # copy paste from multilayer LSTM class
    def multiLSTM(self, inputs, initial_states=None, decoder=False):
        # initialize states as zeros in the encoder
        if initial_states is None: 
            initial_states = []
            for layer in range(self.num_layers):
                state_h = tf.zeros([tf.shape(inputs)[0], self.num_units])
                state_c = tf.zeros([tf.shape(inputs)[0], self.num_units])
                initial_states.append([state_h, state_c])

        all_out = []  # again, we are working with lists, so we might just as well do the same for the outputs
        states = []
        out = inputs  # just a simple trick to prevent usage of one if_else loop for the first layer
        for layer in range(self.num_layers):
            #TODO: this is in order to call a different stack for encoder and decoder, very ugly, need fixing
            if not decoder:
                out, state_h, state_c = self.LSTM[layer](out, initial_state=initial_states[layer])
            else: 
                out, state_h, state_c = self.LSTM_dec[layer](out, initial_state=initial_states[layer])
            if self.use_norm:
                out = self.norm[layer](out)
            # return all states and outputs
            all_out.append(out)
            states.append([state_h, state_c])
        return all_out, states

    # define encoder function which outputs currenlty unnecessary outputs
    # and cits states as context vector for decoder 
    # uses multilayer LSTM function
    def encoder(self, inputs):
        out, context_vector = self.multiLSTM(inputs, initial_states=None)
        return out, context_vector

    # define the decoder which recursively iterates through prediction steps
    # uses multilayer LSTM function
    #TODO: maybe implement teacher forcing (paper mentions it performs better without)
    def decoder(self, context_vector, decoder_inputs, encoder_outputs):
        dec_states = context_vector
        for t in range(self.out_shape[0]):
            # if first timestep use the target, otherwise use the previous prediction step
            if t == 0:
                dec_inputs = decoder_inputs
            else:
                dec_inputs = dec_out
            dec_outputs, dec_states = self.multiLSTM(dec_inputs, initial_states=dec_states, decoder=True)
            # get last layer's output 
            dec_out = dec_outputs[-1]
            # dense layer to get appropriate output shape
            dec_out = self.decoder_wrap(dec_out)
            # get last timestep as prediction
            dec_out = tf.expand_dims(dec_out[:,-1], 2)
            #dec_out = self.reshape_to_1ts(dec_out)
            # concatenate prediction steps
            if t == 0:
                decoder_output = dec_out
            else:
                decoder_output = self.concat_timestep([decoder_output, dec_out])
        return decoder_output

    # run encoder-decoder
    def call(self, inputs):
        # get encoder and decoder inputs 
        encoder_inputs = inputs[0]
        decoder_inputs = inputs[1]
        enc_out, context_vector = self.encoder(encoder_inputs)
        out = self.decoder(context_vector, decoder_inputs, enc_out)
        return out

