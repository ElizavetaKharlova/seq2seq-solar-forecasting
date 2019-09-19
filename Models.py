import tensorflow as tf
#ToDo: WE gotta change shit to keyword arguments for sanity's skae... jesus fuck
def encoder_decoder_model(encoder_block, decoder_block, projection_block,
                          input_shape, output_shape,
                          use_teacher=True,
                          encoder_stateful=True,
                          decoder_stateful=True,
                          decoder_uses_attention=True):
    # output shape is a list with at least two numbers, output_shape = [...., timesteps_forecast, output_size]
    # same as input_shape = [...., timesteps_input, input_size]
    # encoder model is a block of layers for the encoder, it gives out the encoder output that might be used for attention
        # The encoder can be stateful and return a state
    # decoder model is a block of layers to be used as decoder
        # it can use attention, which it would ATM do on the encoder output as values and the decoder input as query
        # it can be stateful, meaning it would return a state and use this for the next turn

    # define some hepful variables
    in_tsteps = input_shape[-2]
    in_dims = input_shape[-1]
    out_tsteps = output_shape[-2]
    out_dims = output_shape[-1]

    # all the inputs we need to define
    encoder_inputs = tf.keras.layers.Input(shape=(in_tsteps, in_dims))
    print('encoder input', encoder_inputs)
    # we need to define this outside, because at inference time we will provide the 0th innput of the decoder, as it is a value thats already known
    # at inference time, the teacher would be concat([1,out_dims], nan*[rest, out_dims])
    teacher_inputs = tf.keras.layers.Input(shape=(out_tsteps, out_dims))
    if use_teacher:
        blend_factor_teacher_forcing = tf.keras.layers.Input(shape=(1))

    # Encoding step, check if we have a model, if so then use that. Otherwise don't
    if encoder_block and encoder_stateful:
        encoder_outputs, encoder_end_state = encoder_block(encoder_inputs)
    elif encoder_block and not encoder_stateful:

        encoder_outputs = encoder_block(encoder_inputs)
        encoder_end_state = None
    else:
        encoder_outputs = encoder_inputs
        encoder_end_state = None
    print('encoder outputs', encoder_outputs)

    # now lets do some decoding
    if not decoder_block:
        print('no decoder provided, this is gonna go real bad. expect an error soon')
    else:
        decoder_kwargs = {}
        if decoder_uses_attention:
            decoder_kwargs['attention_value'] = encoder_outputs
        zeroth_step_input = teacher_inputs[:, 0, :]
        zeroth_step_input = tf.expand_dims(zeroth_step_input, axis=1)
        for forecast_timestep in range(out_tsteps): # we do decoding in recurring fashion
            if forecast_timestep == 0:
                # just grab the right eacher timestep
                decoder_input_forecast_timestep = zeroth_step_input
            # for every other than the zeroth timestep, this means there is a forecast for the previous timestep available
            elif use_teacher:
                def __get_teacher_support():
                 # we might wanna use a teacher while were training
                    teacher_forecast_timestep = tf.expand_dims(
                                                    tf.multiply(blend_factor_teacher_forcing,
                                                                teacher_inputs[:, forecast_timestep, :]),
                                                    axis=1)
                    return tf.add(teacher_forecast_timestep, tf.multiply(tf.subtract(1.0, blend_factor_teacher_forcing),decoder_forecast_timestep))
                decoder_input_forecast_timestep = tf.keras.backend.in_train_phase(x=__get_teacher_support(), alt=decoder_forecast_timestep)
            else: #or, maybe we don't
                decoder_input_forecast_timestep = projection_block(decoder_forecast_timestep)

            if decoder_stateful: #then we wanna find out the state of the decoder and then call it to get a state back
                # This implicitly assumes that we made sure that the encoder states have the right shapes!!
                if forecast_timestep == 0:
                    decoder_kwargs['decoder_state'] = encoder_end_state
                else:
                    # since decoder_state_forecast_timestep is now actually decoder state previous timestep
                    decoder_kwargs['decoder_state'] = decoder_state_forecast_timestep
            if decoder_uses_attention:
                if forecast_timestep == 0:
                    decoder_kwargs['attention_query'] = zeroth_step_input
                else:
                    decoder_kwargs['attention_query'] = tf.concat([zeroth_step_input, forecast], axis=1)

            if decoder_stateful:
                decoder_forecast_timestep, decoder_state_forecast_timestep = decoder_block(decoder_input_forecast_timestep, **decoder_kwargs)
            else: #just call it to get the output
                decoder_forecast_timestep = decoder_block(decoder_input_forecast_timestep, **decoder_kwargs)

            if projection_block:
                # if the decoder is a list, implying several layers of outputs
                decoder_forecast_timestep = projection_block(decoder_forecast_timestep)

            # append the forecast
            if forecast_timestep == 0:
                forecast = decoder_forecast_timestep
            else:
                forecast = tf.concat([forecast, decoder_forecast_timestep], axis=1)





    if use_teacher:
        return tf.keras.Model([encoder_inputs, teacher_inputs, blend_factor_teacher_forcing], forecast)
    else:
        return tf.keras.Model([encoder_inputs, teacher_inputs], forecast)


def mimo_model(function_block, input_shape, output_shape, mode='project', downsample_input=False, downsampling_rate=None, downsample_mode='sample'):

    # define some hepful variables
    in_tsteps = input_shape[-2]
    in_dims = input_shape[-1]
    out_tsteps = output_shape[-2]
    out_dims = output_shape[-1]

    inputs = tf.keras.layers.Input(shape=(in_tsteps, in_dims))
    print('inputs', inputs)
    if downsample_input:
        if downsample_mode =='mean':
            function_block_out = function_block(__downsample_signal_via_mean(inputs, downsampling_rate))
        else:
            function_block_out = function_block(__downsample_signal_via_sample(inputs, downsampling_rate))
        print('downsampled inputs', inputs)
    else:
        function_block_out = function_block(inputs)

    print('block out', print(function_block_out))
    if mode=='project':
        mimo_output = __adjust_temporal_and_feature_dims(function_block_out, out_tsteps, out_dims)

    elif mode=='snip':
        function_block_out_snip = function_block_out[:,-out_tsteps:,:]
        print('block out after snip', function_block_out_snip)

        # If we have the wrong output dimensionality, we do a temporally disttributed transform
        if function_block_out_snip.shape[-1] != out_dims:
            if out_dims > 1:
                squeeze_features = tf.keras.layers.Dense(units=out_dims, activation=tf.keras.layers.Softmax(axis=-1))
            else:
                squeeze_features = tf.keras.layers.Dense(out_dims)
            squeeze_features = tf.keras.layers.TimeDistributed(squeeze_features)
            mimo_output = squeeze_features(function_block_out_snip)
        # else we just keep it at that
        else:
            mimo_output = function_block_out_snip
    else:
        print('wrong mode, please select either project or snip')

    print('model out', mimo_output)

    return tf.keras.Model(inputs, mimo_output)

def __adjust_temporal_and_feature_dims(signal, out_tsteps, out_dims):
    squeeze_time = tf.keras.layers.Dense(out_tsteps)
    if out_dims > 1:
        squeeze_features = tf.keras.layers.Dense(units=out_dims, activation=tf.keras.layers.Softmax(axis=-1))
    else:
        squeeze_features = tf.keras.layers.Dense(out_dims)

    signal = tf.transpose(signal, perm=[0, 2, 1])
    signal = squeeze_time(signal)
    signal = tf.transpose(signal, perm=[0, 2, 1])

    signal = squeeze_features(signal)

    return signal

def __downsample_signal_via_mean(signal, factor):
    if int(factor) != factor:
        print('cannot downsample by a non integer')
    factor = int(factor)

    signal_shape = signal.shape
    steps_to_aggregate = signal_shape[1]/factor
    for downsampled_step in range(int(steps_to_aggregate)):
        start = downsampled_step*factor
        end = start + factor

        if downsampled_step == 0:
            downsampeld_signal = tf.reduce_mean(signal[:,start:end,:], axis=1, keepdims=True)
        else:
            downsampeld_signal = tf.concat([downsampeld_signal, tf.reduce_mean(signal[:,start:end,:], axis=1, keepdims=True)],
                                   axis=1)

    return downsampeld_signal

def __downsample_signal_via_sample(signal, factor):
    if int(factor) != factor:
        print('cannot downsample by a non integer')
    factor = int(factor)

    signal_shape = signal.shape
    steps_to_aggregate = signal_shape[1]/factor
    for downsampled_step in range(int(steps_to_aggregate)):
        start = downsampled_step*factor
        end = start + factor

        if downsampled_step == 0:
            downsampeld_signal = tf.expand_dims(signal[:,start,:], axis=1)
        else:
            downsampeld_signal = tf.concat([downsampeld_signal, tf.expand_dims(signal[:,start,:], axis=1)],
                                   axis=1)

    return downsampeld_signal

