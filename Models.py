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

    istraining = tf.keras.backend.learning_phase()
    # all the inputs we need to define
    encoder_inputs = tf.keras.layers.Input(shape=(in_tsteps, in_dims))
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


    # now lets do some decoding
    if not decoder_block:
        print('no decoder provided, this is gonna go real bad. expect an error soon')
    else:
        decoder_kwargs = {}
        if decoder_uses_attention:
            decoder_kwargs['attention_value'] = encoder_outputs

        for forecast_timestep in range(out_tsteps): # we do decoding in recurring fashion
            if forecast_timestep == 0:
                # just grab the right eacher timestep
                decoder_input_forecast_timestep = teacher_inputs[:, 0, :]
                decoder_input_forecast_timestep = tf.expand_dims(decoder_input_forecast_timestep, axis=1)
            # for every other than the zeroth timestep, this means there is a forecast for the previous timestep available
            elif use_teacher:
                def __get_teacher_support():
                 # we might wanna use a teacher while were training
                    forecast_for_previous_timestep = tf.multiply(tf.subtract(1.0, blend_factor_teacher_forcing),
                                                                 forecast_for_timestep)
                    teacher_forecast_timestep = tf.multiply(blend_factor_teacher_forcing, teacher_inputs[:, forecast_timestep, :])
                    teacher_forecast_timestep = tf.expand_dims(teacher_forecast_timestep, axis=1)
                    return tf.add(teacher_forecast_timestep, forecast_for_previous_timestep)
                decoder_input_forecast_timestep = tf.keras.backend.in_train_phase(x=__get_teacher_support(), alt=forecast_for_timestep)
            else: #or, maybe we don't
                decoder_input_forecast_timestep = forecast_for_timestep

            if decoder_stateful: #then we wanna find out the state of the decoder and then call it to get a state back
                # This implicitly assumes that we made sure that the encoder states have the right shapes!!
                if forecast_timestep == 0:
                    decoder_kwargs['decoder_state'] = encoder_end_state
                else:
                    # since decoder_state_forecast_timestep is now actually decoder state previous timestep
                    decoder_kwargs['decoder_state'] = decoder_state_forecast_timestep

                decoder_forecast_timestep, decoder_state_forecast_timestep = decoder_block(decoder_input_forecast_timestep, **decoder_kwargs)
            else: #just call it to get the output
                decoder_forecast_timestep = decoder_block(decoder_input_forecast_timestep, **decoder_kwargs)

            # if the decoder is not handling projection, then we do it here
            if projection_block:
                # if the decoder is a list, implying several layers of outputs
                forecast_for_timestep = projection_block(decoder_forecast_timestep)
            else:
                forecast_for_timestep = decoder_forecast_timestep

            # append the forecast
            if forecast_timestep == 0:
                forecast = forecast_for_timestep
            else:
                forecast = tf.concat([forecast, forecast_for_timestep], axis=1)

    if use_teacher:
        return tf.keras.Model([encoder_inputs, teacher_inputs, blend_factor_teacher_forcing], forecast)
    else:
        return tf.keras.Model([encoder_inputs, teacher_inputs], forecast)

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


def mimo_model(function_block, input_shape, output_shape, mode='project'):

    # define some hepful variables
    in_tsteps = input_shape[-2]
    in_dims = input_shape[-1]
    out_tsteps = output_shape[-2]
    out_dims = output_shape[-1]

    inputs = tf.keras.layers.Input(shape=(in_tsteps, in_dims))
    function_block_out = function_block(inputs)

    if mode=='project':
        mimo_output = __adjust_temporal_and_feature_dims(function_block_out, out_tsteps, out_dims)

    elif mode=='snip':
        function_block_out_snip = function_block_out[:,-out_tsteps,:]

        if function_block_out_snip.shape[-1] != out_dims:
            if out_dims > 1:
                squeeze_features = tf.keras.layers.Dense(units=out_dims, activation=tf.keras.layers.Softmax(axis=-1))
            else:
                squeeze_features = tf.keras.layers.Dense(out_dims)

            mimo_output = squeeze_features(function_block_out_snip)
    else:
        print('wrong mode, please select either project or snip')

    return tf.keras.Model(inputs, mimo_output)
