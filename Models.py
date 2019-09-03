import tensorflow as tf

def encoder_decoder_model(encoder_block, decoder_block, projection_block, use_teacher, input_shape, output_shape):
    # output shape is a list with at least two numbers, output_shape = [...., timesteps_forecast, output_size]
    # same as input_shape = [...., timesteps_input, input_size]
    # encoder model is a block of layers for the encoder
    # decoder model is a block of layers to be used as decoder

    # define some hepful variables
    in_tsteps = input_shape[-2]
    in_dims = input_shape[-1]
    out_tsteps = output_shape[-2]
    out_dims = output_shape[-1]
    istraining = tf.keras.backend.learning_phase()

    # all the inputs we need to define
    encoder_inputs = tf.keras.layers.Input(shape=(in_tsteps, in_dims))
    teacher_inputs = tf.keras.layers.Input(shape=(out_tsteps, out_dims))
    if use_teacher:
        blend_factor_teacher_forcing = tf.keras.layers.Input(shape=(1))

    # Encoding step, check if we have a model, if so then use that. Otherwise don't
    if encoder_block:
        encoder_outputs, encoder_end_state = encoder_block(encoder_inputs)
    else:
        encoder_outputs = encoder_inputs
        encoder_end_state = None

    # now lets do some decoding
    if not decoder_block:
        print('no decoder provided, this is gonna go real bad. expect an error soon')
    else:

        for forecast_timestep in range(out_tsteps): # we do decoding in recurring fashion

            if use_teacher and istraining: # we might wanna use a teacher while were training

                if forecast_timestep > 0: # for every other than the zeroth timestep
                    # we need to grab the last decoder output and the right teacher step
                    # blend those together
                    forecast_for_previous_timestep = forecast_for_timestep
                    forecast_for_previous_timestep = tf.multiply(tf.subtract(1.0, blend_factor_teacher_forcing),
                                                                 forecast_for_previous_timestep)

                    teacher_forecast_timestep = teacher_inputs[:, forecast_timestep, :]
                    teacher_forecast_timestep = tf.multiply(blend_factor_teacher_forcing, teacher_forecast_timestep)
                    teacher_forecast_timestep = tf.expand_dims(teacher_forecast_timestep, axis=1)

                    decoder_input_forecast_timestep = tf.add(teacher_forecast_timestep, forecast_for_previous_timestep)

                else:
                    # just grab the right eacher timestep
                    teacher_forecast_timestep = teacher_inputs[:, forecast_timestep, :]
                    decoder_input_forecast_timestep = teacher_forecast_timestep
            else:
                # if we dont do teacher forcing or we evaluate, we will realistically
                if forecast_timestep > 0:
                    decoder_input_forecast_timestep = decoder_input_forecast_timestep
                    decoder_state_previous_timestep = decoder_state_forecast_timestep
                else:
                    teacher_forecast_timestep = teacher_inputs[:, forecast_timestep, :]
                    teacher_forecast_timestep = tf.expand_dims(teacher_forecast_timestep, axis=1)
                    decoder_input_forecast_timestep = teacher_forecast_timestep
                    decoder_state_previous_timestep = encoder_end_state

            # use the decoder
            decoder_forecast_timestep, decoder_state_forecast_timestep = decoder_block(decoder_input_forecast_timestep,
                                                                                      decoder_state=decoder_state_previous_timestep,
                                                                                     attention_value=encoder_outputs)

            # if the decoder is not handling projection, then we do it here
            if projection_block:
                forecast_for_timestep = projection_block(decoder_forecast_timestep)
            else:
                forecast_for_timestep = decoder_forecast_timestep

            # append the forecast
            if forecast_timestep > 0:
                forecast = tf.concat([forecast, forecast_for_timestep], axis=1)
            else:
                forecast = forecast_for_timestep

    model = tf.keras.Model([encoder_inputs, teacher_inputs, blend_factor_teacher_forcing], forecast)

    return model

def __adjust_temporal_and_feature_dims(signal, out_tsteps, out_dims):
    squeeze_time = tf.keras.layers.Dense(out_tsteps)
    if out_dims > 1:
        squeeze_features = tf.keras.layers.Dense(units=out_dims, activation=tf.keras.layers.Softmax(axis=-1))
    else:
        squeeze_features = tf.keras.layers.Dense(out_dims)

    print(signal)

    signal_transp = tf.transpose(signal, perm=[0, 2, 1])
    time_adjusted_signal_transp = squeeze_time(signal_transp)
    time_adjusted_signal = tf.transpose(time_adjusted_signal_transp, perm=[0, 2, 1])

    adjusted_signal = squeeze_features(time_adjusted_signal)

    return adjusted_signal


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
