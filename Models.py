import tensorflow as tf
#ToDo: WE gotta change shit to keyword arguments for sanity's skae... jesus fuck

def encoder_decoder_model(encoder_block, decoder_block, projection_block,
                          input_shape, output_shape,
                          use_teacher=True,
                          encoder_stateful=True,
                          decoder_stateful=True,
                          self_recurrent_decoder=True,
                          decoder_uses_attention_on=None):
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
    encoder_inputs = tf.keras.layers.Input(shape=(in_tsteps, in_dims), name='inputs_input')
    # we need to define this outside, because at inference time we will provide the 0th innput of the decoder, as it is a value thats already known
    # at inference time, the teacher would be concat([1,out_dims], nan*[rest, out_dims])
    teacher_inputs = tf.keras.layers.Input(shape=(out_tsteps, out_dims), name='teacher_input')

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
        if decoder_uses_attention_on is None:
            pass
        elif decoder_uses_attention_on == 'hidden' or decoder_uses_attention_on =='output' or decoder_uses_attention_on==True:
            decoder_kwargs['attention_value'] = encoder_outputs


        for forecast_timestep in range(out_tsteps): # we do decoding in recurring fashion
            if forecast_timestep == 0:
                # just grab the right eacher timestep
                teacher_support = tf.expand_dims(teacher_inputs[:, 0, :], axis=1)
                decoder_input_forecast = teacher_support
                decoder_timestep_input = decoder_input_forecast
            # for every other than the zeroth timestep, this means there is a forecast for the previous timestep available
            elif use_teacher:
                if self_recurrent_decoder:
                    zero = tf.expand_dims(teacher_inputs[:, 0, :], axis=1)
                    decoder_timestep_input = tf.concat([zero, forecast_step], axis=1)
                else:
                    decoder_timestep_input = forecast_step

                if self_recurrent_decoder:
                    next_teacher_addon = tf.expand_dims(teacher_inputs[:, forecast_timestep, :], axis=1)
                    teacher_support = tf.concat([teacher_support, next_teacher_addon], axis=1)
                else:
                    teacher_support = tf.expand_dims(teacher_inputs[:, forecast_timestep, :], axis=1)


                decoder_timestep_input = tf.keras.backend.in_train_phase(x=teacher_support, alt=decoder_timestep_input)
            else:
                if self_recurrent_decoder:
                    zero = tf.expand_dims(teacher_inputs[:, 0, :], axis=1)
                    decoder_timestep_input = tf.concat([zero, forecast_step], axis=1)
                else:
                    decoder_timestep_input = forecast_step

            if decoder_stateful: #then we wanna find out the state of the decoder and then call it to get a state back
                # This implicitly assumes that we made sure that the encoder states have the right shapes!!
                if forecast_timestep == 0:
                    decoder_kwargs['decoder_init_state'] = encoder_end_state
                else:
                    # since decoder_state_forecast_timestep is now actually decoder state previous timestep
                    decoder_kwargs['decoder_init_state'] = decoder_state_forecast_timestep

            if decoder_uses_attention_on == 'hidden' or decoder_uses_attention_on =='output' or decoder_uses_attention_on==True:
                decoder_kwargs['attention_query'] = decoder_timestep_input
            if decoder_stateful:
                decoder_timestep_forecast, decoder_state_forecast_timestep = decoder_block(decoder_timestep_input, **decoder_kwargs)
            else: #just call it to get the output
                decoder_timestep_forecast = decoder_block(decoder_timestep_input, **decoder_kwargs)

            if projection_block:
                # if the decoder is a list, implying several layers of outputs
                forecast_step = projection_block(decoder_timestep_forecast)

            if forecast_timestep == 0:
                forecast = forecast_step
            elif not self_recurrent_decoder:
                forecast = tf.concat([forecast, forecast_step], axis=1)
            elif self_recurrent_decoder:
                forecast = forecast_step

    if use_teacher:
        return tf.keras.Model([encoder_inputs, teacher_inputs], forecast)
    else:
        return tf.keras.Model([encoder_inputs, teacher_inputs], forecast)


def mimo_model(function_block, input_shape, output_shape, mode='project', downsample_input=False, downsampling_rate=None, downsample_mode='sample'):

    # define some hepful variables
    in_tsteps = input_shape[-2]
    in_dims = input_shape[-1]
    out_tsteps = output_shape[-2]
    out_dims = output_shape[-1]

    inputs = tf.keras.layers.Input(shape=(in_tsteps, in_dims))

    if downsample_input:
        if downsample_mode =='mean':
            downsampling_rate = int(downsampling_rate)

            signal_shape = inputs.shape
            steps_to_aggregate = signal_shape[1] / downsampling_rate
            for downsampled_step in range(int(steps_to_aggregate)):
                start = downsampled_step * downsampling_rate
                end = start + downsampling_rate
                step = tf.reduce_mean(inputs[:, start:end, :], axis=1, keepdims=True)
                if downsampled_step == 0:
                    downsampeld_signal = step
                else:
                    downsampeld_signal = tf.concat([downsampeld_signal, step], axis=1)

            function_block_out = function_block(downsampeld_signal)
        else:

            downsampling_rate = int(downsampling_rate)
            signal_shape = inputs.shape
            steps_to_aggregate = signal_shape[1] / downsampling_rate
            for downsampled_step in range(int(steps_to_aggregate)):
                start = downsampled_step * downsampling_rate
                sample = tf.expand_dims(inputs[:, start, :], axis=1)
                if downsampled_step == 0:
                    downsampeld_signal = sample
                else:
                    downsampeld_signal = tf.concat([downsampeld_signal, sample], axis=1)

            function_block_out = function_block(downsampeld_signal)
    else:
        function_block_out = function_block(inputs)

    if mode=='project':
        squeeze_time = tf.keras.layers.Dense(out_tsteps)
        if out_dims > 1:
            squeeze_features = tf.keras.layers.Dense(units=out_dims, activation=tf.keras.layers.Softmax(axis=-1))
        else:
            squeeze_features = tf.keras.layers.Dense(out_dims)

        function_block_out = tf.transpose(function_block_out, perm=[0, 2, 1])
        function_block_out = squeeze_time(function_block_out)
        function_block_out = tf.transpose(function_block_out, perm=[0, 2, 1])

        function_block_out = squeeze_features(function_block_out)

        mimo_output = function_block_out
        return tf.keras.Model(inputs, mimo_output)

    elif mode=='snip':
        function_block_out_snip = function_block_out[:,-out_tsteps:,:]

        # If we have the wrong output dimensionality, we do a temporally disttributed transform
        if function_block_out_snip.shape[-1] != out_dims:
            if out_dims > 1:
                squeeze_features = tf.keras.layers.Dense(units=out_dims, activation=tf.keras.layers.Softmax(axis=-1))
            else:
                squeeze_features = tf.keras.layers.Dense(out_dims)
            squeeze_features = tf.keras.layers.TimeDistributed(squeeze_features)
            mimo_output = squeeze_features(function_block_out_snip)
            return tf.keras.Model(inputs, mimo_output)
        # else we just keep it at that
        else:
            mimo_output = function_block_out_snip
            return tf.keras.Model(inputs, mimo_output)
    else:
        print('wrong mode, please select either project or snip')



