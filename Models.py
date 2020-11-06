import tensorflow as tf
#ToDo: WE gotta change shit to keyword arguments for sanity's skae... jesus fuck
def forecaster_model(encoder_block, decoder_block,
                     history_shape, support_shape, output_shape,
                     random_degree=0.6,
                     use_teacher=True, #dont be stupid and dont
                     ):

    # shape shortcuts for readability
    history_steps = history_shape[-2]
    history_dims = history_shape[-1]
    support_steps = support_shape[-2]
    support_dims = support_shape[-1]
    out_steps = output_shape[-2]
    print('num out steps', out_steps)
    out_dims = output_shape[-1]


    if history_dims != out_dims:
        print('encountered different dimensionality between the historical signal, (', history_dims, ') and the desired output (', out_dims, ')')

    support_input = tf.keras.layers.Input(shape=(support_steps, support_dims), name='support_input')
    print(support_input)
    history_input = tf.keras.layers.Input(shape=(history_steps, history_dims), name='history_input')
    print(history_input)

    encoder_features = encoder_block(support_input)
    # print(encoder_features)
    # If we are training and using teacher forcing, do one step faaaaaaast
    # else do recurrent decoding
    print('removing 0th step to prevent overlap between history and teacher!!')

    forecast = decoder_block(history_input, attention_value=encoder_features, timesteps=out_steps)

    print('final forecast', forecast)
    return tf.keras.Model([support_input, history_input], forecast)


def S2S_model(encoder_block, decoder_block,
                          input_shape, output_shape,
                          use_teacher=True):

    # define some hepful variables
    in_tsteps = input_shape[-2]
    in_dims = input_shape[-1]
    out_tsteps = output_shape[-2]
    out_dims = output_shape[-1]

    # all the inputs we need to define
    encoder_inputs = tf.keras.layers.Input(shape=(in_tsteps, in_dims), name='encoder_inputs')
    # we need to define this outside, because at inference time we will provide the 0th innput of the decoder, as it is a value thats already known
    # at inference time, the teacher would be concat([1,out_dims], nan*[rest, out_dims]
    encoder_outputs, encoder_states= encoder_block(encoder_inputs)
    teacher = tf.keras.layers.Input(shape=(out_tsteps, out_dims), name='teacher_inputs')
    if use_teacher:
        forecast = decoder_block(tf.expand_dims(teacher[:, 0, :], axis=1),
                                 teacher=teacher[:,1:,:],
                                 decoder_init_state=encoder_states,
                                 attention_value=encoder_outputs,
                                 timesteps=out_tsteps)
    else:
        forecast = decoder_block(tf.expand_dims(teacher[:,0,:], axis=1),
                                 decoder_init_state=encoder_states,
                                 attention_value=encoder_outputs,
                                 timesteps=out_tsteps)

    return tf.keras.Model([encoder_inputs, teacher], forecast)


def mimo_model(function_block, input_shape, output_shape, projection_block=None, mode='project', downsample_input=False, downsampling_rate=None, downsample_mode='sample'):

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

        function_block_out = tf.transpose(function_block_out, perm=[0, 2, 1])
        function_block_out = squeeze_time(function_block_out)
        function_block_out = tf.transpose(function_block_out, perm=[0, 2, 1])
    elif mode == 'snip':
        function_block_out_snip = function_block_out[:, -out_tsteps:, :]
    else:
        print('wrong output mode specified, needs to be either snip or project')

    forecast = projection_block(function_block_out)
    return tf.keras.Model(inputs, forecast)



