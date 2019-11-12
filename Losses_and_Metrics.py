# will contain losses and metrics used to analyse whatever we wanna train
import tensorflow as tf

def loss_wrapper(last_output_dim_size, loss_type='nME', normalizer_value=1.0):
    if loss_type == 'NME' or loss_type == 'nME' or loss_type == 'nme':
        def nME(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_E_nME(target, prediction, last_output_dim_size, normalizer_value)
        return nME

    elif loss_type == 'NRMSE' or loss_type == 'nRMSE' or loss_type == 'nrmse':
        def nRMSE(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_E_nRMSE(target, prediction, last_output_dim_size, normalizer_value)
        return nRMSE

    elif loss_type == 'MSE' or loss_type == 'mse':
        def MSE(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_E_MSE(target, prediction, last_output_dim_size)
        return MSE

    elif loss_type == 'KL-D' or loss_type == 'kl-d' or loss_type == 'KL-Divergence':
        def KLD(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_KL_Divergence(target, prediction, last_output_dim_size)
        return KLD

    elif loss_type=='CRPS' or loss_type=='Continuous Ranked Probability Score':
        def CRPS(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_CRPS(target, prediction, last_output_dim_size)
        return CRPS
    elif loss_type=='KL_D':
        def KL_D(target,prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_KL_Divergence(target, prediction, last_output_dim_size)
        return KL_D
    #wierdass crap
    elif loss_type=='KS_weighted_KL':
        def KS_weighted_KL(target,prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_KS_weighted_KL(target, prediction, last_output_dim_size)
        return KS_weighted_KL
    elif loss_type == 'tile-to-forecast':
        def designed(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_tile_to_pdf_loss(target, prediction, last_output_dim_size)
        return designed

    #whatever experimental crap
    elif loss_type=='devset':
        def log_KS_test(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_EMD(target, prediction, last_output_dim_size)
        return log_KS_test
    else:
        print('loss_type ', loss_type, ' not recognized.')
        print('please elect loss from available options: nME, nMSE, KL-Divergence, tile-to-forecast')

def __calculate_expected_value(signal, last_output_dim_size):
    indices = tf.range(last_output_dim_size, dtype=tf.float32) # (last_output_dim_size)
    weighted_signal = tf.multiply(signal, indices) # (batches, timesteps, last_output_dim_size)
    expected_value = tf.reduce_sum(weighted_signal, axis=-1, keepdims=True)
    return expected_value

def __pdf_to_cdf(pdf, last_output_dim_size):
    for tile in range(last_output_dim_size):
        if tile == 0:
            cdf = tf.expand_dims(pdf[:,:, tile], axis=-1)
        else:
            new = cdf[:,:,-1:] + tf.expand_dims(pdf[:,:,tile], axis=-1)
            cdf = tf.concat([cdf, new], axis=-1)

    return cdf

def calculate_E_MSE(target, prediction, last_output_dim_size):
    if last_output_dim_size > 1:
        target = __calculate_expected_value(target, last_output_dim_size)
        prediction = __calculate_expected_value(prediction, last_output_dim_size)
    expected_nMSE = tf.subtract(target, prediction)
    expected_nMSE = tf.square(expected_nMSE)

    return tf.reduce_mean(expected_nMSE)

def calculate_EMD(target, prediction, last_output_dim_size):
    EMD_tiles = tf.zeros(shape=[tf.shape(target)[0], tf.shape(target)[1], 1])
    x_entropy = -target*tf.math.log(tf.maximum(prediction, 1e-12)/tf.maximum(target, 1e-12))
    for tile in range(last_output_dim_size):
        EMD_next_tile = EMD_tiles[:,:,tile] + x_entropy[:,:,tile]
        EMD_next_tile = tf.expand_dims(EMD_next_tile, axis=-1)
        EMD_tiles = tf.concat([EMD_tiles, EMD_next_tile], axis=-1)

    #EMD_tiles = tf.square(EMD_tiles)
    EMD = tf.reduce_sum(tf.abs(EMD_tiles), axis=-1)
    #EMD = tf.square(EMD)
    return tf.reduce_mean(EMD)

def calculate_KS_weighted_KL(target, prediction, last_output_dim_size):
    KL_non_weighted = tf.maximum(prediction, 1e-8) / tf.maximum(target, 1e-8)
    KL_non_weighted = -tf.math.log(KL_non_weighted)
    #KL_non_weighted = tf.abs(KL_non_weighted)

    KS_diff = __pdf_to_cdf(prediction, last_output_dim_size) - __pdf_to_cdf(target, last_output_dim_size)
    KS_diff = tf.abs(KS_diff)
    KS_weighted_KL_loss = KS_diff * KL_non_weighted
    # KS_weighted_KL_loss = tf.reduce_sum(KS_weighted_KL_loss, axis=-1)


    # KS_weighted_KL_loss = tf.square(KS_weighted_KL_loss)
    return tf.reduce_mean(KS_weighted_KL_loss, axis=-1)

def calculate_KS(target, prediction, last_output_dim_size):

    # KL_div = -target*tf.math.log(tf.maximum(prediction, 1e-12)/tf.maximum(target, 1e-12)) #KL divergence fromfor target and prediction
    x_entropy_p_q = -target * tf.math.log(tf.maximum(prediction, 1e-12)) #xentropy for target and prediction
    # X_entropy_inverse_prob = -(1.0-target) * tf.math.log(tf.maximum(1.0-prediction, 1e-12))
    x_entropy_q_p = -prediction * tf.math.log(tf.maximum(target, 1e-12))
    # target_error_log =  -target*tf.math.log(tf.maximum(1.0-(target - prediction), 1e-12))
    # prediction_error_log = -prediction * tf.math.log(tf.maximum(1.0 - (prediction - target), 1e-12))

    KL_div = tf.reduce_sum(x_entropy_q_p + x_entropy_p_q, axis=-1)
    KL_div = tf.reduce_sum(KL_div, axis=-1)
    return tf.reduce_mean(KL_div)

def calculate_CRPS(target, prediction, last_output_dim_size):
    forecast_cdf = __pdf_to_cdf(prediction, last_output_dim_size)
    target_cdf = __pdf_to_cdf(target, last_output_dim_size)
    CRPS = tf.subtract(forecast_cdf, target_cdf)
    CRPS = tf.square(CRPS)
    CRPS = tf.reduce_sum(CRPS, axis=-1)
    return tf.reduce_mean(CRPS)

def calculate_E_nRMSE(target, prediction, last_output_dim_size, normalizer_value):
    # if pdfs, convert to expected values
    if last_output_dim_size > 1:
        target = __calculate_expected_value(target, last_output_dim_size)
        prediction = __calculate_expected_value(prediction, last_output_dim_size)
        normalizer_value = last_output_dim_size

    expected_nMSE = tf.subtract(target, prediction)
    expected_nMSE = tf.square(expected_nMSE)
    expected_nMSE = tf.reduce_mean(expected_nMSE)
    expected_nRMSE = tf.sqrt(expected_nMSE)

    return tf.divide(expected_nRMSE, tf.cast(normalizer_value, dtype=tf.float32))

def calculate_E_nME(target, prediction, last_output_dim_size, normalizer_value):
    # if pdfs, convert to expected values
    if last_output_dim_size > 1:
        target = __calculate_expected_value(target, last_output_dim_size)
        prediction = __calculate_expected_value(prediction, last_output_dim_size)
        normalizer_value = last_output_dim_size

    expected_nME = tf.subtract(target, prediction)
    expected_nME = tf.abs(expected_nME)
    expected_nME = tf.reduce_mean(expected_nME)

    return tf.divide(expected_nME, tf.cast(normalizer_value, dtype=tf.float32))

def calculate_tile_to_pdf_loss(target, prediction, last_output_dim_size):
    absolute_probability_errors = tf.subtract(prediction, target)
    absolute_probability_errors = tf.abs(absolute_probability_errors)
    absolute_probability_errors_bounded = tf.math.minimum(absolute_probability_errors, 1.0-1e-8)
    inverse_probability_errors = tf.subtract(1.0, absolute_probability_errors_bounded)
    # inverse_log_probability_error = tf.square(absolute_probability_errors_bounded)
    inverse_log_probability_error = -tf.math.log(inverse_probability_errors)

    loss = 0.0
    num_entries = 0.0
    for prediction_tile in range(last_output_dim_size):
        tile_loss = 0.0
        for target_tile in range(last_output_dim_size):
            tile_distance_weight = tf.cast(target_tile - prediction_tile, dtype=tf.float32)
            tile_distance_weight = tf.abs(tile_distance_weight)

            tile_tile_loss = tf.multiply(target[:, :, target_tile],
                                         inverse_log_probability_error[:, :, prediction_tile])
            tile_tile_loss = tf.multiply(tile_distance_weight, tile_tile_loss)
            num_entries += 1.0
            tile_loss += tile_tile_loss

        loss += tile_loss

    loss = tf.divide(loss, num_entries)

    loss = tf.reduce_sum(loss)
    print(loss)
    return loss

def calculate_KL_Divergence(target, prediction, last_output_dim_size):
    KL_D = -target * tf.math.log(tf.maximum(prediction, 1e-15)/tf.maximum(target, 1e-15))
    KL_D = tf.reduce_sum(KL_D, axis=-1)

    return tf.reduce_mean(KL_D)

def __calculatate_skillscore_baseline(set_targets, sample_spacing_in_mins=5, normalizer_value=1.0, persistent_forecast=None):

    if persistent_forecast is None:
        persistency_offset = (24 * 60) / sample_spacing_in_mins
        persistency_offset = int(persistency_offset)
        targets = set_targets[persistency_offset:, :, :]
        persistent_forecast = set_targets[:-persistency_offset, :,:]
    else:
        targets = set_targets

    targets = tf.cast(targets, dtype=tf.float32)
    persistent_forecast = tf.cast(persistent_forecast, dtype=tf.float32)
    nRMSEs = calculate_E_nRMSE(targets, persistent_forecast, set_targets.shape[-1], normalizer_value)
    nMEs = calculate_E_nME(targets, persistent_forecast, set_targets.shape[-1], normalizer_value)
    CRPS = calculate_CRPS(targets, persistent_forecast, set_targets.shape[-1])

    return {'nRMSE': tf.reduce_mean(nRMSEs).numpy(),
            'nME': tf.reduce_mean(nMEs).numpy(),
            'CRPS': tf.reduce_mean(CRPS).numpy(),
            }