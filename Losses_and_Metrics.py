# will contain losses and metrics used to analyse whatever we wanna train
import tensorflow as tf

def loss_wrapper(last_output_dim_size, loss_type='nME', normalizer_value=1.0, target_as_expected_value=False, forecast_as_expected_value=False):
    def convert_to_proper_signal(signal, is_expected_value):
        signal = tf.cast(signal, dtype=tf.float32)
        if not is_expected_value:
            signal = __calculate_expected_value(signal, last_output_dim_size)
        else:
            signal = tf.squeeze(signal, axis=-1)
        return signal

    if loss_type == 'NME' or loss_type == 'nME' or loss_type == 'nme':
        def nME(target, prediction):
            prediction = convert_to_proper_signal(prediction, forecast_as_expected_value)
            target = convert_to_proper_signal(target, target_as_expected_value)
            return calculate_E_nME(target, prediction, normalizer_value)
        return nME

    elif loss_type == 'NRMSE' or loss_type == 'nRMSE' or loss_type == 'nrmse':
        def nRMSE(target, prediction):
            prediction = convert_to_proper_signal(prediction, forecast_as_expected_value)
            target = convert_to_proper_signal(target, target_as_expected_value)
            return calculate_E_nRMSE(target, prediction, normalizer_value)
        return nRMSE

    elif loss_type == 'MSE' or loss_type == 'mse':
        def MSE(target, prediction):
            prediction = convert_to_proper_signal(prediction, forecast_as_expected_value)
            target = convert_to_proper_signal(target, target_as_expected_value)
            return calculate_E_MSE(target, prediction)
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
    elif loss_type=='EMC':
        def EMC(target,prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_EMC(target, prediction)
        return EMC
    elif loss_type=='symmKLD':
        def symmKLD(target,prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_symmKLD(target, prediction)
        return symmKLD
    else:
        print('loss_type ', loss_type, ' not recognized.')
        print('please elect loss from available options: nME, nMSE, KL-Divergence, tile-to-forecast')

def __calculate_expected_value(signal, last_output_dim_size):
    indices = tf.range(last_output_dim_size, dtype=tf.float32) # (last_output_dim_size)
    weighted_signal = tf.multiply(signal, indices) # (batches, timesteps, last_output_dim_size)
    expected_value = tf.reduce_sum(weighted_signal, axis=-1)
    return expected_value / last_output_dim_size

def __pdf_to_cdf(pdf, last_output_dim_size):
    for tile in range(last_output_dim_size):
        if tile == 0:
            cdf = tf.expand_dims(pdf[:,:, tile], axis=-1)
        else:
            new = cdf[:,:,-1:] + tf.expand_dims(pdf[:,:,tile], axis=-1)
            cdf = tf.concat([cdf, new], axis=-1)
    return cdf



def calculate_KS(target, prediction, last_output_dim_size):

    pred_cdf = __pdf_to_cdf(prediction, last_output_dim_size)
    targ_cdf = __pdf_to_cdf(target, last_output_dim_size)
    KS_integral = tf.abs(pred_cdf - targ_cdf)
    KS_integral = tf.reduce_sum(KS_integral, axis=-1)
    # KS_integral = tf.square(KS_integral)
    return tf.reduce_mean(KS_integral)

def calculate_CRPS(target, prediction, last_output_dim_size):
    forecast_cdf = __pdf_to_cdf(prediction, last_output_dim_size)
    target_cdf = __pdf_to_cdf(target, last_output_dim_size)

    CRPS = tf.square(target_cdf - forecast_cdf) #(batches, timesteps, features)
    CRPS = tf.reduce_sum(CRPS, axis=-1) #(batches, timesteps_)
    CRPS = tf.reduce_mean(CRPS, axis=-1) #(batches)
    return CRPS

def calculate_KL_Divergence(target, prediction, last_output_dim_size):
    target = tf.maximum(target, 1e-15)
    prediction = tf.maximum(prediction, 1e-15)
    KL_D = -target * tf.math.log(prediction/target) #(batches, timesteps, features)
    KL_D = tf.reduce_sum(KL_D, axis=-1)
    KL_D = tf.reduce_mean(KL_D)
    return KL_D

def calculate_symmKLD(target, prediction):

    quotient = tf.maximum(prediction, 1e-15)/tf.maximum(target, 1e-15)
    entropy = tf.math.log(quotient)
    simm_KLD = (prediction-target) * entropy
    simm_KLD = tf.abs(simm_KLD)
    simm_KLD = tf.reduce_sum(simm_KLD, axis=-1)

    simm_KLD = tf.reduce_mean(simm_KLD)
    return simm_KLD

def calculate_EMC(target, prediction):
    kl_d = -target * tf.math.log(tf.maximum(prediction, 1e-15)/tf.maximum(target, 1e-15))
    simm_KLD = tf.reduce_sum(kl_d, axis=-1)
    simm_KLD = tf.reduce_mean(simm_KLD, axis=-1)
    epsilon = 0.1
    condition = tf.math.less_equal(simm_KLD, epsilon)
    condition = tf.cast(condition, dtype=tf.float16)
    percentage = tf.reduce_mean(condition)
    return percentage

def calculate_E_nRMSE(target, prediction, normalizer_value):

    expected_nMSE = tf.square(target - prediction) #(batches, timesteps)
    expected_nMSE = tf.reduce_mean(expected_nMSE, axis=-1) #(batches)
    expected_nRMSE = tf.sqrt(expected_nMSE) #(bactehs)
    return expected_nRMSE / normalizer_value

def calculate_E_nME(target, prediction, normalizer_value):
    # if pdfs, convert to expected values

    expected_nME = tf.abs(target - prediction) #(batches, timesteps)
    expected_nME = tf.reduce_mean(expected_nME, axis=-1) #(batches)
    return expected_nME / normalizer_value

def calculate_E_MSE(target, prediction):
    expected_nMSE = tf.square(target - prediction)
    expected_nMSE = tf.reduce_mean(expected_nMSE, axis=-1)
    expected_nMSE = tf.reduce_mean(expected_nMSE, axis=-1)
    return expected_nMSE

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
    e_target = __calculate_expected_value(targets, 50)
    e_persistent_forecast = __calculate_expected_value(persistent_forecast, 50)

    nRMSEs = calculate_E_nRMSE(e_target, e_persistent_forecast, e_persistent_forecast.shape[-1])
    nMEs = calculate_E_nME(e_target, e_persistent_forecast, e_persistent_forecast.shape[-1])
    CRPS = calculate_CRPS(targets, persistent_forecast, set_targets.shape[-1])

    return {'nRMSE': tf.reduce_mean(nRMSEs).numpy(),
            'nME': tf.reduce_mean(nMEs).numpy(),
            'CRPS': tf.reduce_mean(CRPS).numpy(),
            }