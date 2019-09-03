# will contain losses and metrics used to analyse whatever we wanna train
import tensorflow as tf

def loss_wrapper(last_output_dimension_size=30, loss_type='nME'):
    if loss_type == 'NME' or loss_type == 'nME' or loss_type == 'nme':
        def loss_nME(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_E_nME(target, prediction, last_output_dimension_size)
        return loss_nME

    elif loss_type == 'NMSE' or loss_type == 'nMSE' or loss_type == 'nmse':
        def loss_nMSE(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_E_nMSE(target, prediction, last_output_dimension_size)
        return loss_nMSE

    elif loss_type == 'KL-D' or loss_type == 'kl-d' or loss_type == 'KL-Divergence':
        def loss_KLD(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_KL_Divergence(target, prediction, last_output_dimension_size)
        return loss_KLD

    elif loss_type == 'tile-to-forecast':
        def loss_designed(target, prediction):
            target = tf.cast(target, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            return calculate_tile_to_pdf_loss(target, prediction, last_output_dimension_size)
        return loss_designed
    else:
        print('loss_type ', loss_type, ' not recognized.')
        print('please elect loss from available options: nME, nMSE, KL-Divergence, tile-to-forecast')

def __calculate_expected_value(signal, last_output_dim_size):
    indices = tf.range(last_output_dim_size, dtype=tf.float32) # (last_output_dim_size)
    weighted_signal = tf.multiply(signal, indices) # (batches, timesteps, last_output_dim_size)
    return tf.reduce_sum(weighted_signal, keepdims=True)

def calculate_E_nMSE(target, prediction, last_output_dim_size):

    # if pdfs, convert to expected values
    if last_output_dim_size > 1:
        target = __calculate_expected_value(target, last_output_dim_size)
        prediction = __calculate_expected_value(prediction, last_output_dim_size)

    expected_nMSE = tf.subtract(target, prediction)
    expected_nMSE = tf.square(expected_nMSE)

    return tf.reduce_mean(expected_nMSE)

def calculate_E_nME(target, prediction, last_output_dim_size):

    # if pdfs, convert to expected values
    if last_output_dim_size > 1:
        target = __calculate_expected_value(target, last_output_dim_size)
        prediction = __calculate_expected_value(prediction, last_output_dim_size)

    expected_nME = tf.subtract(target, prediction)
    expected_nME = tf.abs(expected_nME)

    return tf.reduce_mean(expected_nME)

def calculate_tile_to_pdf_loss(target, prediction, last_output_dim_size):
    absolute_probability_errors = tf.abs(tf.subtract(prediction, target))
    inverse_probability_errors = tf.subtract(1.0, absolute_probability_errors)
    inverse_probability_errors_nonzero = tf.math.maximum(inverse_probability_errors, 1e-8)
    inverse_log_probability_error = - tf.math.log(inverse_probability_errors_nonzero)

    loss = 0.0

    for prediction_tile in range(last_output_dim_size):
        forecast_tile_inverse_log_loss = inverse_log_probability_error[:, :, prediction_tile]

        for target_tile in range(last_output_dim_size):
            tile_distance = target_tile - prediction_tile
            tile_distance_sq = tf.square(tile_distance)
            tile_distance_sq = tf.cast(tile_distance_sq, dtype=tf.float32)

            target_tile_probability = target[:, :, target_tile]

            loss += tile_distance_sq * target_tile_probability * forecast_tile_inverse_log_loss

    loss = tf.reduce_sum(loss)

    return loss

def calculate_KL_Divergence(target, prediction, last_output_dim_size):

    if last_output_dim_size == 1:
        return tf.nan

    else:

        # KL_divergence = sum_over_bands( q(band) * ln(q(band)/p(band)))
        KL_divergence = tf.divide(tf.math.maximum(prediction, 1e-8), tf.math.maximum(target, 1e-8))

        KL_divergence = tf.math.log(KL_divergence)
        KL_divergence = tf.multiply(prediction, KL_divergence)
        KL_divergence = tf.reduce_sum(KL_divergence, axis=-1)
        KL_divergence = tf.reduce_mean(KL_divergence)
        return KL_divergence
