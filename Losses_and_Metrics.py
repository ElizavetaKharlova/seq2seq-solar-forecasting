# will contain losses and metrics used to analyse whatever we wanna train
import tensorflow as tf


class losses_and_metrics(object):
    def __init__(self, last_output_dim_size, normalizer_value=1.0, target_as_expected_value=False, forecast_as_expected_value=False):
        self.last_output_dim_size = last_output_dim_size
        self.normalizer_value = normalizer_value
        self.target_as_expected_value = target_as_expected_value
        self.forecast_as_expected_value = forecast_as_expected_value

    def _convert_to_expectedvalue_if_needed(self, signal, is_expected_value):
        if not is_expected_value:
            indices = tf.range(self.last_output_dim_size, dtype=tf.float32)  # (last_output_dim_size)
            expected_value = tf.multiply(signal, indices)  # (batches, timesteps, last_output_dim_size)
            expected_value = tf.reduce_sum(expected_value, axis=-1)
            return expected_value / self.last_output_dim_size
        else:
            return signal

    def nME(self, target, prediction):
        e_target = self._convert_to_expectedvalue_if_needed(target, self.target_as_expected_value)
        e_prediction = self._convert_to_expectedvalue_if_needed(prediction, self.forecast_as_expected_value)
        expected_nME = calculate_E_ME(e_target, e_prediction)/self.normalizer_value
        return expected_nME

    def nRMSE(self, target, prediction):
        e_target = self._convert_to_expectedvalue_if_needed(target, self.target_as_expected_value)
        e_prediction = self._convert_to_expectedvalue_if_needed(prediction, self.forecast_as_expected_value)
        nRMSE = calculate_E_RMSE(e_target, e_prediction)/self.normalizer_value
        return nRMSE

    def MSE(self, target, prediction):
        e_target = self._convert_to_expectedvalue_if_needed(target, self.target_as_expected_value)
        e_prediction = self._convert_to_expectedvalue_if_needed(prediction, self.forecast_as_expected_value)
        expected_MSE = tf.square(e_target - e_prediction)
        return tf.reduce_mean(expected_MSE, axis=-1)
    def KLD(self, target, prediction):
        return calculate_KL_Divergence(target, prediction)

    def CRPS(self, target, prediction):
        CRPS= calculate_CRPS(target, prediction)
        return CRPS

    def EMC(self, target,prediction):
        return calculate_EMC(target, prediction)



def calculate_CRPS(target, prediction):
    CRPS = tf.square(tf.keras.backend.cumsum(prediction, axis=-1) - tf.keras.backend.cumsum(target, axis=-1)) #(batches, timesteps, features)
    CRPS = tf.reduce_sum(CRPS, axis=-1) #(batches, timesteps_)
    CRPS = tf.reduce_mean(CRPS, axis=-1) #(batches)
    return CRPS

def calculate_KL_Divergence(target, prediction):
    KL_D = target * tf.math.log(tf.maximum(target, 1e-15)/tf.maximum(prediction, 1e-15)) #(batches, timesteps, features)
    KL_D = tf.reduce_sum(KL_D, axis=-1)
    KL_D = tf.reduce_mean(KL_D, axis=-1)
    return KL_D


def calculate_EMC(target, prediction):
    kl_d = calculate_KL_Divergence(target, prediction)

    epsilon = 0.01
    condition = tf.math.less_equal(kl_d, epsilon)
    condition = tf.cast(condition, dtype=tf.float16)
    percentage = tf.reduce_mean(condition)
    return percentage

def calculate_E_RMSE(target, prediction):

    expected_nMSE = tf.square(target - prediction) #(batches, timesteps)
    expected_nMSE = tf.reduce_mean(expected_nMSE, axis=-1) #(batches)
    expected_nRMSE = tf.sqrt(expected_nMSE) #(bactehs)
    return expected_nRMSE

def calculate_E_ME(target, prediction):
    # if pdfs, convert to expected values

    expected_nME = tf.abs(target - prediction) #(batches, timesteps)
    expected_nME = tf.reduce_mean(expected_nME, axis=-1) #(batches)
    return expected_nME



def __calculatate_skillscore_baseline(targets, persistent_forecast=None):
    def __calculate_normalized_expected_value(signal):
        indices = tf.range(signal.shape[-1], dtype=tf.float32)  # (last_output_dim_size)
        weighted_signal = tf.multiply(signal, indices)  # (batches, timesteps, last_output_dim_size)
        expected_value = tf.reduce_sum(weighted_signal, axis=-1)
        return expected_value/signal.shape[-1]

    targets = tf.cast(targets, dtype=tf.float32)
    persistent_forecast = tf.cast(persistent_forecast, dtype=tf.float32)
    e_target = __calculate_normalized_expected_value(targets)
    e_persistent_forecast = __calculate_normalized_expected_value(persistent_forecast)

    nRMSEs = calculate_E_RMSE(e_target, e_persistent_forecast) #since the expected value is normnalized already
    nMEs = calculate_E_ME(e_target, e_persistent_forecast) #since the expected value is normnalized already
    CRPS = calculate_CRPS(targets, persistent_forecast)

    return {'nRMSE': tf.reduce_mean(nRMSEs).numpy(),
            'nME': tf.reduce_mean(nMEs).numpy(),
            'CRPS': tf.reduce_mean(CRPS).numpy(),
            }