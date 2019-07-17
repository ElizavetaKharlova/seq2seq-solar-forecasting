# will contain losses and metrics used to analyse whatever we wanna train
import tensorflow as tf
# Loss and Metric for relative Mean Error
def calculate_rME(y_true, y_pred, min_value, max_value):
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    max_offset = tf.math.abs(min_value)
    min_offset = tf.math.abs(max_value)
    abs_offset = tf.add(max_offset, min_offset)

    loss = tf.math.add(y_true, -y_pred)  # (batches, steps, 1)
    loss = tf.math.abs(loss)  # to make sure the error is positive
    loss = tf.math.add(min_offset, loss)  # (batches, steps, 1), make the minimum value 0
    loss = tf.math.divide(loss, abs_offset)  # (batches, steps, 1), normalize the errors relative to the largest error
    loss = tf.squeeze(loss, axis=-1)  # (batches, steps)
    loss = tf.math.reduce_mean(loss, axis=-1)  # (bacthes), get the mean error per timestep

    loss = tf.math.multiply(tf.cast(100.0, dtype=tf.float64), loss)  # (batches) convert into percents

    return loss

def loss_rME(max_value, min_value):
    def rME(y_true, y_pred):
        return calculate_rME(y_true, y_pred, min_value, max_value)

    return rME


# same thing, just relative mean squared error
def calculate_rMSE(y_true, y_pred, min_value, max_value):
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    max_offset = tf.math.abs(min_value)
    min_offset = tf.math.abs(max_value)
    abs_offset = tf.add(max_offset, min_offset)

    y_true = tf.math.add(y_true, min_offset)
    y_pred = tf.math.add(y_pred, min_offset)

    y_true = tf.square(y_true)
    y_pred = tf.square(y_pred)

    loss = tf.math.add(y_true, -y_pred)  # (batches, steps, 1)
    loss = tf.math.abs(loss)
    loss = tf.math.divide(loss, tf.square(
        abs_offset))  # (batches, steps, 1), normalize the errors relative to the largest error
    loss = tf.squeeze(loss, axis=-1)  # (batches, steps)
    loss = tf.math.reduce_mean(loss, axis=-1)  # (bacthes), get the mean error per timestep

    loss = tf.math.multiply(tf.cast(100.0, dtype=tf.float64), loss)  # (batches) convert into percents

    return loss

def loss_rMSE(max_value, min_value):
    def rMSE(y_true, y_pred):
        return calculate_rMSE(y_true, y_pred, min_value, max_value)
    return rMSE

# ToDo: add the probailistic losses n shit