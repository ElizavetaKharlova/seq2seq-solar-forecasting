# will contain losses and metrics used to analyse whatever we wanna train
import tensorflow as tf
# Loss and Metric for relative Mean Error for expected value forecasts

# this calculates the normed mean error,  meaning it is with regards to the biggest and smallest values within the set
def calculate_expected_normME(y_true, y_pred, min_value, max_value):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    min_value = tf.cast(min_value, dtype=tf.float32)
    max_value = tf.cast(max_value, dtype=tf.float32)


    abs_offset = tf.subtract(min_value, max_value)

    loss = tf.math.subtract(y_true, y_pred)  # (batches, steps, 1)
    loss = tf.math.subtract(loss, min_value)  # (batches, steps, 1), make the minimum value 0

    loss = tf.math.abs(loss)  # to make sure the error is positive
    loss = tf.math.divide(loss, abs_offset)  # (batches, steps, 1), normalize the errors relative to the largest error
    loss = tf.math.reduce_mean(loss, axis=-1)  # (bacthes), get the mean error per timestep

    loss = tf.math.multiply(tf.cast(100.0, dtype=tf.float32), loss)  # (batches) convert into percents

    return loss
def loss_expected_normME(max_value, min_value):
    def normME(y_true, y_pred):
        return calculate_expected_normME(y_true, y_pred, min_value, max_value)
    return normME

# this calculates the relative mean error, absolute meaning it is with regards to the target
def calculate_expected_rME(target, prediction):
    target = tf.cast(target, dtype=tf.float32)
    prediction = tf.cast(prediction, dtype=tf.float32)

    target = tf.where(tf.equal(target, 0.0),
                      x=1e-7,
                      y=target)

    relative = tf.math.divide(prediction, target)
    relative_error = tf.math.subtract(relative, 1)
    loss = tf.math.reduce_mean(relative_error)  # (bacthes), get the mean error per timestep

    loss = tf.math.multiply(tf.cast(100.0, dtype=tf.float32), loss)  # (batches) convert into percents

    return loss
def loss_expected_rME():
    def rME(y_true, y_pred):
        return calculate_expected_rME(y_true, y_pred)
    return rME

# this calculates the relative mean expected error, absolute meaning it is with regards to the target for a PDF
def calculate_pdf_rME(target, prediction):

    output_tiles = tf.shape(target)
    output_tiles = output_tiles[-1]
    indices = tf.range(output_tiles)
    indices = tf.cast(indices, dtype=tf.float32)

    ev_target = tf.multiply(target, indices)
    ev_target = tf.reduce_sum(ev_target, axis=-1)

    ev_prediction = tf.multiply(prediction, indices)
    ev_prediction = tf.reduce_sum(ev_prediction, axis=-1)

    loss = calculate_expected_rME(ev_target, ev_prediction)

    return loss
def loss_pdf_rME():
    def pdf_rME(target, prediction):
        return calculate_pdf_rME(target, prediction)
    return pdf_rME

# same thing, just relative mean squared error
def calculate_expected_normMSE(y_true, y_pred, min_value, max_value):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    min_value = tf.cast(min_value, dtype=tf.float32)
    max_value = tf.cast(max_value, dtype=tf.float32)

    abs_offset = tf.subtract(min_value, max_value)

    y_true = tf.math.subtract(y_true, min_value) # make sure they start at 0
    y_pred = tf.math.subtract(y_pred, min_value)

    y_true = tf.square(y_true)
    y_pred = tf.square(y_pred)

    loss = tf.math.subtract(y_true, y_pred)  # (batches, steps, 1)
    loss = tf.math.abs(loss)
    loss = tf.math.divide(loss, tf.square(
        abs_offset))  # (batches, steps, 1), normalize the errors relative to the largest error
    loss = tf.math.reduce_mean(loss)  # (bacthes), get the mean error per timestep

    loss = tf.math.multiply(tf.cast(100.0, dtype=tf.float32), loss)  # (batches) convert into percents

    return loss
def loss_expected_normMSE(min_value, max_value):
    def normMSE(y_true, y_pred):
        return calculate_expected_normMSE(y_true, y_pred, min_value, max_value)
    return normMSE

def calculate_expected_rMSE(target, prediction):
    target = tf.cast(target, dtype=tf.float32)
    prediction = tf.cast(prediction, dtype=tf.float32)

    target = tf.math.square(target)
    target = tf.maximum(target, 1e-7)

    prediction = tf.math.square(prediction)

    # relative error is (pred/target) - 1
    relative = tf.math.divide(prediction, target)
    relative_error = tf.subtract(relative, 1.0)
    relative_error = tf.math.reduce_mean(relative_error)  # (bacthes), get the mean error per timestep

    relative_error = tf.math.multiply(tf.cast(100.0, dtype=tf.float32),
                                      relative_error)  # (batches) convert into percents

    return relative_error
def loss_expected_rMSE():
    def rMSE(y_true, y_pred):
        return calculate_expected_rMSE(y_true)
    return rMSE

def calculate_pdf_rMSE(target, prediction):

    output_tiles = tf.shape(target)
    output_tiles = output_tiles[-1]
    indices = tf.range(output_tiles)
    indices = tf.cast(indices, dtype=tf.float32)

    ev_target = tf.multiply(target, indices)
    ev_target = tf.reduce_sum(ev_target, axis=-1)


    ev_prediction = tf.multiply(prediction, indices)
    ev_prediction = tf.reduce_sum(ev_prediction, axis=-1)

    relative_error = calculate_expected_rMSE(ev_target, ev_prediction)

    return relative_error
def loss_pdf_rMSE():
    def pdf_rMSE(target, prediction):
        return calculate_pdf_rMSE(target, prediction)
    return pdf_rMSE

# ToDo: add the probailistic losses n shit
def pdf_tile_to_fc_loss(target, prediction, distance_tensor):
    # We wanna compare evrery single tile of the PDF to the whole target pdf
    # this way, we develop a measure that includes attributes of the KL divergence and of a distance based error
    target_shape = tf.shape(target)
    output_tiles = target_shape[-1]
    batch_size = target_shape[0]

    target = tf.cast(target, dtype=tf.float32)
    prediction = tf.cast(prediction, dtype=tf.float32)

    distance_tensor = tf.tile(distance_tensor, [batch_size, 1, 1, 1]) # (batches, target_steps, output_tiles, output_tiles)

    loss = tf.subtract(prediction, target)
    loss = tf.abs(loss)
    loss = tf.subtract(1.0, loss)
    loss = tf.math.maximum(loss, 1e-9) #prevent 0s, because log and 0 is not good
    loss = -tf.math.log(loss)
    loss = tf.tile(tf.expand_dims(loss, axis=-1), [1, 1, 1, output_tiles])

    weighted_distance_tensor = tf.tile(tf.expand_dims(target, axis=-1), [1, 1, 1, output_tiles]) # (batches, target_steps, output_tiles, output_tiles)
    weighted_distance_tensor = tf.multiply(tf.cast(distance_tensor, dtype=tf.float32), weighted_distance_tensor)
    weighted_distance_tensor = tf.transpose(weighted_distance_tensor, perm=[0, 1, 3, 2]) # iIrc this is necessary to shift

    loss = tf.multiply(weighted_distance_tensor, loss)
    loss = tf.reduce_sum(loss, axis=-1)  # sum over tile-to-tile errors for each tile
    loss = tf.reduce_sum(loss, axis=-1)  # sum over tiles for each timestep
    loss = tf.reduce_mean(loss)  # average

    return loss
def loss_tile_to_pdf(out_steps, out_tiles):

    # for that, we need a tensor that holds the distances from each point to another
    # if theres a smarter way to do that, I'd be happy
    distance_tensor = [tf.range(out_tiles)]  # shape (output_tiles)
    for tile in range(1, out_tiles):
        distance_tensor = tf.concat([distance_tensor, [tf.subtract(tf.range(out_tiles), tile)]], axis=0)
    distance_tensor = tf.abs(distance_tensor)  # shape (output_tiles, output_tiles)
    distance_tensor = tf.expand_dims(distance_tensor, axis=0)
    distance_tensor = tf.expand_dims(distance_tensor, axis=0)  # shape (1,1, output_tiles, output_tiles)
    distance_tensor = tf.tile(distance_tensor,
                              [1, out_steps, 1, 1])  # (1, target_steps, output_tiles, output_tiles)

    def tile_to_pdf(target, prediction):
        return pdf_tile_to_fc_loss(target, prediction, distance_tensor)
    return tile_to_pdf




def calculate_KL_Divergence(target, prediction):
    target = tf.cast(target, dtype=tf.float32)
    prediction = tf.cast(prediction, dtype=tf.float32)

    # KL_divergence = sum_over_bands( q(band) * ln(q(band)/p(band)))
    KL_divergence = tf.divide(tf.math.maximum(prediction, 1e-8), tf.math.maximum(target, 1e-8))
    KL_divergence = tf.math.log(KL_divergence)
    KL_divergence = tf.multiply(prediction, KL_divergence)
    KL_divergence = tf.reduce_sum(KL_divergence, axis=-1)
    KL_divergence = tf.reduce_mean(KL_divergence)
    return KL_divergence

def loss_KL_Divergence():
    def KL_D(target, prediction):
        return calculate_KL_Divergence(target, prediction)
    return KL_D