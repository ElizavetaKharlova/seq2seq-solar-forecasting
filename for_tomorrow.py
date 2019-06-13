from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import gzip, pickle
from datasets_utils import datasets_from_data

import numpy as np
import os
import time

def get_their_data():
#    from numpy import nan
#    from numpy import isnan
    from pandas import read_csv
#    from pandas import to_numeric
#
#    # fill missing values with a value at the same time one day ago
#    def fill_missing(values):
#        one_day = 60 * 24
#        for row in range(values.shape[0]):
#            for col in range(values.shape[1]):
#                if isnan(values[row, col]):
#                    values[row, col] = values[row - one_day, col]
#
#    # load all data
#    dataset = read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
#    # mark all missing values
#    dataset.replace('?', nan, inplace=True)
#    # make dataset numeric
#    dataset = dataset.astype('float32')
#    # fill missing
#    fill_missing(dataset.values)
#    # add a column for for the remainder of sub metering
#    values = dataset.values
#    dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
#    # save updated dataset
#    dataset.to_csv('household_power_consumption.csv')
#    # load the new file
#    dataset = read_csv('household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#    # resample data to daily
#    daily_groups = dataset.resample('D')
#    daily_data = daily_groups.sum()
#    # save
#    daily_data.to_csv('household_power_consumption_days.csv')

    from numpy import split
    from numpy import array

    # split a univariate dataset into train/test sets
    def split_dataset(data):
        # split into standard weeks
        train, test = data[1:-328], data[-328:-6]
        # restructure into windows of weekly data
        train = array(split(train, len(train)/7))
        test = array(split(test, len(test)/7))
        return train, test

    # load the new file
    dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    train, test = split_dataset(dataset.values)
    
    # convert history into inputs and outputs
    def to_supervised(train, n_input, n_out=7):
        # flatten data
        data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end < len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)
            
    n_input=14
    train_x, train_y = to_supervised(train, n_input)
    test_x, test_y = to_supervised(test, n_input)
    return train_x, train_y, test_x, test_y

def get_data(data_path):
#    data_path = './data/Lizas_data.pklz'
    f = gzip.open(data_path,'rb')
    NetworkData = pickle.load(f)
    f.close()

    mvts_array = NetworkData['dataset']
    specs = NetworkData['specs']
    print('Specs:', specs)

    inputs, ev_targets, pdf_targets = datasets_from_data(data=mvts_array, sw_len_samples=specs['sw_steps'], fc_len_samples=specs['num_steps'], fc_steps=specs['forecast_steps'], fc_tiles=specs['num_tiles'], target_dims=specs['target'], plot=False, steps_to_new_sample=1)
    print('Input shape:', inputs.shape, 'Target shape:', ev_targets.shape, pdf_targets.shape)
    return inputs, ev_targets, pdf_targets

def split_train_test(whole_set, split_size):
    split_ind = int(split_size*whole_set.shape[0])
    return whole_set[split_ind:], whole_set[:split_ind]


class LSTM_layer(tf.keras.layers.Layer):
    def __init__(self, drop_rate, units_size, return_sequences):
        super(LSTM_layer, self).__init__()
        self.lstm_layer = tf.keras.layers.LSTM(units=units_size, return_sequences=return_sequences, return_state=True)
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x):
        out, state1, state2 = self.lstm_layer(x)
        state = [state1, state2]
#        out = self.dropout(out)
        return out, state

class LSTM_encoder(tf.keras.layers.Layer):
    def __init__(self, units_size):
        super(LSTM_encoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(dtype='float32')
#        self.encoder = LSTM_layer(units_size=units_size, return_sequences=True, drop_rate=0.0)
        self.encoder = tf.keras.layers.GRU(units=units_size, return_sequences=True, return_state=True)

    def call(self,x):
#        self.input_layer(x[0].shape)
        out, state = self.encoder(x)
        return out, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class LSTM_decoder(tf.keras.layers.Layer):
    def __init__(self, units_size, return_sequences):
        super(LSTM_decoder, self).__init__()
        self.decoder = tf.keras.layers.GRU(units=units_size, return_sequences=return_sequences, return_state=True, stateful=True)
        self.decoder_wrapped = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=100, activation='relu'))
        self.decoder_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1))

    def call(self, x, hidden, enc_output):
#        state1 = hidden[0]
#        state2 = hidden[1]
#        print(state1.shape[0])
        self.decoder.reset_states(states = hidden)
        out, state= self.decoder(x)
        state = [state1, state2]
        out = self.decoder_wrapped(out)
        out = self.decoder_out(out)
        return out, state


#data_path = './data/Lizas_data.pklz'
#inputs, ev_targets, _ = get_data(data_path)
#inputs_train, inputs_test = split_train_test(inputs, 0.15)
#print('train', inputs_train.shape, 'test', inputs_test.shape)
#ev_target_train, ev_target_test = split_train_test(ev_targets, 0.15)

train_x, train_y, test_x, test_y = get_their_data()
train_x = train_x.astype('float32')
print(train_x.shape, train_y.shape)

encoder = LSTM_encoder(units_size=200)
decoder = LSTM_decoder(units_size=200, return_sequences=True)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.losses.MSE

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)



def loss_fuction(real, pred):
    loss_ = loss_object(real, pred)
    return tf.reduce_mean(loss_)


def build_enc_dec(inputs_train, ev_target_train):
    ev_target_train = ev_target_train.reshape(ev_target_train.shape[0], ev_target_train.shape[1], 1)
    
    loss = 0
    
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inputs_train)
        dec_hidden = enc_hidden
        dec_input = inputs_train[:,-1,0]
        
        for t in range(1,ev_target_train.shape[1]):
            pred, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
            loss =+ loss_fuction(ev_target_train[:,t], pred)
            dec_input = tf.expand_dims(ev_target_train[:,t],1)

    batch_loss = (loss / int(ev_target_train.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

EPOCHS = 10
batch_size = 16

for epoch in range(EPOCHS):
    start = time.time()
    print('EPOCH #:', epoch)
#    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
            
    for batch in range(int(train_x.shape[0]/batch_size)):
        train_x_batch = train_x[batch:batch+batch_size, :,:]
        train_y_batch = train_y[batch:batch+batch_size, :]
        
        batch_loss = build_enc_dec(train_x_batch, train_y_batch)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                 batch,
                                                 batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
                
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate_forecasts(actual, predicted):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


