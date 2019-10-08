# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf
import numpy as np
#from Building_Blocks import build_model
from Benchmarks import build_model
import copy


class Model_Container():
    def __init__(self,
                 dataset,
                 model_kwargs, #see __build_model
                 train_kwargs, #currently only batch size
                 ):
        self.dataset = dataset

        model_kwargs['input_shape'] = dataset['train_inputs'].shape[1:]
        model_kwargs['out_shape'] = dataset['train_targets'].shape[1:]
        model_kwargs['normalizer_value'] = dataset['normalizer_value']

        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs

    def get_results(self):
        tf.keras.backend.clear_session()  # make sure we are working clean
        self.__build_model(**self.model_kwargs)
        del self.model_kwargs

        self.__train_model(**self.train_kwargs)
        del self.train_kwargs
        del self.model

        self.__skill_metrics()
        del self.dataset

        tf.keras.backend.clear_session()
        return self.metrics


    def __build_model(self,
                      input_shape, out_shape, normalizer_value,
                      model_type='MiMo-sth',
                      model_size='small',
                      use_dropout=False, dropout_rate=0.0, use_hw=False, use_norm=False, use_quasi_dense=False, #general architecture stuff
                      use_attention=False, self_recurrent=False, # E-D stuff
                      downsample=True, mode='snip', #MiMo stuff
                      ):

        # MiMo LSTMS, downsampled
        if model_size == 'small':
            units = [[32, 32, 32]]
        elif model_size =='medium' or model_size == 'med':
            units = [[64, 64, 64]]
        elif model_size == 'big':
            units = [[128, 128, 128]]
        elif model_size == 'large':
            units = [[256, 256, 256]]

        if model_type == 'MiMo-LSTM':
            print('building a', model_size, model_type)
            from Building_Blocks import block_LSTM
            encoder_specs = {'units': units,
                             'use_dropout': use_dropout,
                             'dropout_rate': dropout_rate,
                             'use_norm': use_norm,
                             'use_hw': use_hw,
                             'return_state': False,
                             'use_quasi_dense': use_quasi_dense,
                             'only_last_layer_output': True}

            from Models import mimo_model
            self.model = mimo_model(function_block=block_LSTM(**encoder_specs),
                               input_shape=input_shape,
                               output_shape=out_shape,
                               downsample_input=downsample,
                               downsampling_rate=(60 / 5),
                               mode=mode)
    
        elif model_type == 'MiMo-attn-tcn':
            print('building a', model_size, model_type)
            from Building_Blocks import attentive_TCN
            encoder_specs = {'units': [[32],[32],[32],[32],[32]],
                             'use_dropout': use_dropout,
                             'dropout_rate': dropout_rate,
                             'use_norm': True}

            from Models import mimo_model
            self.model = mimo_model(function_block=attentive_TCN(**encoder_specs),
                               input_shape=input_shape,
                               output_shape=out_shape,
                               downsample_input=False,
                               downsampling_rate=(60 / 5),
                               mode='project')

        elif model_type == 'Encoder-TCN' or model_type == 'E-TCN':
            common_specs = {'units': units,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm,
                            'use_hw': use_hw,
                            'use_quasi_dense': use_quasi_dense,
                            'only_last_layer_output': True}

            # encoder_specs = copy.deepcopy(common_specs)
            decoder_specs = copy.deepcopy(common_specs)
            encoder_specs = {'units': [[32],[32],[32]],
                             'use_dropout': use_dropout,
                             'dropout_rate': dropout_rate,
                             'use_norm': use_norm}
            decoder_specs['use_attention'] = True
            decoder_specs['attention_hidden'] = False
            decoder_specs['self_recurrent'] = self_recurrent

            projection_model = tf.keras.layers.Dense(units=out_shape[-1],
                                                     activation=tf.keras.layers.Softmax(axis=-1))
            projection_model = tf.keras.layers.TimeDistributed(projection_model)

            from Building_Blocks import attentive_TCN, decoder_LSTM_block
            model_kwargs = {'encoder_block': attentive_TCN(**encoder_specs),
                            "encoder_stateful": False,
                            'decoder_block': decoder_LSTM_block(**decoder_specs),
                            'use_teacher': True,
                            'decoder_uses_attention_on': decoder_specs['use_attention'],
                            'decoder_stateful': True,
                            'self_recurrent_decoder': decoder_specs['self_recurrent'],
                            'projection_block': projection_model,
                            'input_shape': input_shape,
                            'output_shape': out_shape}

            from Models import encoder_decoder_model
            self.model = encoder_decoder_model(**model_kwargs)

        elif model_type == 'Encoder-Decoder-TCN' or model_type == 'E-D-TCN':
            common_specs = {'units': [[32],[32],[32]],
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': True}

            encoder_specs = copy.deepcopy(common_specs)
            decoder_specs = copy.deepcopy(common_specs)
                             
            decoder_specs['use_attention'] = True
            decoder_specs['self_recurrent'] = self_recurrent
            decoder_specs['mode'] = 'decoder'

            projection_model = tf.keras.layers.Dense(units=out_shape[-1],
                                                     activation=tf.keras.layers.Softmax(axis=-1))
            projection_model = tf.keras.layers.TimeDistributed(projection_model)

            from Building_Blocks import attentive_TCN
            model_kwargs = {'encoder_block': attentive_TCN(**encoder_specs),
                            "encoder_stateful": False,
                            'decoder_block': attentive_TCN(**decoder_specs),
                            'use_teacher': True,
                            'decoder_uses_attention_on': decoder_specs['use_attention'],
                            'decoder_stateful': False,
                            'self_recurrent_decoder': decoder_specs['self_recurrent'],
                            'projection_block': projection_model,
                            'input_shape': input_shape,
                            'output_shape': out_shape}

            from Models import encoder_decoder_model
            self.model = encoder_decoder_model(**model_kwargs)

        elif model_type == 'Encoder-Decoder' or model_type == 'E-D':
            common_specs = {'units': units,
                            'use_dropout': use_dropout,
                            'dropout_rate': 0.15,
                            'use_norm': use_norm,
                            'use_hw': True,
                            'use_quasi_dense': use_quasi_dense,
                            'only_last_layer_output': True}

            encoder_specs = copy.deepcopy(common_specs)
            decoder_specs = copy.deepcopy(common_specs)
            decoder_specs['use_attention'] = True
            decoder_specs['self_recurrent'] = self_recurrent

            projection_model = tf.keras.layers.Dense(units=out_shape[-1],
                                                     activation=tf.keras.layers.Softmax(axis=-1))
            projection_model = tf.keras.layers.TimeDistributed(projection_model)

            from Building_Blocks import block_LSTM, decoder_LSTM_block
            model_kwargs = {'encoder_block': block_LSTM(**encoder_specs),
                            "encoder_stateful": True,
                            'decoder_block': decoder_LSTM_block(**decoder_specs),
                            'use_teacher': True,
                            'decoder_uses_attention_on': decoder_specs['use_attention'],
                            'decoder_stateful': True,
                            'self_recurrent_decoder': decoder_specs['self_recurrent'],
                            'projection_block': projection_model,
                            'input_shape': input_shape,
                            'output_shape': out_shape}

            from Models import encoder_decoder_model
            self.model = encoder_decoder_model(**model_kwargs)

        else:
            print('trying to build with', model_size, model_type, 'but failed')


        from Losses_and_Metrics import loss_wrapper
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='tile-to-forecast'),
                      metrics=[loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nME',
                                            normalizer_value=normalizer_value),
                               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nRMSE',
                                            normalizer_value=normalizer_value),
                               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='CRPS'),
                               ])  # compile, print summary
        self.model.summary()

    def __train_model(self, batch_size=64):
        self.metrics = {}

        tf.keras.backend.clear_session()
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_nRMSE',
                                                                   patience=10,
                                                                   mode='min',
                                                                   restore_best_weights=True)
        reduce_lr_on_plateu_callback =tf.keras.callbacks.ReduceLROnPlateau(monitor='nRMSE',
                                                                           factor=0.10,
                                                                           patience=5,
                                                                           verbose=True,
                                                                           mode='min',
                                                                           cooldown=5)

        train_history = self.model.fit(x=[self.dataset['train_inputs'],
                                     self.dataset['train_teacher'],
                                     self.dataset['train_blend']],
                                  y=self.dataset['train_targets'],
                                  batch_size=batch_size,
                                  epochs=100,
                                  shuffle=True,
                                  verbose=True,
                                  validation_data=([self.dataset['val_inputs'],
                                                    self.dataset['val_teacher'],
                                                    self.dataset['val_blend']],
                                                   self.dataset['val_targets']),
                                  callbacks=[early_stopping_callback, reduce_lr_on_plateu_callback])

        train_history = train_history.history
        for key in train_history.keys():
            self.metrics[key] = train_history[key]

        tf.keras.backend.clear_session()


        test_results = self.model.evaluate(x=[self.dataset['test_inputs'],
                                              self.dataset['test_teacher'],
                                              self.dataset['test_blend']],
                                      y=self.dataset['test_targets'],
                                      batch_size=batch_size,
                                      verbose=False)

        self.metrics['test_loss'] = test_results[0]
        self.metrics['test_nME'] = test_results[1]
        self.metrics['test_nRMSE'] = test_results[2]
        self.metrics['test_CRPS'] = test_results[3]


    def __skill_metrics(self):
        saved_epoch = np.argmax(self.metrics['val_nRMSE'])
        self.metrics['val_nRMSE_skill'] = 1 - (self.metrics['val_nRMSE'][saved_epoch] / self.dataset['val_persistency_baseline']['nRMSE'])
        self.metrics['val_nRMSE_skill'] = 1 - (self.metrics['val_CRPS'][saved_epoch] / self.dataset['val_persistency_baseline']['CRPS'])
        self.metrics['val_nRMSE_skill'] = 1 - (self.metrics['test_nRMSE'] / self.dataset['test_persistency_baseline']['nRMSE'])
        self.metrics['test_CRPS_skill'] = 1 - (self.metrics['test_CRPS'] / self.dataset['test_persistency_baseline']['CRPS'])



def __get_max_min_targets(train_targets, test_targets):
    import numpy as np
    max_value_train = np.amax(train_targets, axis=0)
    max_value_train = np.amax(max_value_train, axis=0)
    max_value_test = np.amax(test_targets, axis=0)
    max_value_test = np.amax(max_value_test, axis=0)
    max_value = np.maximum(max_value_train, max_value_test)

    min_value_train = np.amin(train_targets, axis=0)
    min_value_train = np.amin(min_value_train, axis=0)
    min_value_test = np.amin(test_targets, axis=0)
    min_value_test = np.amin(min_value_test, axis=0)
    min_value = np.minimum(min_value_test, min_value_train)
    return max_value, min_value




#
# class __adjust_teacher_callback(tf.keras.callbacks.Callback):
#     if prev_val_metric < self.metrics['val_nRMSE'][-1]:
#         relative_decrease += 1
#
#     if relative_decrease > 2:
#         # if we have no relative increase in quality towards the previous iteration
#         # then decrease the blend factor
#         self.dataset['train_blend'] = self.dataset['train_blend'] - 0.05
#         self.dataset['train_blend'] = tf.maximum(0.0, self.dataset['train_blend'])
#         print('lowering blend factor')
#         relative_decrease = 0
#
#     prev_val_metric = train_history['val_nRMSE'][0]
#
#     self.model.load_weights('best_weight_set.h5')











