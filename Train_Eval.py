# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf
import numpy as np
#from Building_Blocks import build_model
from Benchmarks import build_model
import copy
import glob
import os
import pickle
from random import shuffle

class Model_Container():
    def __init__(self,
                 dataset_folder,
                 model_kwargs, #see __build_model
                 train_kwargs, #currently only batch size
                 try_distribution_across_GPUs=True,
                 ):
        self.dataset_path = dataset_folder
        #if try_distribution_across_GPUs:

        self.strategy = tf.distribute.MirroredStrategy()

        self.dataset_info = pickle.load(open(dataset_folder + '/dataset_info.pickle', 'rb'))
        self.sw_len_samples = self.dataset_info['sw_len_samples']
        #self.sw_len_samples = int(5 * 24 * 60)
        self.fc_len_samples = self.dataset_info['fc_len_samples']
        #self.fc_len_samples = int(1 * 24 * 60)
        self.fc_steps = self.dataset_info['fc_steps']
        #self.fc_steps = 24
        self.fc_tiles = self.dataset_info['fc_tiles']
        #self.fc_tiles = 40
        self.nwp_downsampling_rate = self.dataset_info['nwp_downsampling_rate']
        # self.nwp_downsampling_rate = 15
        self.nwp_dims = self.dataset_info['nwp_dims']
        self.nwp_dims = 16
        self.teacher_shape = [self.fc_steps, self.fc_tiles]
        self.target_shape = [self.fc_steps, self.fc_tiles]
        self.raw_nwp_shape = [int( (self.sw_len_samples + self.fc_len_samples)/self.nwp_downsampling_rate ), self.nwp_dims]
        self.pdf_history_shape = [int(self.sw_len_samples/(self.fc_len_samples/self.fc_steps)), self.fc_tiles]

        if model_kwargs['model_type'] == 'Encoder-Decoder' or model_kwargs['model_type'] == 'E-D' or 'MiMo' in model_kwargs['model_type']:
            self.actual_input_shape = [int(self.sw_len_samples / self.nwp_downsampling_rate),
                                       self.raw_nwp_shape[1] + 1]
            model_kwargs['input_shape'] = self.actual_input_shape
        elif 'MiMo' in model_kwargs['model_type']:
            self.actual_input_shape = [int(self.sw_len_samples / self.nwp_downsampling_rate),
                                       self.raw_nwp_shape[1] + 1]
            model_kwargs['input_shape'] = self.actual_input_shape
        elif 'generator' in model_kwargs['model_type']:
            model_kwargs['input_shape'] = 7

        model_kwargs['out_shape'] = self.target_shape
        model_kwargs['normalizer_value'] = 1.0
        # self.dataset_info['normalizer_value']

        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs

    def get_results(self):
        tf.keras.backend.clear_session()
        # TODo, see wether this makes sense first

        # strategy = tf.distribute.MirroredStrategy()
        # strategy = tf.distribute.experimental.CentralStorageStrategy()
        # with strategy.scope():
        # make sure we are working clean
        self.__create_datasets()
        # strategy = tf.distribute.MirroredStrategy()
        # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # # strategy = tf.distribute.experimental.CentralStorageStrategy()
        # with strategy.scope():
        self.__build_model(**self.model_kwargs)

        self.__train_model(**self.train_kwargs)
        del self.train_kwargs
        del self.model

        self.__skill_metrics()

        tf.keras.backend.clear_session()
        return self.metrics

    def __create_datasets(self):
        def __get_all_tfrecords_in_folder(folder):
            if os.path.isdir(folder):
                file_list = glob.glob(folder + '/*.tfrecord')
            else:
                print('didnt find training data folder, expect issues!!')
            return file_list
        nwp_shape = self.raw_nwp_shape
        flattened_nwp_shape = 1
        for dimension_shape in nwp_shape:
            flattened_nwp_shape = flattened_nwp_shape * dimension_shape
        pv_input_shape = [self.sw_len_samples]
        output_shape = self.target_shape
        flattened_output_shape = 1
        for dimension_shape in output_shape:
            flattened_output_shape = flattened_output_shape * dimension_shape

        def __read_and_process_train_samples(example):

            features = {'nwp_input': tf.io.FixedLenFeature(flattened_nwp_shape, tf.float32),
                        'raw_historical_input': tf.io.FixedLenFeature(pv_input_shape, tf.float32),
                        'pdf_historical_input': tf.io.FixedLenFeature(self.pdf_history_shape[0] * self.pdf_history_shape[1], tf.float32),
                        'pdf_target': tf.io.FixedLenFeature(flattened_output_shape, tf.float32),
                        'pdf_teacher': tf.io.FixedLenFeature(flattened_output_shape, tf.float32),
                        }
            unadjusted_example = tf.io.parse_single_example(example, features)
            nwp_inputs = tf.reshape(tensor=unadjusted_example['nwp_input'], shape=[nwp_shape[0], nwp_shape[1]])

            target = tf.reshape(tensor=unadjusted_example['pdf_target'], shape=[output_shape[0], output_shape[1]])


            # ToDo: Current Dev Stack
            if 'generator' in self.model_kwargs['model_type']:
                print('yaaay', self.model_kwargs['model_type'], 'time!!!!')

                history_pdf = tf.reshape(tensor=unadjusted_example['pdf_historical_input'],
                                         shape=[self.pdf_history_shape[0], self.pdf_history_shape[1]])
                teacher = tf.reshape(tensor=unadjusted_example['pdf_teacher'],
                                     shape=[output_shape[0], output_shape[1]])
                adjusted_train_target = tf.concat([history_pdf[1:, :], target], axis=0)
                print(adjusted_train_target)
                return{'support_input': nwp_inputs,
                       'history_input': history_pdf,
                       'teacher_input': teacher}, adjusted_train_target

            elif self.model_kwargs['model_type'] == 'Encoder-Decoder' or self.model_kwargs['model_type'] == 'E-D' or 'MiMo' in self.model_kwargs['model_type']:
                history_pv = tf.reshape(tensor=unadjusted_example['raw_historical_input'], shape=[self.nwp_downsampling_rate, int(self.sw_len_samples/self.nwp_downsampling_rate)])
                history_pv = tf.transpose(history_pv, [1,0])
                history_pv = tf.reduce_mean(history_pv, axis=-1, keepdims=True)
                nwp_inputs = nwp_inputs[int(self.fc_len_samples/self.nwp_downsampling_rate):,:]
                inputs =tf.concat([nwp_inputs, history_pv], axis=-1)
                if 'MiMo' in self.model_kwargs['model_type']:
                    return inputs, target
                else:
                    teacher = tf.reshape(tensor=unadjusted_example['pdf_teacher'],
                                         shape=[output_shape[0], output_shape[1]])
                    return {'inputs_input': inputs, 'teacher_input': teacher}, target

        def __read_and_process_normal_samples(example):
            # 'nwp_input':
            # 'raw_historical_input':
            # 'raw_teacher':
            # 'raw_target':
            # 'pdf_historical_input':
            # 'pdf_teacher':
            # 'pdf_target':

            features = {'nwp_input': tf.io.FixedLenFeature(flattened_nwp_shape, tf.float32),
                        'raw_historical_input': tf.io.FixedLenFeature(pv_input_shape, tf.float32),
                        'pdf_historical_input': tf.io.FixedLenFeature(self.pdf_history_shape[0] * self.pdf_history_shape[1], tf.float32),
                        'pdf_target': tf.io.FixedLenFeature(flattened_output_shape, tf.float32),
                        'pdf_teacher': tf.io.FixedLenFeature(flattened_output_shape, tf.float32),
                        }
            unadjusted_example = tf.io.parse_single_example(example, features)
            nwp_inputs = tf.reshape(tensor=unadjusted_example['nwp_input'], shape=[nwp_shape[0], nwp_shape[1]])

            target = tf.reshape(tensor=unadjusted_example['pdf_target'], shape=[output_shape[0], output_shape[1]])


            # ToDo: Current Dev Stack
            if 'generator' in self.model_kwargs['model_type']:
                print('yaaay', self.model_kwargs['model_type'], 'time!!!!')

                history_pdf = tf.reshape(tensor=unadjusted_example['pdf_historical_input'],
                                         shape=[self.pdf_history_shape[0], self.pdf_history_shape[1]])
                teacher = tf.reshape(tensor=unadjusted_example['pdf_teacher'],
                                     shape=[output_shape[0], output_shape[1]])
                return{'support_input': nwp_inputs,
                       'history_input': history_pdf,
                       'teacher_input': teacher}, target

            elif self.model_kwargs['model_type'] == 'Encoder-Decoder' or self.model_kwargs['model_type'] == 'E-D' or 'MiMo' in self.model_kwargs['model_type']:
                history_pv = tf.reshape(tensor=unadjusted_example['raw_historical_input'], shape=[self.nwp_downsampling_rate, int(self.sw_len_samples/self.nwp_downsampling_rate)])
                history_pv = tf.transpose(history_pv, [1,0])
                history_pv = tf.reduce_mean(history_pv, axis=-1, keepdims=True)
                nwp_inputs = nwp_inputs[int(self.fc_len_samples/self.nwp_downsampling_rate):,:]
                inputs =tf.concat([nwp_inputs, history_pv], axis=-1)
                if 'MiMo' in self.model_kwargs['model_type']:
                    return inputs, target
                else:
                    teacher = tf.reshape(tensor=unadjusted_example['pdf_teacher'],
                                         shape=[output_shape[0], output_shape[1]])
                    return {'inputs_input': inputs, 'teacher_input': teacher}, target

        def get_batched_dataset(file_list, batch_size, train_mode=False):
            option_no_order = tf.data.Options()
            option_no_order.experimental_deterministic = False

            #dataset = tf.data.Dataset.list_files(file_list)
            dataset = tf.data.TFRecordDataset(file_list, num_parallel_reads=20)
            dataset = dataset.with_options(option_no_order)
            #dataset = dataset.interleave(map_func=tf.data.TFRecordDataset,
            #                              num_parallel_calls=20)

            # does this order make any fucking sense at all?!?!?!?
            if train_mode:
                dataset = dataset.map(__read_and_process_train_samples, num_parallel_calls=10)
            else:
                dataset = dataset.map(__read_and_process_normal_samples, num_parallel_calls=10)
            dataset = dataset.repeat()
            dataset = dataset.shuffle(4 * batch_size)
            dataset = dataset.prefetch(5 * batch_size)
            dataset = dataset.batch(batch_size, drop_remainder=False)
            return dataset

        batch_size = self.train_kwargs['batch_size']
        # ToDo: change back to '/train'
        train_folder = self.dataset_path + '/train'
        train_list = __get_all_tfrecords_in_folder(train_folder)
        shuffle(train_list)
        def get_training_dataset():
            return get_batched_dataset(train_list, batch_size, train_mode=False)
        self.train_dataset_generator = get_training_dataset
        self.train_steps_epr_epoch = int(np.ceil(len(train_list)/batch_size))

        val_folder = self.dataset_path + '/validation'
        val_list = __get_all_tfrecords_in_folder(val_folder)
        shuffle(val_list)
        def get_val_dataset():
            return get_batched_dataset(val_list, batch_size, train_mode=False)
        self.val_dataset_generator = get_val_dataset
        self.val_steps_epr_epoch = int(np.ceil(len(val_list) / batch_size))

        test_folder = self.dataset_path + '/test'
        test_list = __get_all_tfrecords_in_folder(test_folder)
        shuffle(test_list)
        def get_test_dataset():
            return get_batched_dataset(test_list, batch_size, train_mode=False)
        self.test_dataset_generator = get_test_dataset
        self.test_steps_epr_epoch = int(np.ceil(len(test_list) / batch_size))

    def __build_model(self,
                      input_shape, out_shape, normalizer_value,
                      model_type='MiMo-sth',
                      model_size='small',
                      use_dropout=False, dropout_rate=0.0, use_hw=False, use_norm=False, use_quasi_dense=False, #general architecture stuff
                      use_attention=False, attention_hidden=False, self_recurrent=False, # E-D stuff
                      downsample=True, mode='snip', #MiMo stuff
                      ):

        # MiMo LSTMS, downsampled
        if model_size == 'small':
            units = [[32, 32, 32]]
        elif model_size =='medium' or model_size == 'med':
            units = [[64, 64, 64, 64, 64]]
        elif model_size == 'big':
            units = [[128, 128, 128]]
        elif model_size == 'large':
            units = [[256, 256]]
        elif model_size == 'generator':
            units = [[128]]

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

        elif model_type == 'MiMo-just-tcn':
            print('building a', model_size, model_type)
            from Building_Blocks import DenseTCN
            encoder_specs = {'num_blocks': 3,
                             'num_layers_per_block': 3,
                            'growth_rate': 12,
                            'use_dropout': False,
                            'dropout_rate': 0.0,
                            'use_norm': False,
                            'downsample_rate': 1}


            from Models import mimo_model
            self.model = mimo_model(function_block=DenseTCN(**encoder_specs),
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
            decoder_specs['use_attention'] = use_attention
            decoder_specs['attention_hidden'] = attention_hidden
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

        elif model_type =='whatcouldgowrong':
            from decoder_idea import DenseTCN, whatever
            from Models import forecaster_model
            units_tcn = [[10,10,10,10],[10,10,10,10]]
            encoder = DenseTCN(num_blocks=len(units_tcn),
                                            num_layers_per_block=len(units_tcn[0]),
                                            growth_rate=units_tcn[0][0],
                                            squeeze_factor=0.5,
                                            use_dropout = use_dropout,
                                            dropout_rate=dropout_rate,
                                            use_norm=use_norm,
                                            kernel_sizes=[3])

            projection = tf.keras.layers.Conv1D(filters=out_shape[-1],
                                                kernel_size=1,
                                                strides=1,
                                                padding='causal',
                                                activation=tf.keras.layers.Softmax(axis=-1),
                                                kernel_initializer='glorot_uniform')
            self.model = forecaster_model(encoder_block=encoder,
                                     decoder_block=whatever(),
                                     projection_block=projection,
                                    use_teacher=True,
                                     output_shape=out_shape,
                                     support_shape=self.raw_nwp_shape,
                                     history_shape=self.pdf_history_shape)

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
            print('building E-D')
            common_specs = {'units': units,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm,
                            'use_hw': use_hw,
                            'use_quasi_dense': use_quasi_dense,
                            'only_last_layer_output': True}

            encoder_specs = copy.deepcopy(common_specs)
            decoder_specs = copy.deepcopy(common_specs)
            decoder_specs['use_attention'] = use_attention
            decoder_specs['attention_hidden'] = attention_hidden
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

        elif model_type == 'generator' or model_type == 'Generator':
            print('building E-D')
            common_specs = {'units': units,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm,
                            'use_hw': use_hw,
                            'use_quasi_dense': use_quasi_dense,
                            'only_last_layer_output': True}

            encoder_specs = copy.deepcopy(common_specs)
            encoder_specs['return_state'] = False
            decoder_specs = copy.deepcopy(common_specs)

            decoder_specs['projection'] = tf.keras.layers.Conv1D(filters=out_shape[-1],
                                                kernel_size=1,
                                                strides=1,
                                                padding='causal',
                                                activation=tf.keras.layers.Softmax(axis=-1),
                                                kernel_initializer='glorot_uniform')

            from Building_Blocks import block_LSTM, generator_LSTM_block
            from Models import forecaster_model
            self.model = forecaster_model(encoder_block=block_LSTM(**encoder_specs),
                                     decoder_block=generator_LSTM_block(**decoder_specs),
                                    use_teacher=True,
                                     output_shape=out_shape,
                                     support_shape=self.raw_nwp_shape,
                                     history_shape=self.pdf_history_shape)

        else:
            print('trying to build with', model_size, model_type, 'but failed')
        from Losses_and_Metrics import loss_wrapper
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
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
        callbacks = []
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_nRMSE',
                                                                   patience=10,
                                                                   mode='min',
                                                                   restore_best_weights=True))
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                           factor=0.10,
                                                                           patience=5,
                                                                           verbose=True,
                                                                           mode='min',
                                                                           cooldown=5))
        print('starting to train model')
        train_history = self.model.fit(self.train_dataset_generator(),
                                       steps_per_epoch=self.train_steps_epr_epoch,
                                      epochs=100,
                                      verbose=2,
                                      validation_data=self.val_dataset_generator(),
                                       validation_steps=self.val_steps_epr_epoch,
                                      callbacks=callbacks)

        train_history = train_history.history
        for key in train_history.keys():
            self.metrics[key] = train_history[key]

        test_results = self.model.evaluate(self.test_dataset_generator(),
                                           steps=self.test_steps_epr_epoch,
                                      verbose=2)

        self.metrics['test_loss'] = test_results[0]
        self.metrics['test_nME'] = test_results[1]
        self.metrics['test_nRMSE'] = test_results[2]
        self.metrics['test_CRPS'] = test_results[3]


    def __skill_metrics(self):
        saved_epoch = np.argmax(self.metrics['val_nRMSE'])
        # self.metrics['val_nRMSE_skill'] = 1 - (self.metrics['val_nRMSE'][saved_epoch] / self.dataset_info['val_persistency_baseline']['nRMSE'])
        # self.metrics['val_nRMSE_skill'] = 1 - (self.metrics['val_CRPS'][saved_epoch] / self.dataset_info['val_persistency_baseline']['CRPS'])
        # self.metrics['val_nRMSE_skill'] = 1 - (self.metrics['test_nRMSE'] / self.dataset_info['test_persistency_baseline']['nRMSE'])
        # self.metrics['test_CRPS_skill'] = 1 - (self.metrics['test_CRPS'] / self.dataset_info['test_persistency_baseline']['CRPS'])



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









