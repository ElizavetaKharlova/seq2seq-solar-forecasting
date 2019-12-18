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
from datetime import datetime

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
        print(self.dataset_info)
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
        # self.nwp_dims = 16
        self.teacher_shape = [self.fc_steps, self.fc_tiles]
        self.target_shape = [self.fc_steps, self.fc_tiles]
        self.raw_nwp_shape = [int( (self.sw_len_samples + self.fc_len_samples)/self.nwp_downsampling_rate ), self.nwp_dims]
        self.pdf_history_shape = [int(self.sw_len_samples/(self.fc_len_samples/self.fc_steps)), self.fc_tiles]

        if model_kwargs['model_type'] == 'Encoder-Decoder' or model_kwargs['model_type'] == 'E-D' or 'MiMo' in model_kwargs['model_type'] or 'E-D' in model_kwargs['model_type']:
            self.actual_input_shape = [int(self.sw_len_samples / self.nwp_downsampling_rate),
                                       self.raw_nwp_shape[1] + 1]
            model_kwargs['input_shape'] = self.actual_input_shape
            # TODO: this is not required, only becasue it asks for it later
            model_kwargs['support_shape'] = [int(3*24*60/15), self.dataset_info['nwp_dims']]
            model_kwargs['history_shape'] = [int(2*24), self.fc_tiles]
        elif 'MiMo' in model_kwargs['model_type']:
            self.actual_input_shape = [int(self.sw_len_samples / self.nwp_downsampling_rate),
                                       self.raw_nwp_shape[1] + 1]
            model_kwargs['input_shape'] = self.actual_input_shape
        elif 'generator' in model_kwargs['model_type']:
            print(self.dataset_info['nwp_dims'])
            model_kwargs['support_shape'] = [int(3*24*60/15), self.dataset_info['nwp_dims']]
            model_kwargs['history_shape'] = [int(2*24), self.fc_tiles]


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
        #tf.debugging.set_log_device_placement(True)
        # cross_ops = tf.distribute.ReductionToOneDevice()
        cross_ops = tf.distribute.HierarchicalCopyAllReduce()
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_ops)
        with strategy.scope():
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

        real_support_length = self.model_kwargs['support_shape'][0]
        real_history_length = self.model_kwargs['history_shape'][0]
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

            elif self.model_kwargs['model_type'] == 'Encoder-Decoder' or self.model_kwargs['model_type'] == 'E-D' or 'MiMo' in self.model_kwargs['model_type'] or 'E-D' in self.model_kwargs['model_type']:
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
                nwp_inputs = nwp_inputs[-real_support_length:,:]
                history_pdf = history_pdf[-real_history_length:,:]
                # ATM sampling last 2days plus forecast day from NWP and last 2 days from history
                return{'support_input': nwp_inputs,
                       'history_input': history_pdf,
                       'teacher_input': teacher}, target

            elif self.model_kwargs['model_type'] == 'Encoder-Decoder' or self.model_kwargs['model_type'] == 'E-D' or 'MiMo' in self.model_kwargs['model_type'] or 'E-D' in self.model_kwargs['model_type']:
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
                dataset = dataset.map(__read_and_process_train_samples, num_parallel_calls=20)
            else:
                dataset = dataset.map(__read_and_process_normal_samples, num_parallel_calls=20)
            dataset = dataset.repeat()
            dataset = dataset.shuffle(20 * batch_size)
            dataset = dataset.prefetch(10 * batch_size)
            dataset = dataset.batch(batch_size, drop_remainder=False)
            return dataset

        batch_size = self.train_kwargs['batch_size']
        # ToDo: change back to '/train'
        train_folder = self.dataset_path + '/train'
        train_list = __get_all_tfrecords_in_folder(train_folder)
        shuffle(train_list)
        val_folder = self.dataset_path + '/validation'
        val_list = __get_all_tfrecords_in_folder(val_folder)
        shuffle(val_list)
        test_folder = self.dataset_path + '/test'
        test_list = __get_all_tfrecords_in_folder(test_folder)
        shuffle(test_list)

        def get_training_dataset():
            #ToDo: revert to train_list instead of val_list
            return get_batched_dataset(train_list, batch_size, train_mode=False)
        self.train_dataset_generator = get_training_dataset
        self.train_steps_epr_epoch = int(len(train_list)/batch_size)


        def get_val_dataset():
            return get_batched_dataset(val_list, batch_size, train_mode=False)
        self.val_dataset_generator = get_val_dataset
        self.val_steps_epr_epoch = int(np.ceil(len(val_list) / batch_size))


        def get_test_dataset():
            return get_batched_dataset(test_list, batch_size, train_mode=False)
        self.test_dataset_generator = get_test_dataset
        self.test_steps_epr_epoch = int(np.ceil(len(test_list) / batch_size))

    def __build_model(self,
                      out_shape, normalizer_value,
                      support_shape, history_shape,
                      input_shape=None,
                      model_type='MiMo-sth',
                      model_size='small',
                      L1=0.0, L2=0.0,
                      use_dropout=False, dropout_rate=0.0,
                      use_hw=False, use_norm=False, use_quasi_dense=False, #general architecture stuff
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
            from Building_Blocks import fixed_attentive_TCN
            encoder_specs = {'units': [[32,32,32], [48,48], [64,64]],
                            'attention_heads': 2,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm}

            from Models import mimo_model
            self.model = mimo_model(function_block=fixed_attentive_TCN(**encoder_specs),
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
                                     history_shape=history_shape)

        elif model_type == 'Encoder-Decoder-TCN' or model_type == 'E-D-TCN':
            common_specs = {'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm}

            decoder_specs = copy.deepcopy(common_specs)

            decoder_specs['use_attention'] = True
            decoder_specs['mode'] = 'decoder'

            projection_model = tf.keras.layers.Dense(units=out_shape[-1],
                                                     activation=tf.keras.layers.Softmax(axis=-1))
            projection_model = tf.keras.layers.TimeDistributed(projection_model)
            projection_layer = tf.keras.layers.Conv1D(filters=out_shape[-1],
                                                kernel_size=1,
                                                strides=1,
                                                padding='causal',
                                                activation=tf.keras.layers.Softmax(axis=-1),
                                                kernel_initializer='glorot_uniform')

            from Building_Blocks import attentive_TCN
            encoder = attentive_TCN(use_norm=use_norm,
                                  use_dropout=use_dropout,
                                  dropout_rate=dropout_rate)

            model_kwargs = {'encoder_block': attentive_TCN(**common_specs),
                            'decoder_block': attentive_TCN(**decoder_specs),
                            'projection_block': projection_layer,
                            'input_shape': input_shape,
                            'output_shape': out_shape,
                            'encoder_stateful': False,
                            'use_teacher': True,
                            'decoder_uses_attention_on': decoder_specs['use_attention'],
                            'decoder_stateful': False,}

            from Models import encoder_decoder_model
            self.model = encoder_decoder_model(**model_kwargs)

        elif model_type == 'Encoder-Decoder' or model_type == 'E-D':
            print('building E-D')
            common_specs = {'units': units,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm,
                            'use_hw': use_hw,}

            encoder_specs = copy.deepcopy(common_specs)
            decoder_specs = copy.deepcopy(common_specs)
            decoder_specs['use_attention'] = use_attention
            decoder_specs['self_recurrent'] = self_recurrent

            projection_model = tf.keras.layers.Conv1D(filters=out_shape[-1],
                                                kernel_size=1,
                                                strides=1,
                                                padding='causal',
                                                activation=tf.keras.layers.Softmax(axis=-1),
                                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                kernel_initializer='glorot_uniform')

            from Building_Blocks import block_LSTM, decoder_LSTM_block
            model_kwargs = {'encoder_block': block_LSTM(**encoder_specs),
                            'encoder_stateful': True,
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
            common_specs = {'attention_heads': 3,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm}

            encoder_specs = copy.deepcopy(common_specs)
            encoder_specs['units'] = [[32], [48], [64]]

            decoder_specs = copy.deepcopy(common_specs)
            decoder_specs['units'] = [[32], [32]]
            decoder_specs['projection'] = tf.keras.layers.Conv1D(filters=out_shape[-1],
                                                kernel_size=1,
                                                strides=1,
                                                padding='causal',
                                                activation=tf.keras.layers.Softmax(axis=-1),
                                                kernel_initializer='glorot_uniform')

            from Building_Blocks import sa_encoder_LSTM, fixed_generator_LSTM_block
            from Models import forecaster_model
            self.model = forecaster_model(encoder_block=sa_encoder_LSTM(**encoder_specs),
                                     decoder_block=fixed_generator_LSTM_block(**decoder_specs),
                                    use_teacher=True,
                                     output_shape=out_shape,
                                     support_shape=support_shape,
                                     history_shape=history_shape)
        elif model_type == 'FFWgenerator' or model_type == 'FFWGenerator':
            print('building E-D')

            projection_layer = tf.keras.layers.Conv1D(filters=out_shape[-1],
                                                kernel_size=1,
                                                strides=1,
                                                padding='causal',
                                                activation=tf.keras.layers.Softmax(axis=-1),
                                                kernel_initializer='glorot_uniform')

            from Building_Blocks import FFW_encoder, FFW_generator
            encoder = FFW_encoder(units=[[64], [96], [128]],
                                  use_norm=use_norm,
                                  use_dropout=use_dropout,
                                  dropout_rate=dropout_rate)

            generator = FFW_generator(units=[[48], [48]],
                                    use_norm=use_norm,
                                    use_dropout=use_dropout,
                                      dropout_rate=dropout_rate,
                                      projection=projection_layer)
            from Models import forecaster_model
            self.model = forecaster_model(encoder_block=encoder,
                                     decoder_block=generator,
                                    use_teacher=True,
                                     output_shape=out_shape,
                                     support_shape=support_shape,
                                     history_shape=history_shape)
        elif model_type == 'Densegenerator' or model_type == 'DenseGenerator':
            print('building E-D')
            common_specs = {'attention_heads': 5,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'L1': L1, 'L2': L2,
                            'use_norm': use_norm}
            growth = 18
            encoder_specs = copy.deepcopy(common_specs)
            encoder_specs['units'] = [[growth, growth, growth, growth, growth, 6*growth],
                                      [growth, growth, growth, growth, growth, 9*growth],
                                      [growth, growth, growth, growth, growth, 12*growth],
                                      ]
            decoder_specs = copy.deepcopy(common_specs)
            decoder_specs['units'] = [[growth, growth, growth, growth, growth, 5*growth],
                                      [growth, growth, growth, growth, growth, 5*growth],
                                      ]

            decoder_specs['projection'] = tf.keras.layers.Conv1D(filters=out_shape[-1],
                                                kernel_size=1,
                                                strides=1,
                                                padding='causal',
                                                activation=tf.keras.layers.Softmax(axis=-1),
                                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                kernel_initializer='glorot_uniform')

            from Building_Blocks import fixed_attentive_TCN, fixed_generator_Dense_block
            from Models import forecaster_model
            self.model = forecaster_model(encoder_block=fixed_attentive_TCN(**encoder_specs),
                                     decoder_block=fixed_generator_Dense_block(**decoder_specs),
                                    use_teacher=True,
                                     output_shape=out_shape,
                                     support_shape=support_shape,
                                     history_shape=history_shape)

        else:
            print('trying to build with', model_size, model_type, 'but failed')
        from Losses_and_Metrics import loss_wrapper
        # learning_rate = tf.keras.experimental.CosineDecayRestarts(3*1e-3,
        #                                                         20*self.train_steps_epr_epoch,
        #                                                         t_mul=1.0,
        #                                                         m_mul=1.0,
        #                                                         alpha=1e-5,
        #                                                         name=None
        #                                                     )

        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=3*1e-3,
                                                              #clipnorm=1.0,
                                                              ),
                    # loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
                    loss=loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='KL_D'),
                      metrics=[loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nME',
                                            normalizer_value=normalizer_value),
                               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nRMSE',
                                            normalizer_value=normalizer_value),
                               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='CRPS'),
                               ],
                           )  # compile, print summary
        self.model.summary()

    def __train_model(self, batch_size=64):
        self.metrics = {}
        callbacks = []
        epochs = 500
        # def scheduler(epoch):
        #     max_lr = 3*1e-2
        #     min_lr = 5*1e-4
        #     up_down_length=40
        #
        #     where_in_cycle = epoch%up_down_length
        #     if where_in_cycle <= up_down_length/2:
        #         alpha = min_lr + where_in_cycle*(max_lr - min_lr)/(up_down_length/2)
        #         return alpha
        #     if where_in_cycle > up_down_length/2:
        #         alpha = max_lr - where_in_cycle*(max_lr - min_lr)/(up_down_length/2)
        #         return alpha

        logdir =  os.path.join("tboard_logs")
        print('copy paste for tboard:', logdir)
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_nRMSE',
                                                                   patience=epochs,
                                                                   mode='min',
                                                                   restore_best_weights=True))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                        write_graph=False,
                                                        #update_freq='epoch',
                                                        ))

        #callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
        # callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
        #                                                        factor=2,
        #                                                        patience=20,
        #                                                       min_delta=0.05,
        #                                                        verbose=True,
        #                                                        mode='min',
        #                                                        cooldown=5))
        print('starting to train model')
        train_history = self.model.fit(self.train_dataset_generator(),
                                       steps_per_epoch=self.train_steps_epr_epoch,
                                      epochs=epochs,
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
        saved_epoch = np.argmin(self.metrics['val_nRMSE'])
        self.metrics['val_nRMSE_skill'] = 1 - (self.metrics['val_nRMSE'][saved_epoch] / self.dataset_info['val_baseline']['nRMSE'])
        self.metrics['val_CRPS_skill'] = 1 - (self.metrics['val_CRPS'][saved_epoch] / self.dataset_info['val_baseline']['CRPS'])
        self.metrics['test_nRMSE_skill'] = 1 - (self.metrics['test_nRMSE'] / self.dataset_info['test_baseline']['nRMSE'])
        self.metrics['test_CRPS_skill'] = 1 - (self.metrics['test_CRPS'] / self.dataset_info['test_baseline']['CRPS'])
        print('val_skill nRMSE', self.metrics['val_nRMSE_skill'] )
        print('val_skill CRPS', self.metrics['val_CRPS_skill'])
        print('test_skill nRMSE', self.metrics['test_nRMSE_skill'])
        print('test_skill CRPS', self.metrics['test_CRPS_skill'])


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









