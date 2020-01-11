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
                 experiment_name='Default_Name',
                 try_distribution_across_GPUs=True,
                 ):
        self.dataset_path = dataset_folder
        self.experiment_name = experiment_name
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

            history_pdf = tf.reshape(tensor=unadjusted_example['pdf_historical_input'],
                                     shape=[self.pdf_history_shape[0], self.pdf_history_shape[1]])
            teacher = tf.reshape(tensor=unadjusted_example['pdf_teacher'],
                                 shape=[output_shape[0], output_shape[1]])

            # ToDo: Current Dev Stack
            if 'generator' in self.model_kwargs['model_type']:
                print('yaaay', self.model_kwargs['model_type'], 'time!!!!')

                nwp_inputs = nwp_inputs[-real_support_length:,:]
                history_pdf = history_pdf[-real_history_length:,:]
                # ATM sampling last 2days plus forecast day from NWP and last 2 days from history
                return{'support_input': nwp_inputs,
                       'history_input': history_pdf,
                       'teacher_input': teacher}, target

            else:
                history_pv = tf.reshape(tensor=unadjusted_example['raw_historical_input'], shape=[self.nwp_downsampling_rate, int(self.sw_len_samples/self.nwp_downsampling_rate)])
                history_pv = tf.transpose(history_pv, [1,0])
                history_pv = tf.reduce_mean(history_pv, axis=-1, keepdims=True)
                nwp_inputs = nwp_inputs[int(self.fc_len_samples/self.nwp_downsampling_rate):,:]
                inputs =tf.concat([nwp_inputs, history_pv], axis=-1)
                if 'MiMo' in self.model_kwargs['model_type']:
                    return inputs, target
                else:
                    return {'encoder_inputs': inputs, 'teacher_inputs': teacher}, target

        def get_batched_dataset(file_list, batch_size):
            option_no_order = tf.data.Options()
            option_no_order.experimental_deterministic = False

            #dataset = tf.data.Dataset.list_files(file_list)
            dataset = tf.data.TFRecordDataset(file_list, num_parallel_reads=20)
            dataset = dataset.with_options(option_no_order)
            #dataset = dataset.interleave(map_func=tf.data.TFRecordDataset,
            #                              num_parallel_calls=20)

            # does this order make any fucking sense at all?!?!?!?

            dataset = dataset.map(__read_and_process_normal_samples, num_parallel_calls=20)
            dataset = dataset.repeat()
            dataset = dataset.shuffle(10 * batch_size)
            # dataset = dataset.prefetch(10 * batch_size)
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
            return get_batched_dataset(train_list, batch_size,)
        self.train_dataset_generator = get_training_dataset
        self.train_steps_epr_epoch = int(len(train_list)/batch_size)


        def get_val_dataset():
            return get_batched_dataset(val_list, batch_size)
        self.val_dataset_generator = get_val_dataset
        self.val_steps_epr_epoch = int(np.ceil(len(val_list) / batch_size))


        def get_test_dataset():
            return get_batched_dataset(test_list, batch_size)
        self.test_dataset_generator = get_test_dataset
        self.test_steps_epr_epoch = int(np.ceil(len(test_list) / batch_size))

    def __build_model(self,
                      out_shape, normalizer_value,
                      support_shape, history_shape,
                      input_shape=None,
                      model_type='MiMo-sth',
                      units=[[96],[96],[96]],
                      L1=0.0, L2=0.0,
                      use_dropout=False, dropout_rate=0.0,
                      use_hw=False, use_norm=False, use_residual=False, #general architecture stuff
                      use_attention=False, attention_heads=3,
                      downsample=False, mode='project', #MiMo stuff
                      ):


        if model_type == 'MiMo-LSTM':
            print('building a', units, model_type)
            from Building_Blocks import block_LSTM
            encoder_specs = {'units': units,
                             'use_dropout': use_dropout,
                             'dropout_rate': dropout_rate,
                             'use_norm': use_norm,
                             'use_hw': use_hw,
                             'return_state': False}

            from Models import mimo_model
            self.model = mimo_model(function_block=block_LSTM(**encoder_specs),
                               input_shape=input_shape,
                               output_shape=out_shape,
                               downsample_input=downsample,
                               downsampling_rate=(60 / 5),
                               mode=mode)
        elif model_type == 'MiMo-FFNN':
            print('building a', units, model_type)
            from Building_Blocks import FFW_block
            encoder_specs = {'units': units,
                             'use_dropout': use_dropout,
                             'dropout_rate': dropout_rate,
                             'use_norm': use_norm,}

            from Models import mimo_model
            self.model = mimo_model(function_block=FFW_block(**encoder_specs),
                               input_shape=input_shape,
                               output_shape=out_shape,
                               downsample_input=downsample,
                               downsampling_rate=(60 / 5),
                               mode=mode)

        elif model_type == 'Encoder-Decoder' or model_type == 'E-D':
            from Building_Blocks import block_LSTM, decoder_LSTM_block
            from Models import S2S_model
            print('building E-D')
            common_specs = {'units': units,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm, 'use_hw': use_hw, 'use_residual': use_residual,
                            'L1': L1, 'L2': L2,
                            }

            encoder_specs = copy.deepcopy(common_specs)
            encoder = block_LSTM(**encoder_specs)
            decoder_specs = copy.deepcopy(common_specs)
            decoder_specs['use_attention'] = use_attention
            decoder_specs['attention_heads'] = attention_heads
            decoder_specs['projection_layer'] = tf.keras.layers.Conv1D(filters=out_shape[-1],
                                                kernel_size=1,
                                                strides=1,
                                                padding='causal',
                                                activation=tf.keras.layers.Softmax(axis=-1),
                                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                kernel_initializer='glorot_uniform')
            decoder = decoder_LSTM_block(**decoder_specs)
            self.model = S2S_model(encoder_block=encoder,
                                   decoder_block=decoder,
                                   input_shape=input_shape,
                                   output_shape=out_shape)

        elif model_type == 'Densegenerator' or model_type == 'DenseGenerator':
            print('building E-D')
            common_specs = {'attention_heads': 5,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'L1': L1, 'L2': L2,
                            'use_norm': use_norm}
            growth = 16
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
            print('trying to buid', model_type, 'but failed')
        from Losses_and_Metrics import loss_wrapper

        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=3*1e-3,
                                                             momentum=0.75,
                                                             nesterov=True,
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
        epochs = 200

        logdir =  os.path.join(self.experiment_name)
        print('copy paste for tboard:', logdir)
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_nRMSE',
                                                                   patience=20,
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
        print('Val bets values: ',
              'Loss: ', self.metrics['val_loss'][saved_epoch], '///',
              'NRMSE: ', self.metrics['val_nRMSE'][saved_epoch], '///',
              'NME: ', self.metrics['val_nME'][saved_epoch],  '///',
              'CRPS: ', self.metrics['val_CRPS'][saved_epoch],)
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









