# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf
import numpy as np
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
                 sw_len_days=3,
                 try_distribution_across_GPUs=True,
                 ):
        self.dataset_path = dataset_folder
        self.experiment_name = experiment_name
        #if try_distribution_across_GPUs:

        self.strategy = tf.distribute.MirroredStrategy()
        self.forecast_mode = model_kwargs['forecast_mode']
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
        elif 'Generator' in model_kwargs['model_type']:
            self.len_history = int((sw_len_days-1)*24)
            self.len_nwp = int(sw_len_days*24*60/15)
            model_kwargs['support_shape'] = [self.len_nwp, self.dataset_info['nwp_dims']]
            model_kwargs['history_shape'] = [self.len_history, self.fc_tiles]


        model_kwargs['out_shape'] = self.target_shape

        model_kwargs['normalizer_value'] = self.dataset_info['normalizer_value']

        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs

    def get_results(self, runs=1):
        tf.keras.backend.clear_session() # make sure we are working clean

        self.__create_datasets()
        self.metrics = {}
        # cross_ops = tf.distribute.ReductionToOneDevice()
        cross_ops = tf.distribute.HierarchicalCopyAllReduce()
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_ops)

        for run in range(runs):
            tf.keras.backend.clear_session()
            with strategy.scope():
                self.__build_model(**self.model_kwargs)

                train_history, test_results = self.__train_model()
                del self.model

                self.metrics[run] = {}

                self.__manage_metrics(train_history, test_results)
        del self.train_kwargs



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
        pdf_output_shape = self.target_shape
        flattened_pdf_output_shape = 1
        for dimension_shape in pdf_output_shape:
            flattened_pdf_output_shape = flattened_pdf_output_shape * dimension_shape

        ev_output_shape = [self.target_shape[0], 1]
        flattened_ev_output_shape = 1
        for dimension_shape in ev_output_shape:
            flattened_ev_output_shape = flattened_ev_output_shape * dimension_shape

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
            def __calculate_expected_value(signal, last_output_dim_size):
                indices = tf.range(last_output_dim_size, dtype=tf.float32)  # (last_output_dim_size)
                weighted_signal = tf.multiply(signal, indices)  # (batches, timesteps, last_output_dim_size)
                expected_value = tf.reduce_sum(weighted_signal, axis=-1, keepdims=True)
                return expected_value / last_output_dim_size

            features = {'nwp_input': tf.io.FixedLenFeature(flattened_nwp_shape, tf.float32),
                        'raw_historical_input': tf.io.FixedLenFeature(pv_input_shape, tf.float32),
                        'pdf_historical_input': tf.io.FixedLenFeature(self.pdf_history_shape[0] * self.pdf_history_shape[1], tf.float32),
                        'pdf_target': tf.io.FixedLenFeature(flattened_pdf_output_shape, tf.float32),
                        'pdf_teacher': tf.io.FixedLenFeature(flattened_pdf_output_shape, tf.float32),
                        #'raw_target': tf.io.FixedLenFeature(flattened_ev_output_shape, tf.float32),
                        #'raw_teacher': tf.io.FixedLenFeature(flattened_ev_output_shape, tf.float32),
                        }

            unadjusted_example = tf.io.parse_single_example(example, features)
            nwp_inputs = tf.reshape(tensor=unadjusted_example['nwp_input'], shape=nwp_shape)
            teacher = tf.reshape(tensor=unadjusted_example['pdf_teacher'],
                                 shape=pdf_output_shape)
            target = tf.reshape(tensor=unadjusted_example['pdf_target'], shape=pdf_output_shape)
            history_pdf = tf.reshape(tensor=unadjusted_example['pdf_historical_input'],
                                     shape=self.pdf_history_shape)

            if self.forecast_mode == 'ev':
                target = __calculate_expected_value(target, pdf_output_shape[-1])
                target = target * self.dataset_info['normalizer_value']
                teacher = __calculate_expected_value(teacher, pdf_output_shape[-1])
                teacher = teacher * self.dataset_info['normalizer_value']


            if 'Generator' in self.model_kwargs['model_type']:
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
                      forecast_mode='pdf',
                      model_type='MiMo-sth',
                      encoder_units=[[96],[96],[96]],
                      decoder_units=[[64],[64],[64]],
                      L1=0.0, L2=0.0,
                      use_dropout=False, dropout_rate=0.0,
                      use_hw=False, use_norm=False, use_residual=False, use_dense=False,#general architecture stuff
                      use_attention=False, attention_heads=3,
                      downsample=False, mode='project', #MiMo stuff
                      encoder_blocks=3,
                      decoder_blocks=3,
                      positional_embedding=False,
                      force_relevant_context=True,
                      encoder_receptive_window=None,
                      decoder_receptive_window=None,
                      ):

        if self.forecast_mode == 'pdf':
            projection_block = tf.keras.layers.Conv1D(filters=out_shape[-1],
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='causal',
                                                    activation=tf.keras.layers.Softmax(axis=-1),
                                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                    kernel_initializer='glorot_uniform')
        elif self.forecast_mode =='ev':
            projection_block = tf.keras.layers.Conv1D(filters=1,
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='causal',
                                                    activation=None,
                                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2),
                                                    kernel_initializer='glorot_uniform')
        else:
            print('wrong forecast mode flag, must be either pdf or ev')

        if model_type == 'MiMo-LSTM':
            from Building_Blocks import block_LSTM
            encoder_specs = {'units': encoder_units,
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
                                projection_block=projection_block,
                               downsampling_rate=(60 / 5),
                               mode=mode)

        elif model_type == 'MiMo-FFNN':
            from Building_Blocks import FFW_block
            encoder_specs = {'units': encoder_units,
                             'use_dropout': use_dropout,
                             'dropout_rate': dropout_rate,
                             'use_norm': use_norm,}

            from Models import mimo_model
            self.model = mimo_model(function_block=FFW_block(**encoder_specs),
                               input_shape=input_shape,
                               output_shape=out_shape,
                               downsample_input=downsample,
                               downsampling_rate=(60 / 5),
                               projection_block = projection_block,
                               mode=mode)

        elif model_type == 'Encoder-Decoder' or model_type == 'E-D':
            from Building_Blocks import block_LSTM, decoder_LSTM_block
            from Models import S2S_model
            print('building E-D')
            common_specs = {'units': encoder_units,
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
            decoder_specs['projection_layer'] = projection_block
            decoder = decoder_LSTM_block(**decoder_specs)
            self.model = S2S_model(encoder_block=encoder,
                                   decoder_block=decoder,
                                   input_shape=input_shape,
                                   output_shape=[out_shape[0], 1] if self.forecast_mode =='ev' else out_shape)

        elif model_type == 'LSTM-Generator':
            from Building_Blocks import ForecasterModel
            print('building E-D')
            common_specs = {'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm, 'use_hw': use_hw, 'use_residual': use_residual,
                            'L1': L1, 'L2': L2,
                            }
            encoder_specs = copy.deepcopy(common_specs)
            encoder_specs['return_state'] = False
            encoder_specs['units'] = encoder_units
            decoder_specs = copy.deepcopy(common_specs)
            decoder_specs['units'] = decoder_units
            decoder_specs['use_attention'] = use_attention
            decoder_specs['attention_heads'] = attention_heads
            decoder_specs['projection_layer'] = projection_block

            self.model = ForecasterModel(output_shape=out_shape,
                                        encoder_specs=encoder_specs,
                                        decoder_specs=decoder_specs,
                                        model_type='LSTM')

        elif model_type=='CNN-Generator':
            decoder_specs = {'num_initial_features': decoder_units,
                             'sequence_length': decoder_receptive_window,
                             'attention_heads': attention_heads,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'attention_squeeze': 0.5,
                             'positional_embedding': positional_embedding,
                             'force_relevant_context': force_relevant_context,
                             'projection_layer': projection_block}
            encoder_specs = {'num_initial_features': encoder_units,
                             'sequence_length': encoder_receptive_window,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'attention_heads': attention_heads,
                              'positional_embedding': positional_embedding,
                              'force_relevant_context': force_relevant_context,
                             'attention_squeeze': 0.5}
            from Building_Blocks import ForecasterModel
            self.model = ForecasterModel(output_shape=out_shape,
                                        encoder_specs=encoder_specs,
                                        decoder_specs=decoder_specs,
                                        model_type=model_type)
        elif model_type == 'FFNN-Generator':
            from Building_Blocks import FFNN_encoder, FFNN_decoder
            # decoder: width=256, depth=3, attention_heads=3, norm=True, attention_squeeze=0.5, L1=0.0, L2=0.0, projection_layer=None)
            # encoder: width=256, depth=3, attention_heads=3, norm=True, attention_squeeze=0.5, L1=0.0, L2=0.0
            decoder_specs = {'width': decoder_units,
                             'depth': decoder_blocks,
                             'attention_heads': attention_heads,
                             'use_norm': use_norm,
                             'attention_squeeze': 0.5,
                             'L1':L1, 'L2': L2,
                             'projection_layer': projection_block}
            encoder_specs = {'width': encoder_units,
                             'depth': encoder_blocks,
                             'L1': L1, 'L2': L2,
                             'use_norm': use_norm,
                             'attention_heads': attention_heads,
                             'attention_squeeze': 0.5}
            from Building_Blocks import ForecasterModel
            self.model = ForecasterModel(output_shape=out_shape,
                                         encoder_specs=encoder_specs,
                                         decoder_specs=decoder_specs,
                                         model_type=model_type)
        elif model_type == 'TCN-Generator':
            from Building_Blocks import ForecasterModel
            print('building E-D')
            common_specs = {'units': encoder_units,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm, 
                            'L1': L1, 'L2': L2,
                            }

            encoder_specs = copy.deepcopy(common_specs)
            decoder_specs = copy.deepcopy(common_specs)
            decoder_specs['attention_heads'] = attention_heads
            decoder_specs['projection'] = projection_block

            self.model = ForecasterModel(output_shape=out_shape,
                                        encoder_specs=encoder_specs,
                                        decoder_specs=decoder_specs,
                                        model_type='TCN')

        else:
            print('trying to buid', model_type, 'but failed')

        from Losses_and_Metrics import loss_wrapper
        # assign the losses depending on scenario
        if self.forecast_mode == 'pdf':
            loss = loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='KL-D')
            metrics = [loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nME',
                                            normalizer_value=1.0,
                                            target_as_expected_value=False,
                                            forecast_as_expected_value=False),
                               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nRMSE',
                                            normalizer_value=1.0,
                                            target_as_expected_value=False,
                                            forecast_as_expected_value=False
                                            ),
                               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='CRPS',
                                            target_as_expected_value=False,
                                            forecast_as_expected_value=False
                                            ),
                               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='EMC',
                                            target_as_expected_value=False,
                                            forecast_as_expected_value=False
                                            )
                               ]
        elif self.forecast_mode =='ev':
            loss = loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='MSE',
                                target_as_expected_value=True,
                                forecast_as_expected_value=True
                                )
            metrics = [loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nME',
                                            normalizer_value=normalizer_value,
                                            target_as_expected_value=True,
                                            forecast_as_expected_value=True),
                               loss_wrapper(last_output_dim_size=out_shape[-1], loss_type='nRMSE',
                                            normalizer_value=normalizer_value,
                                            target_as_expected_value=True,
                                            forecast_as_expected_value=True
                                            )
                               ]
        else:
            print('forecast mode was not specified as either <pdf> or <ev>, no idea how it got this far but expect some issues!!')

        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = 1/self.train_steps_epr_epoch,
                                                            momentum=0.85,
                                                            nesterov=True,
                                                            # clipnorm=1.0,
                                                            ),
                        loss=loss,
                        metrics=metrics,)  # compile, print summary

    def __train_model(self):

        callbacks = []
        epochs = 1000

        logdir =  os.path.join(self.experiment_name)
        print('copy paste for tboard:', logdir)
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_nRMSE',
                                                                   patience=10,
                                                                   mode='min',
                                                                   restore_best_weights=True))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                        write_graph=False,
                                                        #update_freq='epoch',
                                                        ))

        # callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
        #                                                        factor=0.2,
        #                                                        patience=5,
        #                                                       min_lr=1e-6,
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
        test_results = self.model.evaluate(self.test_dataset_generator(),
                                           steps=self.test_steps_epr_epoch,
                                            verbose=2)
        self.model.summary()

        return train_history, test_results



    def __manage_metrics(self, train_history, test_results):

        last_run_dict = {}
        for key in train_history.keys():
            last_run_dict[key] = train_history[key]
        last_run_dict['test_loss'] = test_results[0]
        last_run_dict['test_nME'] = test_results[1]
        last_run_dict['test_nRMSE'] = test_results[2]

        if self.forecast_mode == 'pdf':
            last_run_dict['test_CRPS'] = test_results[3]

        saved_epoch = np.argmin(last_run_dict['val_nRMSE'])
        last_run_dict['val_nRMSE_skill'] = 1 - (last_run_dict['val_nRMSE'][saved_epoch] / self.dataset_info['val_baseline']['nRMSE'])
        last_run_dict['test_nRMSE_skill'] = 1 - (last_run_dict['test_nRMSE'] / self.dataset_info['test_baseline']['nRMSE'])

        if self.forecast_mode == 'pdf':
            last_run_dict['val_CRPS_skill'] = 1 - (last_run_dict['val_CRPS'][saved_epoch] / self.dataset_info['val_baseline']['CRPS'])
            last_run_dict['test_CRPS_skill'] = 1 - (last_run_dict['test_CRPS'] / self.dataset_info['test_baseline']['CRPS'])

        print('val_skill nRMSE', last_run_dict['val_nRMSE_skill'])
        print('test_skill nRMSE', last_run_dict['test_nRMSE_skill'])

        if self.forecast_mode == 'pdf':
            print('val_skill CRPS', last_run_dict['val_CRPS_skill'])
            print('test_skill CRPS', last_run_dict['test_CRPS_skill'])

        last_run = [*self.metrics][-1]
        self.metrics[last_run] = last_run_dict

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









