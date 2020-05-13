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

            model_kwargs['input_shape'] = [int(self.sw_len_samples / self.nwp_downsampling_rate),
                                       self.raw_nwp_shape[1] + 1]
            # TODO: this is not required, only becasue it asks for it later
            model_kwargs['support_shape'] = [int(3*24*60/15), self.dataset_info['nwp_dims']]
            model_kwargs['history_shape'] = [int(2*24), self.fc_tiles]

        elif 'MiMo' in model_kwargs['model_type']:
            model_kwargs['input_shape'] = [int(self.sw_len_samples / self.nwp_downsampling_rate),
                                       self.raw_nwp_shape[1] + 1]

        elif 'Generator' in model_kwargs['model_type']:
            self.len_history = int((sw_len_days-1)*24)
            self.len_nwp = int(sw_len_days*24*60/15)
            model_kwargs['support_shape'] = [self.len_nwp, self.dataset_info['nwp_dims']]
            model_kwargs['history_shape'] = [self.len_history, self.fc_tiles]


        model_kwargs['out_shape'] = self.target_shape

        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs

    def get_results(self):
        tf.keras.backend.clear_session() # make sure we are working clean

        self.metrics = {}
        # cross_ops = tf.distribute.ReductionToOneDevice()
        cross_ops = tf.distribute.HierarchicalCopyAllReduce()
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_ops)

        with strategy.scope():
            self.__build_model(**self.model_kwargs)

            train_history, test_results = self.__train_model()
            del self.model

        tf.keras.backend.clear_session()
        results_dict = self.__manage_metrics(train_history, test_results)
        del self.train_kwargs

        return results_dict

    def __build_model(self,
                      out_shape,
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
                      encoder_max_length_sequence=None,
                      encoder_receptive_window=None,
                      encoder_self_attention=False,
                      encoder_transformer_blocks=1,
                      decoder_max_length_sequence=None,
                      decoder_receptive_window=None,
                      decoder_self_attention=False,
                      decoder_transformer_blocks=1,
                      ):

        if self.forecast_mode == 'pdf':
            projection_block = tf.keras.layers.Conv1D(filters=self.target_shape[-1],
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
                               output_shape=self.target_shape,
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
                               output_shape=self.target_shape,
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
                                   output_shape=[self.target_shape[0], 1] if self.forecast_mode =='ev' else self.target_shape)

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

            self.model = ForecasterModel(output_shape=self.target_shape,
                                        encoder_specs=encoder_specs,
                                        decoder_specs=decoder_specs,
                                        model_type='LSTM')

        elif model_type=='CNN-Generator':
            decoder_specs = {'num_initial_features': decoder_units,
                             'max_length_sequence': decoder_max_length_sequence,
                             'length_receptive_window': decoder_receptive_window,
                             'attention_heads': attention_heads,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'force_relevant_context': force_relevant_context,
                             'attention_squeeze': 0.5,
                             'use_self_attention': decoder_self_attention,
                             'transformer_blocks': decoder_transformer_blocks,
                             'positional_embedding': positional_embedding,
                             'projection_layer': projection_block}
            encoder_specs = {'num_initial_features': encoder_units,
                             'max_length_sequence': encoder_max_length_sequence,
                             'length_receptive_window': encoder_receptive_window,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'force_relevant_context': force_relevant_context,
                             'attention_heads': attention_heads,
                             'use_self_attention': encoder_self_attention,
                             'transformer_blocks': encoder_transformer_blocks,
                              'positional_embedding': positional_embedding,
                             'attention_squeeze': 0.5}
            from Building_Blocks import ForecasterModel
            self.model = ForecasterModel(output_shape=self.target_shape,
                                        encoder_specs=encoder_specs,
                                        decoder_specs=decoder_specs,
                                        model_type=model_type)
        elif model_type == 'FFNN-Generator':
            from Building_Blocks import FFNN_encoder, FFNN_decoder
            # decoder: width=256, depth=3, attention_heads=3, norm=True, attention_squeeze=0.5, L1=0.0, L2=0.0, projection_layer=None)
            # encoder: width=256, depth=3, attention_heads=3, norm=True, attention_squeeze=0.5, L1=0.0, L2=0.0
            decoder_specs = {'num_initial_features': decoder_units,
                             'max_length_sequence': decoder_max_length_sequence,
                             'attention_heads': attention_heads,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'force_relevant_context': force_relevant_context,
                             'attention_squeeze': 0.5,
                             'use_self_attention': decoder_self_attention,
                             'transformer_blocks': decoder_transformer_blocks,
                             'positional_embedding': positional_embedding,
                             'projection_layer': projection_block}
            encoder_specs = {'num_initial_features': encoder_units,
                             'max_length_sequence': encoder_max_length_sequence,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'force_relevant_context': force_relevant_context,
                             'attention_heads': attention_heads,
                             'use_self_attention': encoder_self_attention,
                             'transformer_blocks': encoder_transformer_blocks,
                              'positional_embedding': positional_embedding,
                             'attention_squeeze': 0.5}
            from Building_Blocks import ForecasterModel
            self.model = ForecasterModel(output_shape=self.target_shape,
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

            self.model = ForecasterModel(output_shape=self.target_shape,
                                        encoder_specs=encoder_specs,
                                        decoder_specs=decoder_specs,
                                        model_type='TCN')

        else:
            print('trying to buid', model_type, 'but failed')

    def get_losses_and_metrics(self):
        from Losses_and_Metrics import loss_wrapper
        # assign the losses depending on scenario
        if self.forecast_mode == 'pdf':
            loss = loss_wrapper(last_output_dim_size=self.target_shape[-1], loss_type='KL-D')
            metrics = [loss_wrapper(last_output_dim_size=self.target_shape[-1], loss_type='nME',
                                    normalizer_value=1.0,
                                    target_as_expected_value=False,
                                    forecast_as_expected_value=False),
                       loss_wrapper(last_output_dim_size=self.target_shape[-1], loss_type='nRMSE',
                                    normalizer_value=1.0,
                                    target_as_expected_value=False,
                                    forecast_as_expected_value=False
                                    ),
                       loss_wrapper(last_output_dim_size=self.target_shape[-1], loss_type='CRPS',
                                    target_as_expected_value=False,
                                    forecast_as_expected_value=False
                                    ),
                       loss_wrapper(last_output_dim_size=self.target_shape[-1], loss_type='EMC',
                                    target_as_expected_value=False,
                                    forecast_as_expected_value=False
                                    )
                       ]
            return loss, metrics

        elif self.forecast_mode == 'ev':
            loss = loss_wrapper(last_output_dim_size=self.target_shape[-1], loss_type='MSE',
                                target_as_expected_value=True,
                                forecast_as_expected_value=True
                                )
            metrics = [loss_wrapper(last_output_dim_size=self.target_shape[-1], loss_type='nME',
                                    normalizer_value=self.dataset_info['normalizer_value'],
                                    target_as_expected_value=True,
                                    forecast_as_expected_value=True),
                       loss_wrapper(last_output_dim_size=self.target_shape[-1], loss_type='nRMSE',
                                    normalizer_value=self.dataset_info['normalizer_value'],
                                    target_as_expected_value=True,
                                    forecast_as_expected_value=True
                                    )
                       ]
            return loss, metrics
        else:
            print(
                'forecast mode was not specified as either <pdf> or <ev>, no idea how it got this far but expect some issues!!')

    def __train_model(self):

        callbacks = []
        epochs = 1000
        # ToDo: change back to '/train'
        PV_dataset = dataset_generator_PV(dataset_path=self.dataset_path,
                              train_batch_size=self.train_kwargs['batch_size'],
                              support_shape=self.model_kwargs['support_shape'],
                              history_shape=self.model_kwargs['history_shape'],
                              raw_history_shape=[self.nwp_downsampling_rate, int(self.sw_len_samples/self.nwp_downsampling_rate)],
                              val_target_shape=self.target_shape,
                              )

        if 'Generator' in self.model_kwargs['model_type']:
            train_set = PV_dataset.pdf_generator_training_dataset
            val_set = PV_dataset.pdf_generator_val_dataset
            test_set = PV_dataset.pdf_generator_test_dataset

        train_steps = PV_dataset.get_train_steps_per_epoch()
        val_steps = PV_dataset.get_train_steps_per_epoch()
        test_steps = PV_dataset.get_train_steps_per_epoch()

        # Transformer LR schedule, doesnt work .... too fast
        learning_rate_schedule = CustomSchedule(128, warmup_steps=4000)
        optimizer = tf.keras.optimizers.Adam(learning_rate_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        loss, metrics = self.get_losses_and_metrics()
        # learning_rate = np.sqrt(1/train_steps)
        # optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate,
        #                                      momentum=0.9,
        #                                      nesterov=True,
        #                                     # clipnorm=1.0, # Apparently some works do clipnorm
        #                                     )

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)  # compile, print summary

        logdir =  os.path.join(self.experiment_name)
        print('copy paste for tboard:', logdir)
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_nRMSE',
                                                                   patience=100,
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
        train_history = self.model.fit(train_set(),
                                        steps_per_epoch=train_steps,
                                        epochs=epochs,
                                        verbose=2,
                                        validation_data=val_set(),
                                        validation_steps=val_steps,
                                        callbacks=callbacks)
        train_history = train_history.history
        test_results = self.model.evaluate(test_set(),
                                           steps=test_steps,
                                            verbose=2)
        self.model.summary()

        return train_history, test_results



    def __manage_metrics(self, train_history, test_results):

        results_dict = {}
        results_dict['test_loss'] = test_results[0]
        results_dict['test_nME'] = test_results[1]
        results_dict['test_nRMSE'] = test_results[2]
        if self.forecast_mode == 'pdf':
            results_dict['test_CRPS'] = test_results[3]

        for key in train_history.keys():
            results_dict[key] = train_history[key]

        saved_epoch = np.argmin(results_dict['val_nRMSE'])
        results_dict['save_epoch'] = saved_epoch
        results_dict['val_nRMSE_skill'] = 1 - (results_dict['val_nRMSE'][saved_epoch] / self.dataset_info['val_baseline']['nRMSE'])
        results_dict['test_nRMSE_skill'] = 1 - (results_dict['test_nRMSE'] / self.dataset_info['test_baseline']['nRMSE'])

        if self.forecast_mode == 'pdf':
            results_dict['val_CRPS_skill'] = 1 - (results_dict['val_CRPS'][saved_epoch] / self.dataset_info['val_baseline']['CRPS'])
            results_dict['test_CRPS_skill'] = 1 - (results_dict['test_CRPS'] / self.dataset_info['test_baseline']['CRPS'])

        print('val_skill nRMSE', results_dict['val_nRMSE_skill'])
        print('test_skill nRMSE', results_dict['test_nRMSE_skill'])

        if self.forecast_mode == 'pdf':
            print('val_skill CRPS', results_dict['val_CRPS_skill'])
            print('test_skill CRPS', results_dict['test_CRPS_skill'])

        return results_dict

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

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# ToDo: Rewrite this to be neater and work more flexibly, prepare for ragged shit
class dataset_generator_PV():
    def __init__(self,
                 dataset_path=None,
                 train_batch_size=None,
                 support_shape=None,
                 history_shape=None,
                 raw_history_shape=None,
                 train_target_shape=None,
                 val_target_shape=None):

        # We keep the training size as large as possible, this means the val size needs to be smaller due to memory thingies!
        self.train_batch_size = train_batch_size
        self.val_batch_size = int(train_batch_size / 5)

        self.support_shape = support_shape
        self.history_shape = history_shape
        self.raw_history_shape = raw_history_shape
        self.train_target_shape = train_target_shape if train_target_shape is not None else self.history_shape
        self.val_target_shape = val_target_shape

        self.flattened_nwp_shape = tf.reduce_prod(self.support_shape).numpy()
        self.flattened_historical_shape = tf.reduce_prod(self.history_shape).numpy()
        self.flattened_val_targets_shape = tf.reduce_prod(self.val_target_shape).numpy()


        self.train_folder = dataset_path + '/train'
        self.val_folder = dataset_path + '/validation'
        self.test_folder = dataset_path + '/test'

    def __get_all_tfrecords_in_folder(self, folder):

        if os.path.isdir(folder):
            file_list = glob.glob(folder + '/*.tfrecord')
            shuffle(file_list)
            return file_list
        else:
            print('didnt find training data folder, expect issues!!')


    def get_train_steps_per_epoch(self):
        train_list = self.__get_all_tfrecords_in_folder(self.train_folder)
        return int(len(train_list) / self.train_batch_size)

    def get_val_steps_per_epoch(self):
        val_list = self.__get_all_tfrecords_in_folder(self.val_folder)
        return int(np.ceil(len(val_list) / self.val_batch_size))

    def get_test_steps_epr_epoch(self):
        test_list = self.__get_all_tfrecords_in_folder(self.val_folder)
        return int(np.ceil(len(test_list) / self.val_batch_size))

    # Turn this into get the different types of datasets for generator etc

    def pdf_generator_training_dataset(self):
        return self.__dataset_from_folder_and_sample(file_list=self.__get_all_tfrecords_in_folder(self.train_folder),
                                                     batch_size=self.train_batch_size,
                                                     process_sample=self.get_pdf_generator_train_sample)

    def pdf_generator_val_dataset(self):
        return self.__dataset_from_folder_and_sample(file_list=self.__get_all_tfrecords_in_folder(self.val_folder),
                                                     batch_size=self.val_batch_size,
                                                     process_sample=self.get_pdf_generator_inference_sample)

    def pdf_generator_test_dataset(self):
        return self.__dataset_from_folder_and_sample(file_list=self.__get_all_tfrecords_in_folder(self.test_folder),
                                                  batch_size=self.val_batch_size,
                                                  process_sample=self.get_pdf_generator_inference_sample)


    def get_pdf_generator_inference_sample(self, example):

        features = {'nwp_input': tf.io.FixedLenFeature(self.flattened_nwp_shape, tf.float32),
                    'pdf_historical_input': tf.io.FixedLenFeature(self.flattened_historical_shape, tf.float32),
                    'pdf_target': tf.io.FixedLenFeature(self.flattened_val_targets_shape, tf.float32),
                    'pdf_teacher': tf.io.FixedLenFeature(self.flattened_val_targets_shape, tf.float32)
                    }

        raw_unprocessed_sample = tf.io.parse_single_example(example, features)
        support_data = tf.reshape(tensor=raw_unprocessed_sample['nwp_input'], shape=self.support_shape)
        target = tf.reshape(tensor=raw_unprocessed_sample['pdf_target'], shape=self.val_target_shape)
        teacher = tf.reshape(tensor=raw_unprocessed_sample['pdf_teacher'],
                             shape=self.val_target_shape)
        history_pdf = tf.reshape(tensor=raw_unprocessed_sample['pdf_historical_input'],
                                 shape=self.history_shape)

        # ATM sampling last 2days plus forecast day from NWP and last 2 days from history
        return {'support_input': support_data,
                'history_input': history_pdf,
                'teacher': teacher}, target

    def get_pdf_generator_train_sample(self, example):


        features = {'nwp_input': tf.io.FixedLenFeature(self.flattened_nwp_shape, tf.float32),
                    'pdf_historical_input': tf.io.FixedLenFeature(self.flattened_historical_shape, tf.float32),
                    'pdf_target': tf.io.FixedLenFeature(self.flattened_val_targets_shape, tf.float32),
                    'pdf_teacher': tf.io.FixedLenFeature(self.flattened_val_targets_shape, tf.float32),
                    }

        raw_unprocessed_sample = tf.io.parse_single_example(example, features)

        support_data = tf.reshape(tensor=raw_unprocessed_sample['nwp_input'], shape=self.support_shape)
        history_pdf = tf.reshape(tensor=raw_unprocessed_sample['pdf_historical_input'],
                                 shape=self.history_shape)
        teacher = tf.reshape(tensor=raw_unprocessed_sample['pdf_teacher'],
                             shape=self.val_target_shape)

        target = tf.reshape(tensor=raw_unprocessed_sample['pdf_target'], shape=self.val_target_shape)
        target = tf.concat([history_pdf[1:,:], target], axis=0)

        # ATM sampling last 2days plus forecast day from NWP and last 2 days from history
        return {'support_input': support_data,
                'history_input': history_pdf,
                'teacher': teacher}, target

    def get_S2S_train_sample(self, example):
        pass

    def __calculate_expected_value(self, signal, last_output_dim_size):
        indices = tf.range(last_output_dim_size)  # (last_output_dim_size)
        weighted_signal = tf.multiply(signal, indices)  # (batches, timesteps, last_output_dim_size)
        expected_value = tf.reduce_sum(weighted_signal, axis=-1, keepdims=True)
        return expected_value / last_output_dim_size

    def __dataset_from_folder_and_sample(self, process_sample, file_list, batch_size):

        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = False

        # dataset = tf.data.Dataset.list_files(file_list)
        dataset = tf.data.TFRecordDataset(file_list, num_parallel_reads=6)
        dataset = dataset.with_options(option_no_order)

        dataset = dataset.map(process_sample, num_parallel_calls=6)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10 * batch_size)
        dataset = dataset.prefetch(5 * batch_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset

    # def __read_and_process_normal_samples(self, example):
    #     # 'nwp_input':
    #     # 'raw_historical_input':
    #     # 'raw_teacher':
    #     # 'raw_target':
    #     # 'pdf_historical_input':
    #     # 'pdf_teacher':
    #     # 'pdf_target':
    #
    #     flattened_nwp_shape = np.multiply(self.support_shape)
    #     pdf_output_shape = np.multiply(self.val_target_shape)
    #     flattened_historical_shape = np.multiply(self.history_shape)
    #     flattened_val_targets_shape = np.multiply(self.val_target_shape)
    #     flattened_raw_history_shape =np.multiply(self.raw_history_shape)
    #     ev_output_shape = self.val_target_shape
    #
    #     features = {'nwp_input': tf.io.FixedLenFeature(flattened_nwp_shape, tf.float32),
    #                 'raw_historical_input': tf.io.FixedLenFeature(flattened_raw_history_shape, tf.float32),
    #                 'pdf_historical_input': tf.io.FixedLenFeature(flattened_historical_shape, tf.float32),
    #                 'pdf_target': tf.io.FixedLenFeature(flattened_val_targets_shape, tf.float32),
    #                 'pdf_teacher': tf.io.FixedLenFeature(flattened_val_targets_shape, tf.float32),
    #                 #'raw_target': tf.io.FixedLenFeature(flattened_ev_output_shape, tf.float32),
    #                 #'raw_teacher': tf.io.FixedLenFeature(flattened_ev_output_shape, tf.float32),
    #                 }
    #
    #     unadjusted_example = tf.io.parse_single_example(example, features)
    #     support_data = tf.reshape(tensor=unadjusted_example['nwp_input'], shape=self.support_shape)
    #     teacher = tf.reshape(tensor=unadjusted_example['pdf_teacher'],
    #                          shape=pdf_output_shape)
    #     target = tf.reshape(tensor=unadjusted_example['pdf_target'], shape=pdf_output_shape)
    #     history_pdf = tf.reshape(tensor=unadjusted_example['pdf_historical_input'],
    #                              shape=self.pdf_history_shape)
    #
    #     if self.forecast_mode == 'ev':
    #         target = self.__calculate_expected_value(target, pdf_output_shape[-1])
    #         target = target * self.dataset_info['normalizer_value']
    #         teacher = self.__calculate_expected_value(teacher, pdf_output_shape[-1])
    #         teacher = teacher * self.dataset_info['normalizer_value']
    #
    #
    #     history_pv = tf.reshape(tensor=unadjusted_example['raw_historical_input'], shape=self.raw_history_shape)
    #     history_pv = tf.transpose(history_pv, [1,0])
    #     history_pv = tf.reduce_mean(history_pv, axis=-1, keepdims=True)
    #     nwp_inputs = support_data[int(self.fc_len_samples/self.nwp_downsampling_rate):,:]
    #     inputs =tf.concat([nwp_inputs, history_pv], axis=-1)
    #     if 'MiMo' in self.model_kwargs['model_type']:
    #         return inputs, target
    #     else:
    #         return {'encoder_inputs': inputs, 'teacher_inputs': teacher}, target
