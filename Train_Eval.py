# dumb variation of the encoder decoder
# this will be using teacher forcing
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import copy
import glob
import os
import pickle
from random import shuffle
from Losses_and_Metrics import losses_and_metrics
from datetime import datetime
def check_dataset_length(dataset_path):
    if os.path.isdir(dataset_path):
        file_list = glob.glob(dataset_path + '/*.tfrecord')
        return len(file_list)

def aggregate_dataset_info(dataset_path_list):
    print('extracting dataset information')

    #prep help variables
    aggregated_dataset_info = {}
    accumulated_samples = {'train': 0, 'validation': 0, 'test': 0}
    new_samples = {'train': 0, 'validation': 0, 'test': 0}

    for dataset_path in dataset_path_list:
        dataset_info = pickle.load(open(dataset_path + '/dataset_info.pickle', 'rb'))
        for key in dataset_info.keys():
            if 'baseline' in key:
                # aggregate weighted averages by dataset sample size for baseline metrics, lets do this properly
                if key not in aggregated_dataset_info.keys():
                    aggregated_dataset_info[key] = dataset_info[key]

                    # get the name of the dataset
                    if 'train' in key:
                        set_type = 'train'
                    elif 'val' in key:
                        set_type ='validation'
                    elif 'test' in key:
                        set_type ='test'
                    else:
                        print('huh ... stumbled upon baseline info not belonging to either test, train or val: ', key)
                    accumulated_samples[set_type] = check_dataset_length(dataset_path + '/' + set_type)
                else:
                    # get the name of the dataset
                    if 'train' in key:
                        set_type = 'train'
                    elif 'val' in key:
                        set_type ='validation'
                    elif 'test' in key:
                        set_type ='test'
                    else:
                        print('huh ... stumbled upon baseline info not belonging to either test, train or val: ', key)
                    new_samples[set_type] = check_dataset_length(dataset_path + '/' + set_type)

                    for metric in dataset_info[key].keys():
                        aggregated_dataset_info[key][metric] = (accumulated_samples[set_type] * aggregated_dataset_info[key][metric]
                                                               + new_samples[set_type]*dataset_info[key][metric]) / (accumulated_samples[set_type] + new_samples[set_type])


                    accumulated_samples[set_type] = accumulated_samples[set_type] + new_samples[set_type]


            else:
                # other metrics, we should make sure those are consistent between datasets!
                if key not in aggregated_dataset_info.keys():
                    aggregated_dataset_info[key] = dataset_info[key]
                else:
                    if aggregated_dataset_info[key] != dataset_info[key]:
                        print('UhOh ... stumbled upon discrepancy between datasets')
                        print(key, 'aggregated holds ', aggregated_dataset_info[key], 'and new dataset', dataset_path, ' holds ', dataset_info[key])

    print(aggregated_dataset_info)
    return aggregated_dataset_info

class Model_Container():
    def __init__(self,
                 dataset_path_list,

                 model_kwargs, #see __build_model
                 train_kwargs, #currently only batch size
                 experiment_name='Default_Name',
                 sw_len_days=3, #deprecated
                 try_distribution_across_GPUs=True,
                 ):
        self.dataset_path_list = dataset_path_list
        self.experiment_name = experiment_name

        self.forecast_mode = model_kwargs['forecast_mode']
        self.dataset_info = aggregate_dataset_info(dataset_path_list)

        self.fc_steps = self.dataset_info['fc_steps']
        self.fc_tiles = self.dataset_info['fc_tiles']
        self.nwp_dims = self.dataset_info['support_shape'][-1]
        # self.nwp_dims = 16
        self.teacher_shape = [self.fc_steps, self.fc_tiles]
        self.target_shape = [self.fc_steps, self.fc_tiles]
        self.pdf_history_shape = self.dataset_info['pdf_history_shape']
        self.raw_history_shape = self.dataset_info['raw_history_shape']

        model_kwargs['support_shape'] = self.dataset_info['support_shape']
        model_kwargs['history_shape'] = self.dataset_info['pdf_history_shape']
        model_kwargs['out_shape'] = self.target_shape
        if model_kwargs['model_type'] == 'Encoder-Decoder' or model_kwargs['model_type'] == 'E-D' or 'MiMo' in model_kwargs['model_type'] or 'E-D' in model_kwargs['model_type'] or model_kwargs['model_type'] == 'Transformer':

            model_kwargs['input_shape'] = self.dataset_info['support_shape']
            model_kwargs['input_shape'] = (model_kwargs['input_shape'][0], model_kwargs['input_shape'][-1] + 1) #since we have to add the PV dimension

        elif 'MiMo' in model_kwargs['model_type']:
            model_kwargs['input_shape'] = self.dataset_info['support_shape']
            model_kwargs['input_shape'][-1] = model_kwargs['input_shape'][-1] + 1 #since we have to add th

        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs
        self.folder_name = 'models/' + self.experiment_name

    def train(self):
        tf.keras.backend.clear_session() # make sure we are working clean

        self.metrics = {}
        # # cross_ops = tf.distribute.ReductionToOneDevice()
        # cross_ops = tf.distribute.HierarchicalCopyAllReduce()
        # strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_ops)
        #
        # with strategy.scope():


        self.__build_model(**self.model_kwargs)

        if self.train_kwargs['mode'] == 'fine-tune':
            pretrain_folder = self.folder_name + '-pre-trained' # to extract the weights
            self.experiment_name = self.experiment_name + '-fine-tuned' + self.dataset_path_list[0] # to save tboard logs
            self.folder_name = self.folder_name + '-fine-tuned' + self.dataset_path_list[0] # to save the fine-tuned model
            if not os.path.isdir('./' + pretrain_folder): # if there are no weights in the folder
                print('There is an error with finding model checkpoint. Folder ', pretrain_folder, 'does not exist.')
            else:
                print('...Loading model weights from checkpoint...', pretrain_folder)
                self.model.load_weights(pretrain_folder + "/model_ckpt")

        elif self.train_kwargs['mode'] == 'pre-train':
            self.experiment_name = self.experiment_name + '-pre-trained' # to save tboard logs
            self.folder_name = self.folder_name + '-pre-trained' # to save pre-trained model

        train_history, test_results = self.__train_model()

        print('Saving model to ...', self.folder_name)
        self.model.save_weights(self.folder_name + "/model_ckpt")

        del self.model

        tf.keras.backend.clear_session()
        results_dict = self.__manage_metrics(train_history, test_results)
        del self.train_kwargs

        return results_dict

    # ToDo: maybe build the model separately before when we call the experiment, and then in fine tune load the weights and go from there?

    def test(self):
        tf.keras.backend.clear_session() # make sure we are working clean

        self.metrics = {}
        # # cross_ops = tf.distribute.ReductionToOneDevice()
        # cross_ops = tf.distribute.HierarchicalCopyAllReduce()
        # strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_ops)
        #
        # with strategy.scope():

        pretrain_folder = self.folder_name + '-pre-trained' # to extract the weights

        if not os.path.isdir('./'+pretrain_folder): # if there are no weights in the folder 
            print('There is an error with finding model checkpoint. Folder', pretrain_folder, 'or', self.folder_name, 'does not exist.')

        self.__build_model(**self.model_kwargs)

        print('...Loading model weights from checkpoint...', pretrain_folder)
        self.model.load_weights(pretrain_folder + "/model_ckpt").expect_partial()

        self.experiment_name = self.experiment_name + 'test' # to save tboard logs

        test_results = self.__test_model()
        del self.model
        tf.keras.backend.clear_session()

        results_dict = self.__manage_test_metrics(test_results)
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
                      decoder_attention=False,
                      decoder_transformer_blocks=1,
                      full_targets=True
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
            print('building E-D')
            encoder_specs = {'units': encoder_units,
                            'num_encoder_blocks': encoder_transformer_blocks,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm,
                            'use_hw': use_hw,
                            'use_residual': use_residual,
                            'L1': L1, 'L2': L2}
            decoder_specs = {'units': decoder_units,
                            'num_decoder_blocks': decoder_transformer_blocks,
                            'use_dropout': use_dropout,
                            'dropout_rate': dropout_rate,
                            'use_norm': use_norm,
                            'use_hw': use_hw,
                            'use_residual': use_residual,
                            'L1': L1, 'L2': L2,
                            'use_attention': decoder_attention,
                            'attention_heads': attention_heads,
                            'projection_layer': projection_block}

            from Building_Blocks import S2SModel
            self.model = S2SModel(output_shape=self.target_shape,
                                encoder_specs=encoder_specs,
                                decoder_specs=decoder_specs,
                                model_type=model_type)
                            
        elif model_type == 'Transformer':
            print('Transformer')
            decoder_specs = {'num_initial_features': decoder_units,
                             'max_length_sequence_history': decoder_max_length_sequence,
                             'max_length_sequence_supplement': encoder_max_length_sequence,
                             'attention_heads': attention_heads,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'force_relevant_context': force_relevant_context,
                             'use_self_attention': decoder_self_attention,
                             'use_attention': decoder_attention,
                             'transformer_blocks': decoder_transformer_blocks,
                             'positional_embedding': positional_embedding,
                             'projection_layer': projection_block}
            encoder_specs = {'num_initial_features': encoder_units,
                             'max_length_sequence_supplement': encoder_max_length_sequence,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'attention_heads': attention_heads,
                             'transformer_blocks': encoder_transformer_blocks,
                             'positional_embedding': positional_embedding,}

            from Building_Blocks import S2SModel
            self.model = S2SModel(output_shape=self.target_shape,
                                encoder_specs=encoder_specs,
                                decoder_specs=decoder_specs,
                                model_type=model_type)

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
                             'max_length_sequence_history': decoder_max_length_sequence,
                             'max_length_sequence_supplement': encoder_max_length_sequence,
                             'attention_heads': attention_heads,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'force_relevant_context': force_relevant_context,
                             'use_self_attention': decoder_self_attention,
                             'use_attention': decoder_attention,
                             'transformer_blocks': decoder_transformer_blocks,
                             'positional_embedding': positional_embedding,
                             'projection_layer': projection_block,
                             'full_targets': full_targets}
            encoder_specs = {'num_initial_features': encoder_units,
                             'max_length_sequence_supplement': encoder_max_length_sequence,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'force_relevant_context': force_relevant_context,
                             'attention_heads': attention_heads,
                             'use_self_attention': encoder_self_attention,
                             'transformer_blocks': encoder_transformer_blocks,
                             'positional_embedding': positional_embedding,}
            from Building_Blocks import ForecasterModel
            self.model = ForecasterModel(output_shape=self.target_shape,
                                         encoder_specs=encoder_specs,
                                         decoder_specs=decoder_specs,
                                         model_type=model_type,
                                         history_and_features=decoder_attention)

        elif model_type == 'FFNN-LSTM-Generator':
            from Building_Blocks import FFNN_encoder, FFNN_decoder
            # decoder: width=256, depth=3, attention_heads=3, norm=True, attention_squeeze=0.5, L1=0.0, L2=0.0, projection_layer=None)
            # encoder: width=256, depth=3, attention_heads=3, norm=True, attention_squeeze=0.5, L1=0.0, L2=0.0
            decoder_specs = {'num_initial_features': decoder_units,
                             'max_length_sequence_history': decoder_max_length_sequence,
                             'max_length_sequence_supplement': encoder_max_length_sequence,
                             'attention_heads': attention_heads,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'force_relevant_context': force_relevant_context,
                             'use_self_attention': decoder_self_attention,
                             'use_attention': decoder_attention,
                             'transformer_blocks': decoder_transformer_blocks,
                             'positional_embedding': positional_embedding,
                             'projection_layer': projection_block,
                             'full_targets': full_targets}
            encoder_specs = {'num_initial_features': encoder_units,
                             'max_length_sequence_supplement': encoder_max_length_sequence,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'force_relevant_context': force_relevant_context,
                             'attention_heads': attention_heads,
                             'use_self_attention': encoder_self_attention,
                             'transformer_blocks': encoder_transformer_blocks,
                             'positional_embedding': positional_embedding,}
            from Building_Blocks import ForecasterModel
            self.model = ForecasterModel(output_shape=self.target_shape,
                                         encoder_specs=encoder_specs,
                                         decoder_specs=decoder_specs,
                                         model_type=model_type,
                                         history_and_features=decoder_attention)

        elif model_type == 'Transformer-Generator':
            from Building_Blocks import FFNN_encoder, FFNN_decoder
            # decoder: width=256, depth=3, attention_heads=3, norm=True, attention_squeeze=0.5, L1=0.0, L2=0.0, projection_layer=None)
            # encoder: width=256, depth=3, attention_heads=3, norm=True, attention_squeeze=0.5, L1=0.0, L2=0.0
            decoder_specs = {'num_initial_features': decoder_units,
                             'max_length_sequence_history': decoder_max_length_sequence,
                             'max_length_sequence_supplement': encoder_max_length_sequence,
                             'attention_heads': attention_heads,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                             'use_dense': use_dense,
                             'force_relevant_context': force_relevant_context,
                             'use_self_attention': decoder_self_attention,
                             'use_attention': decoder_attention,
                             'transformer_blocks': decoder_transformer_blocks,
                             'positional_embedding': positional_embedding,
                             'projection_layer': projection_block,
                             'full_targets': full_targets}
            encoder_specs = {'num_initial_features': encoder_units,
                             'max_length_sequence_supplement': encoder_max_length_sequence,
                             'use_residual': use_residual,
                             'use_norm': use_norm,
                            #  'force_relevant_context': force_relevant_context,
                             'attention_heads': attention_heads,
                            #  'use_self_attention': encoder_self_attention,
                             'transformer_blocks': encoder_transformer_blocks,
                             'positional_embedding': positional_embedding,}
            from Building_Blocks import ForecasterModel
            self.model = ForecasterModel(output_shape=self.target_shape,
                                         encoder_specs=encoder_specs,
                                         decoder_specs=decoder_specs,
                                         model_type=model_type,
                                         history_and_features=decoder_attention)

        else:
            print('trying to buid', model_type, 'but failed')

    def get_losses_and_metrics(self):

        # assign the losses depending on scenario
        if self.forecast_mode is not 'pdf' and self.forecast_mode is not 'ev':
            print('forecast mode was not specified as either <pdf> or <ev>, no idea how compilation got this far but expect some issues!!')

        losses = losses_and_metrics(last_output_dim_size=self.target_shape[-1],
                                    normalizer_value=1.0,
                                    target_as_expected_value=False if self.forecast_mode=='pdf' else True,
                                    forecast_as_expected_value=False if self.forecast_mode=='pdf' else True)
        loss = losses.KLD if self.forecast_mode == 'pdf' else losses.MSE

        metrics = [losses.nME,
                   losses.nRMSE]
        if self.forecast_mode == 'pdf':
            metrics.append(losses.CRPS)
            metrics.append(losses.EMC)

        return loss, metrics

    def __get_callbacks(self,
                        tboard=True,
                        ):
        callbacks = []
        if tboard:
            logdir = os.path.join(self.experiment_name)
            print('copy paste for tboard:', logdir)
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                            write_graph=False,
                                                            # update_freq='epoch',
                                                            ))

        if self.train_kwargs['mode'] ==  'pre-train':
            print('setting autostop criteria to lowest training loss')
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                              min_delta=1e-4,
                                                              patience=20,
                                                              mode='min',
                                                              restore_best_weights=True))
        elif self.train_kwargs['mode'] ==  'fine-tune' or self.train_kwargs['mode'] == 'normal':
            print('setting autostop criteria to lowest validation nRMSE')
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_nRMSE',
                                                              patience=10,
                                                              mode='min',
                                                              restore_best_weights=True))
        return callbacks

    def __get_optimizer(self, train_steps):
        # For fine-tuning we want smaller learning rate
        if self.train_kwargs['mode'] == 'fine-tune':  # assuming we will be able to use the Transformer schedule for fine-tuning
            print('setting optimizer parameters to fine tuning')
            schedule_parameter = int(self.model_kwargs['decoder_units'])
            warmup_steps = train_steps * 4
        elif self.train_kwargs['mode'] == 'normal':
            print('setting optimizer parameters to normal training')
            schedule_parameter = int(self.model_kwargs['decoder_units'])
            warmup_steps = train_steps * 4
        elif self.train_kwargs['mode'] == 'pre-train':
            print('setting optimizer parameters to pre-training')
            schedule_parameter = int(self.model_kwargs['decoder_units'])
            warmup_steps = train_steps * 6

        optimizer = tf.keras.optimizers.Adam(CustomSchedule(schedule_parameter,
                                                            warmup_steps=warmup_steps),
                                             beta_1=0.9,
                                             beta_2=0.98,
                                             epsilon=1e-9)

        # optimizer = tfa.optimizers.SWA(optimizer,
        #                                start_averaging=int(warmup_steps),
        #                                average_period=int(max(20, train_steps / 100)),
        #                                sequential_update=True)
        return optimizer

    def __train_model(self):

        epochs = 100
        dataset = dataset_generator(dataset_path_list=self.dataset_path_list,
                              train_batch_size=self.train_kwargs['batch_size'],
                              support_shape=self.model_kwargs['support_shape'],
                              history_shape=self.model_kwargs['history_shape'],
                              raw_history_shape=self.raw_history_shape,
                              val_target_shape=self.target_shape,
                              dataset_info=self.dataset_info,
                              full_targets=self.model_kwargs['full_targets'],
                              )

        if 'Generator' in self.model_kwargs['model_type']:
            train_set = dataset.pdf_generator_training_dataset
            val_set = dataset.pdf_generator_val_dataset
            test_set = dataset.pdf_generator_test_dataset
        elif 'E-D' in self.model_kwargs['model_type'] or self.model_kwargs['model_type'] == 'Transformer':
            train_set = dataset.pdf_s2s_training_dataset
            val_set = dataset.pdf_s2s_val_dataset
            test_set = dataset.pdf_s2s_test_dataset

        train_steps = dataset.get_train_steps_per_epoch()
        val_steps = dataset.get_val_steps_per_epoch()
        test_steps = dataset.get_test_steps_per_epoch()

        loss, metrics = self.get_losses_and_metrics()
        optimizer = self.__get_optimizer(train_steps)
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)  # compile, print summary

        print('starting to train model')
        train_history = self.model.fit(train_set(),
                                        steps_per_epoch=train_steps,
                                        epochs=epochs,
                                        verbose=2,
                                        validation_data=val_set(),
                                        validation_steps=val_steps,
                                        callbacks=self.__get_callbacks(tboard=True))

        # ToDo: why do we test here if we also have a class that is called test??
        test_results = self.model.evaluate(test_set(),
                                           steps=test_steps,
                                            verbose=2)
        self.model.summary()
        del dataset

        return train_history.history, test_results

    #ToDO: avoid loading the same datasset multiple times, we should move this to its own function call, make a self.dataset and then refer to this?!
    def __test_model(self):

        dataset = dataset_generator(dataset_path_list=self.dataset_path_list,
                              train_batch_size=self.train_kwargs['batch_size'],
                              support_shape=self.model_kwargs['support_shape'],
                              history_shape=self.model_kwargs['history_shape'],
                              raw_history_shape=self.raw_history_shape,
                              val_target_shape=self.target_shape,
                              dataset_info=self.dataset_info,
                              full_targets=self.model_kwargs['full_targets'],
                              )

        if 'Generator' in self.model_kwargs['model_type']:
            test_set = dataset.pdf_generator_test_dataset
        elif 'E-D' in self.model_kwargs['model_type'] or self.model_kwargs['model_type'] == 'Transformer':
            test_set = dataset.pdf_s2s_test_dataset

        test_steps = dataset.get_test_steps_per_epoch()

        # Transformer LR schedule, doesnt work .... too fast
        # ToDO: what do we need the optimizer for?
        optimizer = tf.keras.optimizers.Adam(1e-9, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        loss, metrics = self.get_losses_and_metrics()

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)  # compile, print summary

        test_results = self.model.evaluate(test_set(),
                                           steps=test_steps,
                                            verbose=2)
        self.model.summary()
        del dataset

        return test_results


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

    def __manage_test_metrics(self, test_results):

        results_dict = {}
        results_dict['test_loss'] = test_results[0]
        results_dict['test_nME'] = test_results[1]
        results_dict['test_nRMSE'] = test_results[2]
        if self.forecast_mode == 'pdf':
            results_dict['test_CRPS'] = test_results[3]

        results_dict['test_nRMSE_skill'] = 1 - (results_dict['test_nRMSE'] / self.dataset_info['test_baseline']['nRMSE'])

        if self.forecast_mode == 'pdf':
            results_dict['test_CRPS_skill'] = 1 - (results_dict['test_CRPS'] / self.dataset_info['test_baseline']['CRPS'])

        print('test_skill nRMSE', results_dict['test_nRMSE_skill'])

        if self.forecast_mode == 'pdf':
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

#ToDo: Rewrite the keras model checkpoint callback to give back and average the last 5 weights!!
class dataset_generator():
    def __init__(self,
                 dataset_path_list=None,
                 train_batch_size=None,
                 support_shape=None,
                 history_shape=None,
                 raw_history_shape=None,
                 train_target_shape=None,
                 val_target_shape=None,
                 dataset_info=None,
                 full_targets=True):

        # We keep the training size as large as possible, this means the val size needs to be smaller due to memory thingies!
        self.train_batch_size = train_batch_size
        self.val_batch_size = int(train_batch_size / 6)*2

        self.support_shape = support_shape
        self.history_shape = history_shape
        self.raw_history_shape = raw_history_shape
        self.train_target_shape = train_target_shape if train_target_shape is not None else self.history_shape
        self.val_target_shape = val_target_shape
        self.dataset_info = dataset_info
        self.full_targets = full_targets

        self.flattened_support_shape = tf.reduce_prod(self.support_shape).numpy()
        print(self.flattened_support_shape)
        self.flattened_historical_shape = tf.reduce_prod(self.history_shape).numpy()
        self.flattened_val_targets_shape = tf.reduce_prod(self.val_target_shape).numpy()


        self.train_sets = [dataset_path + '/train' for dataset_path in dataset_path_list]
        self.val_sets = [dataset_path + '/validation' for dataset_path in dataset_path_list]
        self.test_sets = [dataset_path + '/test' for dataset_path in dataset_path_list]

    def __get_all_tfrecords_in_folder(self, dataset_list):
        accumulatet_dataset_files = []
        for dataset in dataset_list:
            if os.path.isdir(dataset):
                dataset_files = glob.glob(dataset + '/*.tfrecord')
                accumulatet_dataset_files.extend(dataset_files)
            else:
                print('didnt find training data folder, expect issues!!')
        return accumulatet_dataset_files



    def get_train_steps_per_epoch(self):
        train_list = self.__get_all_tfrecords_in_folder(self.train_sets)
        return int(len(train_list) / self.train_batch_size)

    def get_val_steps_per_epoch(self):
        val_list = self.__get_all_tfrecords_in_folder(self.val_sets)
        return int(np.ceil(len(val_list) / self.val_batch_size))

    def get_test_steps_per_epoch(self):
        test_list = self.__get_all_tfrecords_in_folder(self.test_sets)
        return int(np.ceil(len(test_list) / self.val_batch_size))

    # Turn this into get the different types of datasets for generator etc

    def pdf_generator_training_dataset(self):
        return self.__dataset_from_folder_and_sample(file_list=self.__get_all_tfrecords_in_folder(self.train_sets),
                                                     batch_size=self.train_batch_size,
                                                     process_sample=self.get_pdf_generator_train_sample)

    def pdf_generator_val_dataset(self):
        return self.__dataset_from_folder_and_sample(file_list=self.__get_all_tfrecords_in_folder(self.val_sets),
                                                     batch_size=self.val_batch_size,
                                                     process_sample=self.get_pdf_generator_inference_sample)

    def pdf_generator_test_dataset(self):
        return self.__dataset_from_folder_and_sample(file_list=self.__get_all_tfrecords_in_folder(self.test_sets),
                                                  batch_size=self.val_batch_size,
                                                  process_sample=self.get_pdf_generator_inference_sample)

    def pdf_s2s_training_dataset(self):
        return self.__dataset_from_folder_and_sample(file_list=self.__get_all_tfrecords_in_folder(self.train_sets),
                                                     batch_size=self.train_batch_size,
                                                     process_sample=self.get_s2s_train_sample)

    def pdf_s2s_val_dataset(self):
        return self.__dataset_from_folder_and_sample(file_list=self.__get_all_tfrecords_in_folder(self.val_sets),
                                                     batch_size=self.val_batch_size,
                                                     process_sample=self.get_s2s_train_sample)

    def pdf_s2s_test_dataset(self):
        return self.__dataset_from_folder_and_sample(file_list=self.__get_all_tfrecords_in_folder(self.test_sets),
                                                  batch_size=self.val_batch_size,
                                                  process_sample=self.get_s2s_train_sample)


    def get_pdf_generator_inference_sample(self, example):
        features = {'support': tf.io.FixedLenFeature(self.flattened_support_shape, tf.float32),
                    'pdf_history': tf.io.FixedLenFeature(self.flattened_historical_shape, tf.float32),
                    }

        raw_unprocessed_sample = tf.io.parse_single_example(example, features)
        support_data = tf.reshape(tensor=raw_unprocessed_sample['support'], shape=self.support_shape)
        full_pdf_history = tf.reshape(tensor=raw_unprocessed_sample['pdf_history'], shape=self.history_shape)
        target = full_pdf_history[-self.val_target_shape[0]:,:]
        history_input =  full_pdf_history[:-self.val_target_shape[0],:]

        return {'support_input': support_data,
                'history_input': history_input}, target

    def get_pdf_generator_train_sample(self, example):
        features = {
        'support': tf.io.FixedLenFeature(self.flattened_support_shape, tf.float32),
        'pdf_history': tf.io.FixedLenFeature(self.flattened_historical_shape, tf.float32),}

        raw_unprocessed_sample = tf.io.parse_single_example(example, features)
        full_pdf_history = tf.reshape(tensor=raw_unprocessed_sample['pdf_history'], shape=self.history_shape)
        support_data = tf.reshape(tensor=raw_unprocessed_sample['support'], shape=self.support_shape)

        if self.full_targets:
            target = full_pdf_history[1:, :] # for predicting full targets vs. last 24 steps

        history_input =  full_pdf_history[:-1,:]

        # ATM sampling last 2days plus forecast day from NWP and last 2 days from history
        return {'support_input': support_data,
                'history_input': history_input}, target

    #ToDo: fix those
    def get_s2s_train_sample(self, example):
        
        features = {'support': tf.io.FixedLenFeature(self.flattened_support_shape, tf.float32),
                    'raw_history': tf.io.FixedLenFeature(self.raw_history_shape, tf.float32),
                    'pdf_history': tf.io.FixedLenFeature(self.flattened_historical_shape, tf.float32),
                    }
                    # 'pdf_target': tf.io.FixedLenFeature(self.flattened_val_targets_shape, tf.float32),
                    # 'pdf_teacher': tf.io.FixedLenFeature(self.flattened_val_targets_shape, tf.float32),
                    # }

        raw_unprocessed_sample = tf.io.parse_single_example(example, features)

        nwp_inputs = tf.reshape(tensor=raw_unprocessed_sample['support'], shape=self.support_shape)
        history = tf.reshape(tensor=raw_unprocessed_sample['raw_history'], shape=self.raw_history_shape)
        full_pdf_history = tf.reshape(tensor=raw_unprocessed_sample['pdf_history'], shape=self.history_shape)

        target = full_pdf_history[-self.val_target_shape[0]:,:]
        teacher = full_pdf_history[-(self.val_target_shape[0]+1):-1,:]
        history_input =  full_pdf_history[:-self.val_target_shape[0],:]

        # downsample both to (hourly) rate accodring to the target
        history_downs_factor = int(history.shape[0]/full_pdf_history.shape[0])
        nwp_downs_factor = int(nwp_inputs.shape[0]/full_pdf_history.shape[0])
        nwp_inputs = nwp_inputs[::nwp_downs_factor,:]
        history = history[::history_downs_factor,:]
        # concat into one input
        inputs = tf.concat([nwp_inputs,history], axis=-1)
        # cut off until target
        inputs = inputs[:-self.val_target_shape[0],:]

        # ATM sampling last 2days plus forecast day from NWP and last 2 days from history
        return {'nwp_pv_input': inputs}, target

    def __calculate_expected_value(self, signal, last_output_dim_size):
        indices = tf.range(last_output_dim_size)  # (last_output_dim_size)
        weighted_signal = tf.multiply(signal, indices)  # (batches, timesteps, last_output_dim_size)
        expected_value = tf.reduce_sum(weighted_signal, axis=-1, keepdims=True)
        return expected_value / last_output_dim_size

    def __dataset_from_folder_and_sample(self, process_sample, file_list, batch_size):
        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = False

        # dataset = tf.data.Dataset.list_files(file_list)
        dataset = tf.data.TFRecordDataset(file_list, num_parallel_reads=5)
        dataset = dataset.with_options(option_no_order)
        dataset = dataset.map(process_sample, num_parallel_calls=5)
        dataset = dataset.repeat()

        dataset = dataset.shuffle(20 * batch_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(3)
        return dataset