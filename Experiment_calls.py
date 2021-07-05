import numpy as np
import pickle
import matplotlib.pyplot as plt
from Train_Eval import Model_Container
import tensorflow as tf
#ToDo: Experiment1 is to check wether placement of dropout before or after norm is better
#ToDo: Retest cost curves with KS and CRPS!
#ToDo: Experiment2 is to check for different dropout rates
#ToDo: Experiment3 is to check for layer depth in Encoder and Decoder


def do_experiment(model_type,
                exp_name,

                full_targets=True,
                encoder_units=256,
                encoder_self_attention=True,
                encoder_transformer_blocks=1,
                decoder_units=256,
                decoder_self_attention=True,
                decoder_attention=True,
                decoder_transformer_blocks=3,
                attention_heads=3,
                positional_embedding=True,
                use_residual=True,
                use_norm=True,
                use_gru=False,
                mode='fine-tune', # normal, pre-train or fine-tune or test
                dataset_path_list=None,
                ):


    experiment_name = exp_name
    sliding_window_length_days = 6
    model_kwargs = {'model_type': model_type,  #'FFNN-Generator',
                    'forecast_mode': 'pdf',
                    'full_targets': full_targets,

                    # Architecture Hyperparameters
                    # Encoder:
                        'encoder_units' : encoder_units,
                        'encoder_receptive_window': 6*4,
                        'encoder_self_attention': encoder_self_attention,
                        'encoder_max_length_sequence': 2*sliding_window_length_days * 24 * 4,
                        'encoder_transformer_blocks': encoder_transformer_blocks,
                    # Decoder:
                        'decoder_units': decoder_units,
                        'decoder_receptive_window': 6,
                        'decoder_self_attention': decoder_self_attention,
                        'decoder_attention': decoder_attention,
                        'decoder_transformer_blocks': decoder_transformer_blocks,
                        'decoder_max_length_sequence': 2*sliding_window_length_days*24,

                    'attention_heads': attention_heads,

                    # General information flow
                        'positional_embedding': positional_embedding,
                        'force_relevant_context': False,
                        'use_dense': False,
                        'use_residual': use_residual,
                        'use_gru': use_gru,
                        # 'downsample': False, 'mode': 'project',

                    # Regularization Hyperparameters
                        # 'use_dropout' : False, 'dropout_rate' : 0.0,
                        'L1': 0.0, 'L2': 0.0, #1e-6,
                        'use_norm' : use_norm,
                    }

    train_kwargs = {'batch_size': 2**5,
                    'mode': mode}
    runs = 1
    metrics = {}
    for run in range(runs):

        if mode == 'normal':
            print('Training model', model_type, experiment_name)

        elif mode == 'pre-train':
            print('Training model', model_type, experiment_name, '-pre-trained')

        elif mode =='fine-tune':
            print('Fine-tuning model', model_type, experiment_name, '-fine-tuned')

        elif mode == 'test':
            print('Testing model', model_type, experiment_name, '-test')

        else:
            print('no recognized training procedure requested')

        experiment = Model_Container(dataset_path_list=dataset_path_list,
                                    experiment_name=experiment_name,
                                    sw_len_days=sliding_window_length_days,
                                    model_kwargs=model_kwargs,
                                    train_kwargs=train_kwargs,)


        if mode == 'fine-tune' or mode == 'normal' or mode == 'pre-train':
            results_dict = experiment.train()
        elif mode == 'test':
            results_dict = experiment.test()

        tf.keras.backend.clear_session()
        del experiment





        for key in results_dict:
            if key not in metrics:
                metrics[key] = [results_dict[key]]
            else:
                metrics[key].append(results_dict[key])

    for key in metrics:
        if 'skill' in key:
            print(key, ': ', metrics[key])


    with open('exp/'+experiment_name+'.pickle', 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

   # __plot_training_curves(metrics, experiment_name=experiment_name)

def __plot_training_curves(metrics, experiment_name):

    val_color = [255 / 255, 165 / 255, 0 / 255]
    test_color = 'g'
    train_color = 'b'

    val_metrics = {}
    test_metrics = {}
    train_metrics = {}

    run_linestyles = [':', '--', '-.', '-']

    for set in metrics.keys():
        for key in metrics[set][0].keys():
            if 'val' in str(key):
                val_metrics[key] = [run[key] for run in metrics[set]]

            if 'test' in str(key):
                test_metrics[key] = [run[key] for run in metrics[set]]

            elif 'val' not in str(key) and 'test' not in str(key):
                train_metrics[key] = [run[key] for run in metrics[set]]

        for train_key in train_metrics.keys():
            for run in range(len(train_metrics[train_key])):
                plt.plot(train_metrics[train_key][run],
                         label=str(train_key),
                         color=train_color,
                         linestyle=run_linestyles[run])

                saved_val_index = int(np.argmin(val_metrics['val_nRMSE'][run]))
                for val_key in val_metrics.keys():
                    if str(train_key) in str(val_key) and 'skill' not in str(val_key):
                        plt.plot(val_metrics[val_key][run],
                                 label=str(val_key) + '_run'+ str(run),
                                 color=val_color,
                                 linestyle=run_linestyles[run])
                        plt.plot(saved_val_index, val_metrics[val_key][run][saved_val_index],
                                 label='chosen' + str(val_key) + '_run' + str(run),
                                 color=val_color,
                                 marker='o', markersize=4,
                                 linestyle=run_linestyles[run])

                        if np.argmax(val_metrics[val_key][run]) != saved_val_index:
                            best_val_index = int(np.argmin(val_metrics[val_key][run]))
                            best_val = np.amin(val_metrics[val_key][run])
                            plt.plot(best_val_index, best_val,
                                     label='bext' + str(val_key) + '_run' + str(run),
                                     color=val_color,
                                     marker='*', markersize=4,
                                     linestyle=run_linestyles[run])

                for test_key in test_metrics.keys():
                    if str(train_key) in str(test_key) and 'skill' not in str(test_key):
                        plt.plot(saved_val_index, test_metrics[test_key][run],
                                 label=str(test_key) + '_run'+ str(run),
                                    marker='o', markersize=4,
                                 color=test_color,
                                 linestyle=run_linestyles[run])

            plt.title(str(len(train_metrics[train_key])) + 'for dataset ' + str(set))
            plt.ylabel(str(train_key))
            plt.xlabel('number of Epochs')
            plt.grid()
            plt.legend()
            plt.savefig(experiment_name, dpi=500, format='pdf')
            plt.show()

#########################################################################################################################
# Perform experiments. 

experiments = []
dataset_name = 'thesis_dataset'

# experiments.append({'model_type': 'E-D-luong',
#                                 'exp_name': 'S2S-ATTN-Luong_2x110'+'_shifted_SW'+'_TF_'+dataset_name,
#                                 'encoder_units':110,
#                                 'encoder_transformer_blocks': 2,
#                                 'decoder_units': 110,
#                                 'decoder_attention': True,
#                                 'decoder_transformer_blocks': 2,
#                                 'attention_heads': 1,
#                                 'mode': 'normal',
#                                 'use_residual':False,
#                                 'full_targets':False,
#                                 'use_gru':False,
#                                 'dataset_path_list': ['./'+dataset_name]#['/media/elizaveta/Seagate Portable Drive/egauge2474solar+'] #, 'egauge4183grid'], #, '/media/elizaveta/Seagate Portable Drive/egauge22785solar+'],
#                                 })

experiments.append({'model_type': 'E-D',
                                'exp_name': 'S2S-ATTN_2x110'+'_shifted_SW'+'_noTF_'+dataset_name, 
                                'encoder_units':110,
                                'encoder_transformer_blocks': 2,
                                'decoder_units': 110,
                                'decoder_attention': True,
                                'decoder_transformer_blocks': 2,
                                'attention_heads': 1,
                                'mode': 'test',
                                'use_residual':False,
                                'full_targets':False,
                                'use_gru':False,
                                'dataset_path_list': ['./'+dataset_name]
                                })

# experiments.append({'model_type': 'E-D',
#                                 'exp_name': 'S2S_2x128'+'_shifted_SW'+'_TF_'+dataset_name,
#                                 'encoder_units':128,
#                                 'encoder_transformer_blocks': 2,
#                                 'decoder_units': 128,
#                                 'decoder_attention': False,
#                                 'decoder_transformer_blocks': 2,
#                                 'attention_heads': 1,
#                                 'mode': 'normal',
#                                 'use_residual':False,
#                                 'full_targets':False,
#                                 'use_gru':False,
#                                 'dataset_path_list': ['./'+dataset_name]
#                                 })

# experiments.append({'model_type': 'MiMo-LSTM',
#                                 'exp_name': 'LSTM_2x184'+'_shifted_SW'+'_TF_'+dataset_name,
#                                 'encoder_units':184,
#                                 'encoder_transformer_blocks': 2,
#                                 'decoder_units': 184,
#                                 'decoder_attention': False,
#                                 'decoder_transformer_blocks': 2,
#                                 'attention_heads': 1,
#                                 'mode': 'normal',
#                                 'use_residual':False,
#                                 'full_targets':False,
#                                 'use_gru':False,
#                                 'dataset_path_list': ['./'+dataset_name]
#                                 })
#
# # Classic Transformer.
# experiments.append({'model_type': 'Transformer',
#                     'exp_name': 'Transformer-3x-12H-256',
#                     'encoder_units': 256,
#                      'encoder_self_attention': True,
#                     'encoder_transformer_blocks': 3,
#                     'decoder_units': 256,
#                     'decoder_self_attention': True,
#                     'decoder_attention': True,
#                     'decoder_transformer_blocks':3,
#                     'attention_heads': 12,
#                     'positional_embedding': True,
#                     'use_residual': True,
#                     'use_norm': True,
#                     'mode': 'normal',
#                     'dataset_path_list': ['/media/elizaveta/Seagate Portable Drive/egauge2474solar+'],})
#
# # Transformer encoder + generator.
# experiments.append({'model_type': 'Transformer-Generator',
#                     'exp_name': 'Transformer-Gen-3x-12H-256',
#                     # 'full_targets': True, # set for full target case
#                     'encoder_units': 256,
#                     'encoder_self_attention': True,
#                     'encoder_transformer_blocks': 3,
#                     'decoder_units': 256,
#                     'decoder_self_attention': True,
#                     'decoder_attention': True,
#                     'decoder_transformer_blocks':3,
#                     'attention_heads': 12,
#                     'positional_embedding': True,
#                     'use_residual': True,
#                     'use_norm': True,})
#
# # Transformer encoder + generator with full targets
# experiments.append({'model_type': 'Transformer-Generator',
#                     'exp_name': 'Transformer-Gen-3x-12H-256-full-targets',
#                     'full_targets': True, # set for full target case
#                     'encoder_units': 256,
#                     'encoder_self_attention': True,
#                     'encoder_transformer_blocks': 3,
#                     'decoder_units': 256,
#                     'decoder_self_attention': True,
#                     'decoder_attention': True,
#                     'decoder_transformer_blocks':3,
#                     'attention_heads': 12,
#                     'positional_embedding': True,
#                     'use_residual': True,
#                     'use_norm': True,
#                     'mode': 'normal',
#                     'dataset_path_list': ['/media/elizaveta/Seagate Portable Drive/egauge2474solar+'],})
#
# # Generator with full targets and with NO encoder.
# experiments.append({'model_type': 'FFNN-Generator',
#                     'exp_name': 'FFNNGen-3x-12H-256-NoFeatures-full-targets',
#                     'full_targets': True, # only set for full target
#                     'encoder_units': 256,
#                     'encoder_self_attention': True,
#                     'encoder_transformer_blocks': 1,
#                     'decoder_units': 256,
#                     'decoder_self_attention': True,
#                     'decoder_attention': False, # this is for no features (aka no encoder)
#                     'decoder_transformer_blocks':3,
#                     'attention_heads': 12,
#                     'positional_embedding': True,
#                     'use_residual': True,
#                     'use_norm': True,
#                     'mode': 'normal',
#                     'dataset_path_list': ['/media/elizaveta/Seagate Portable Drive/egauge2474solar+'],
#                     })
#
# # Generator with LSTM instead of SA and full targets.
# experiments.append({'model_type': 'FFNN-LSTM-Generator',
#                     'exp_name': 'FFNN-LSTM-Gen-3x-3H-256-full-targets',
#                     'full_targets': True, # only set for full target
#                     'encoder_units': 256,
#                     'encoder_self_attention': True,
#                     'encoder_transformer_blocks': 1,
#                     'decoder_units': 256,
#                     'decoder_self_attention': True,
#                     'decoder_attention': True,
#                     'decoder_transformer_blocks': 3,
#                     'attention_heads': 3,
#                     'positional_embedding': True,
#                     'use_residual': True,
#                     'use_norm': True,
                    # 'mode': 'normal',
                    # 'dataset_path_list': ['/media/elizaveta/Seagate Portable Drive/egauge2474solar+'],
#                     })
#
# # Generator with LSTM instead of SA and NO encoder. (full targets)
# experiments.append({'model_type': 'FFNN-LSTM-Generator',
#                     'exp_name': 'FFNN-LSTM-Gen-3x-12H-256-NoFeatures-full-targets',
#                     'full_targets': True, # only set for full target
#                     'encoder_units': 256,
#                     'encoder_self_attention': True,
#                     'encoder_transformer_blocks': 1,
#                     'decoder_units': 256,
#                     'decoder_self_attention': True,
#                     'decoder_attention': False, # no exogenous variables (no encoder)
#                     'decoder_transformer_blocks': 3,
#                     'attention_heads': 12,
#                     'positional_embedding': True,
#                     'use_residual': True,
#                     'use_norm': True,
#                     })
#
# Generator with 24 step targets.
# experiments.append({'model_type': 'FFNN-Generator',
#                     'exp_name': 'FFNNGen-not_full_targets',
#                     'full_targets': False, # only set for full target
#                     'encoder_units': 256,
#                      'encoder_self_attention': True,
#                     'encoder_transformer_blocks': 1,
#                     'decoder_units': 256,
#                     'decoder_self_attention': True,
#                     'decoder_attention': True,
#                     'decoder_transformer_blocks': 3,
#                     'attention_heads': 12,
#                     'positional_embedding': True,
#                     'use_residual': True,
#                     'use_norm': True,
#                     'mode': 'normal',
#                     'dataset_path_list': ['/media/elizaveta/Seagate Portable Drive/egauge2474solar+'],
#                     })

# Generator with full targets: supposed to be best!
# experiments.append({'model_type': 'FFNN-Generator',
#                     'exp_name': 'FFNNGen_Edmonton_ThomasOldPipeline',
#                     'full_targets': True, # only set for full target
#                     'encoder_units': 256, #256
#                     'encoder_self_attention': True,
#                     'encoder_transformer_blocks': 1,
#                     'decoder_units': 256, #256
#                     'decoder_self_attention': True,
#                     'decoder_attention': True,
#                     'decoder_transformer_blocks': 3, #3
#                     'attention_heads': 12, #12
#                     'positional_embedding': True,
#                     'use_residual': True,
#                     'use_norm': True,
#                     'mode': 'normal',
#                     'dataset_path_list': ['/media/elizaveta/Seagate Portable Drive/egauge2474solar+'],
#                     })

# experiments.append({'model_type': 'LSTM-Generator',
#                     'exp_name': 'FFNNGen_Edmonton_ThomasOldPipeline',
#                     'full_targets': False, # only set for full target
#                     'encoder_units': 256, #256
#                     'encoder_self_attention': True,
#                     'encoder_transformer_blocks': 1,
#                     'decoder_units': 256, #256
#                     'decoder_self_attention': True,
#                     'decoder_attention': True,
#                     'decoder_transformer_blocks': 3, #3
#                     'attention_heads': 12, #12
#                     'positional_embedding': True,
#                     'use_residual': True,
#                     'use_norm': True,
#                     'mode': 'normal',
#                     'dataset_path_list': ['/media/elizaveta/Seagate Portable Drive/egauge2474solar+'],
#                     })


# do our experiments
for exp_args in experiments: 
    do_experiment(**exp_args)
    # test on the new dataset
    # exp_args['mode'] = 'test'
    # exp_args['dataset_path_list'] = ['/media/elizaveta/Seagate Portable Drive/egauge22785solar+']  # TODO: CHANGE DATASET NAMES HERE
    # do_experiment(**exp_args)
    # # fine-tune on the new dataset
    # exp_args['mode'] = 'fine-tune'
    # do_experiment(**exp_args)