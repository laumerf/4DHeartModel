import logging, sys, time, json
import numpy as np
from itertools import chain
import multiprocessing as mp
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses

import sklearn.metrics as sk_metrics


import source.utils as utils
from source.shape_model_utils import mesh_echo_iou_and_dice

from source.constants import ROOT_LOGGER_STR, ECHO_DATA_PARAMS

# get the logger
logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

LEOMED_RUN = "--leomed" in sys.argv # True if running on LEOMED

# ------------------------------------------------------- Generators --------------------------------------------------------
class Echo_to_Mesh_FFNN(Layer):

    def __init__(self, dense_layers, latent_space_mesh_dim, name='Echo_to_Mesh_FFNN', **kwargs):
        super(Echo_to_Mesh_FFNN, self).__init__(name=name, **kwargs)

        # make sure the last dense layer outputs a vector of size "nb of mesh's shape params"
        assert dense_layers[-1]['units'] == (latent_space_mesh_dim - 2)

        self.dense_layers = []
        self.batch_norms = []
        self.activations = []
        for i, dense_layer in enumerate(dense_layers):
            # get if batch_norm used
            batch_norm = dense_layer.get("batch_norm", False)
            # remove the key batch_norm from dense_layer dict (not using pop to be able to call save_me without issues)
            dense_layer = utils.skip_dict_keys(dense_layer, ["batch_norm"])
            # get activation function
            logger.info(dense_layer)
            act_str = dense_layer['activation']['name'].lower() # get activation in lowercase
            if act_str == 'leakyrelu':
                activation = LeakyReLU(alpha=dense_layer['activation']['alpha'])
            elif act_str == 'prelu':
                activation = PReLU()
            else:
                activation = tf.keras.activations.get(dense_layer['activation']['name'])
            
            # Linear Dense
            dense_layer['activation'] = 'linear'
            if dense_layer["use_bias"] and batch_norm: # do not use_bias if using batch_norm
                dense_layer["use_bias"] = False
            dense = Dense(**dense_layer)
            self.dense_layers.append(dense)

            if batch_norm: # apply batch_norm
                self.batch_norms.append(BatchNormalization())
            else: # pass through
                self.batch_norms.append(None)
            
            self.activations.append(activation)

    def call(self, inputs, training=False):
        """
        inputs is expected to be of shape [batch_size, latent_space_echo_dim]
        """
        freqs = inputs[:, 0, None]
        phases = inputs[:, 1, None]
        h = inputs[:, 2:] # echo shape params
        
        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](h)
            if self.batch_norms[i] is not None:
                h = self.batch_norms[i](h, training=training)
            h = self.activations[i](h)

        output = tf.concat([freqs, phases, h], axis=1)

        return output
    
# ---------------------------------------------------------------------------------------------------------------------------

class Echo_to_Mesh_Model(Model):

    def __init__(self, log_dir, echo_data_dir, mesh_ae, training_params, model_params, save_metrics=True, name='Echo_to_Mesh_Model', **kwargs):

        super(Echo_to_Mesh_Model, self).__init__(name=name, **kwargs)

        # save experiment path in self
        self.log_dir = log_dir # current experiment log dir (datetime dir)
        self.echo_data_dir = echo_data_dir
        # save model, training parameters and model parameters in self
        self.mesh_ae = mesh_ae
        self.training_params = training_params
        self.model_params = model_params

        # build the Mesh EF Predictor model
        self.build_model(model_params)

        if save_metrics:
            self.create_metrics_writers()

        logger.info("Model initialized and built.")

    def build_model(self, model_params):
        self.echo_to_mesh = Echo_to_Mesh_FFNN(dense_layers=model_params['dense_layers'], latent_space_mesh_dim=256)
    
    def create_metrics_writers(self):
        
        train_metrics = ['loss', 'learning_rate']
        val_metrics = ['loss']

        self._val_metrics = dict()
        self._train_metrics = dict()

        train_log_dir = self.log_dir / 'metrics' / 'train'
        val_log_dir = self.log_dir / 'metrics' / 'validation'
        self._train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
        self._val_summary_writer = tf.summary.create_file_writer(str(val_log_dir))

        for metric in train_metrics:
            self._train_metrics[metric] = tf.keras.metrics.Mean(metric)

        for metric in val_metrics:
            self._val_metrics[metric] = tf.keras.metrics.Mean(metric)
    
    @tf.function
    def call(self, latent_params, training=False):
        return self.echo_to_mesh(latent_params, training=training)

    # --------------------------------------------------- Computing losses ----------------------------------------------

    def mse(self, y_true, y_pred):
        h_true = y_true[:, 2:] # true mesh shape params
        h_pred = y_pred[:, 2:] # pred mesh shape params
        mse = tf.reduce_mean(losses.mean_squared_error(h_true, h_pred))
        return mse
    
    def mae(self, y_true, y_pred):
        h_true = y_true[:, 2:] # true mesh shape params
        h_pred = y_pred[:, 2:] # pred mesh shape params
        mae = tf.reduce_mean(losses.mean_absolute_error(h_true, h_pred))
        return mae

    def _loss(self, y_true, y_pred):
        which_loss = self.training_params['loss_str']
        if which_loss == 'l2':
            return self.mse(y_true, y_pred)
        elif which_loss == 'l1':
            return self.mae(y_true, y_pred)
    # -------------------------------------------------------------------------------------------------------------------
    
    @tf.function
    def _train_step(self, echo_true_latents, mesh_true_latents):
        with tf.GradientTape() as tape:
            # run model end-to-end in train mode:
            mesh_pred_latents = self.call(echo_true_latents, training=True)
            # compute loss
            loss = self._loss(mesh_true_latents, mesh_pred_latents)
        
        # update model weights
        variables = self.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return {"loss": loss}
    
    def _val_step(self, echo_true_latents_v, mesh_true_latents_v):
        # run model end-to-end in non-training mode
        mesh_pred_latents_v = self.call(echo_true_latents_v)
        # compute loss
        loss = self._loss(mesh_true_latents_v, mesh_pred_latents_v)

        return loss
    
    def fit(self, train_dataset, val_dataset, test_dataset):

        opt_early_stopping_metric = np.inf
        count = epoch = epoch_step = global_step = 0
        opt_weights = self.get_weights()

        optimizer_params = self.training_params['optimizer']
        patience = self.training_params['patience']
        learning_rate_decay = self.training_params['decay_rate']
        lr = optimizer_params['learning_rate']

        # create optimizer
        self.optimizer = Adam(**optimizer_params)

        num_steps_summaries = self.training_params['num_steps_summaries']
        num_summaries_val = self.training_params['num_summaries_val']
        max_steps = self.training_params['max_steps']

        # log number of trainable weights before training
        logger.info("Start training...\n")

        logged_weights = False

        t1_steps = time.time() # start epoch timer
        
        self.epoch = epoch # save epoch nb in self

        for echo_true_latents, mesh_true_latents in train_dataset:
            if global_step == max_steps:
                break
            
            losses = self._train_step(echo_true_latents, mesh_true_latents)
            self._log_train_metrics(losses)

            if not logged_weights:
                self.log_weights()
                logged_weights = True
            
            global_step += 1 # increment nb total of steps

            if global_step % num_steps_summaries == 0:
                self._train_metrics['learning_rate'](lr) # log learning rate with train metrics

                # stop "steps timer" log train metrics and log "steps time"
                t2_steps = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2_steps - t1_steps)
                self.write_summaries(self._train_summary_writer, self._train_metrics, global_step, "Train")
                logger.info("{} steps done in {}:{}:{}\n".format(num_steps_summaries, h, m, s))
            
                # reset metrics
                self.reset_metrics()

                # start steps timer again
                t1_steps = time.time()
            
            if global_step % (num_steps_summaries * num_summaries_val) == 0:
                # Validation at the end of every epoch.
                logger.info("Computing validation loss...")
                t1_val = time.time() # start validation timer

                for echo_true_latents_v, mesh_true_latents_v in val_dataset:
                    # perform a validation step and log metrics
                    loss = self._val_step(echo_true_latents_v, mesh_true_latents_v)
                    self._val_metrics['loss'](loss)
                
                # stop validation timer, log validation metrics and log epoch time
                t2_val = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2_val - t1_val)
                self.write_summaries(self._val_summary_writer, self._val_metrics, global_step, "Validation")
                logger.info("Validation done in {}:{}:{}\n".format(h, m, s))

                # get validation loss
                val_loss = self._val_metrics['loss'].result()
                # if new validation loss worse than previous one:
                if val_loss >= opt_early_stopping_metric:
                    count += 1
                    logger.info(f"Validation loss did not improve from {opt_early_stopping_metric}. Counter: {count}")

                    # update learning rate after waiting "patience" validations, decay learning rate (by multiplying by the decay_rate) 
                    # to see if any improvement on validation loss can be seen
                    if count % patience == 0:
                        lr = float(self.optimizer.learning_rate * learning_rate_decay) # new learning rate
                        logger.info(self.optimizer.learning_rate)
                        logger.info(lr)
                        logger.info(f"Reduced learning rate to {lr} using decay_rate {learning_rate_decay}")
                        self.optimizer.learning_rate = lr
                else: # validation loss improved
                    logger.info("Validation loss improved, saving model.")
                    opt_early_stopping_metric = float(val_loss)
                    opt_weights = self.get_weights()

                    # save best model
                    self.save_me()

                    # reset counter
                    count = 0                    

                # reset metrics
                self.reset_metrics()

                # start steps timer again
                t1_steps = time.time()

    # ----------------------------------------------- Logging, Saving and Loading -----------------------------------------------
    def log_weights(self):
        # log number of trainable weights
        n_trainable = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_weights])
        logger.info(f"Number of trainable weights: {n_trainable}")

    def _log_train_metrics(self, losses):
        for metric in losses:
            self._train_metrics[metric](losses[metric])
    
    @staticmethod
    def write_summaries(summary_writer, metrics, global_step, log_str):
        with summary_writer.as_default():
            for metric, value in metrics.items():
                tf.summary.scalar(metric, value.result(), step=global_step)

        # print train metrics
        strings = ['%s: %.5e' % (k, v.result()) for k, v in metrics.items()]
        logger.info(f"\nGlobal Step: {global_step} | {log_str}: {' - '.join(strings)} ")
    
    def reset_metrics(self):
        metrics = chain(self._train_metrics.values(), self._val_metrics.values())
        for metric in metrics:
            metric.reset_states()
    
    def save_me(self):
        
        trained_model_path = self.log_dir / "trained_models" / "echo_to_mesh"
        self.save_weights(str(trained_model_path) + "_best")
        
        logger.info(f"Model saved to file {trained_model_path}")
    # ---------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------ Experiments --------------------------------------------------------
    def dice_and_IoU_echo_mesh_volume_based(self, echo_ae, mesh_ae, echo_dataset, echo_filenames, echo_edfs, framerate_ds_factor=1, mean_mesh_iou_dice=True, compute_ef=False, save=True, iou_save_filename="IoUs.csv", dice_save_filename="Dice.csv", ef_save_filename="EF.csv"):
        volume_tracings = utils.get_volume_tracings_df(self.echo_data_dir)

        # for each echo filename, the tracing_data is a pair of triple (one for EDF and one for ESF)
        # where triple = (tracing_frame, lax, frame_nb)
        output_size = (112, 112)
        echo_tracings_data, skipped_echos = utils.generate_echo_tracings(echo_filenames, echo_edfs, volume_tracings, output_size)

        # mesh dataset handler
        mesh_data_handler = mesh_ae.data_handler

        # echo ef index
        echo_ef_index = ECHO_DATA_PARAMS.index("EF")

        # parallel jobs
        manager = mp.Manager()
        return_dict = manager.dict()
        max_parallel_jobs = mp.cpu_count()
        parallel_jobs = []
        
        t_start = time.time()
        t1 = time.time()
        batch_size = -1
        all_echo_efs = {}
        for b, (echo_times, echo_frames, echo_params) in enumerate(echo_dataset): # b'th batch

            # translate echo vids to mesh vids aligned in time (transform relevant arrays to numpy)
            real_echo_latents = echo_ae.encode(echo_times, echo_frames, return_freq_phase=True, training=False)
            fake_mesh_latents = self.call(real_echo_latents)

            # decrease framerate of decoded fake meshes for faster processing
            echo_times_tensor = echo_times.to_tensor(default_value=-1.0) # comvert to tensor with default value -1
            max_len = echo_times_tensor.shape[1]
            indices = list(range(0, max_len, framerate_ds_factor)) # indices to take, take 1 frame for every "framerate_ds_factor" frames
            fake_mesh_times = tf.gather(echo_times_tensor, indices=indices, axis=1) # downsampled echo times
            fake_mesh_times = tf.RaggedTensor.from_tensor(fake_mesh_times, padding=-1) # convert back to RaggedTensor

            fake_mesh_feats = mesh_ae.decode(fake_mesh_latents, fake_mesh_times.values, fake_mesh_times.row_lengths(), training=False).numpy()

            echo_params = echo_params.to_tensor(default_value=0.0)
            echo_efs = echo_params[:, echo_ef_index].numpy()

            # get row lengths and row limits
            row_lengths = fake_mesh_times.row_lengths().numpy()
            row_limits = np.cumsum(row_lengths)
            if batch_size == -1: # get batch_size
                batch_size = row_lengths.shape[0]
                logger.info(f"Batch size: {batch_size}")

            if b % 10 == 0: # print progress
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b} after {h}:{m}:{s}")
            
            i = 0
            for k, j in enumerate(row_limits): # k'th video in the batch "b"
                mesh_feats = fake_mesh_feats[i:j]
                global_index = b * batch_size + k # index of the video in the whole dataset
                echo_filename = echo_filenames[global_index]
                if echo_filename in skipped_echos: # skip echo video if no tracing data available
                    i = j
                    continue
                
                # save echo ef
                echo_ef = echo_efs[k]
                all_echo_efs[echo_filename] = echo_ef
                
                # get the echo EDF and ESF tracing data
                echo_tracing_data = echo_tracings_data[echo_filename]
                echo_edf_tracing_data, echo_esf_tracing_data = echo_tracing_data

                echo_edf_tracing_frame, echo_edf_lax, edf = echo_edf_tracing_data
                echo_esf_tracing_frame, echo_esf_lax, esf = echo_esf_tracing_data

                echo_frames = [echo_edf_tracing_frame, echo_esf_tracing_frame]
                echo_laxes = [echo_edf_lax, echo_esf_lax]
                
                output_dir = None
                if save:
                    output_dir = self.log_dir / "echo_mesh_lv_tracings"
                
                # mesh_echo_iou_and_dice(echo_filename, echo_frames, echo_laxes, mesh_feats, mesh_data_handler, mean_mesh_iou_dice, output_size, True, compute_ef, False, None, output_dir)
                # exit(0)
                args=(echo_filename, echo_frames, echo_laxes, mesh_feats, mesh_data_handler, mean_mesh_iou_dice, output_size, True, compute_ef, True, return_dict, output_dir)
                p = mp.Process(target=mesh_echo_iou_and_dice, args=args)
                parallel_jobs.append(p)
                p.start()

                i = j
            
            # wait for all parallel jobs of batch to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)
            
            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done 1 batch of {batch_size} videos in {h}:{m}:{s}")
            t1 = time.time()
        
        # generated mesh IoU and Dice values
        edf_IoUs, esf_IoUs = [], []
        edf_Dices, esf_Dices = [], []
        # mean mesh IoU and Dice values
        edf_IoUs_MM, esf_IoUs_MM = [], []
        edf_Dices_MM, esf_Dices_MM = [], []
        filenames = []

        # ef values
        edvs_3d, esvs_3d, edvs_biplane, esvs_biplane = [], [], [], []
        efs_vol, efs_biplane, efs_echo = [], [], []

        for echo_filename in echo_filenames:
            if echo_filename in skipped_echos:
                continue

            IoUs, Dices, IoUs_MM, Dices_MM, EFs, Volumes = return_dict[echo_filename]
                
            edf_IoUs.append(IoUs[0])
            esf_IoUs.append(IoUs[1])
            edf_Dices.append(Dices[0])
            esf_Dices.append(Dices[1])

            edf_IoUs_MM.append(IoUs_MM[0])
            esf_IoUs_MM.append(IoUs_MM[1])
            edf_Dices_MM.append(Dices_MM[0])
            esf_Dices_MM.append(Dices_MM[1])

            efs_vol.append(EFs[0])
            efs_biplane.append(EFs[1])
            efs_echo.append(all_echo_efs[echo_filename])

            edvs_3d.append(Volumes[0])
            esvs_3d.append(Volumes[1])
            edvs_biplane.append(Volumes[2])
            esvs_biplane.append(Volumes[3])

            filenames.append(echo_filename)
        
        IoU_df = pd.DataFrame({"FileName": filenames, "EDF_IoU": edf_IoUs, "ESF_IoU": esf_IoUs, "Mean_Mesh_EDF_IoU": edf_IoUs_MM, "Mean_Mesh_ESF_IoU": esf_IoUs_MM})
        Dice_df = pd.DataFrame({"FileName": filenames, "EDF_Dice": edf_Dices, "ESF_Dice": esf_Dices, "Mean_Mesh_EDF_Dice": edf_Dices_MM, "Mean_Mesh_ESF_Dice": esf_Dices_MM})
        EF_df = pd.DataFrame({"FileName": filenames, "EF_Vol": efs_vol, "EF_Biplane": efs_biplane, "EF_Echo": efs_echo, "Mesh_EDV_3D": edvs_3d, "Mesh_ESV_3D": esvs_3d, "Mesh_EDV_Biplane": edvs_biplane, "Mesh_ESV_Biplane": esvs_biplane})
        
        if save:
            logger.info("Saving computed IoU and Dice values...")
            
            # Save IoU values
            IoU_csv_file = self.log_dir / iou_save_filename
            IoU_df.to_csv(IoU_csv_file, index=False)

            # Save Dice values
            Dice_csv_file = self.log_dir / dice_save_filename
            Dice_df.to_csv(Dice_csv_file, index=False)

            if compute_ef:
                logger.info("Saving computed EF values...")
                EF_csv_file = self.log_dir / ef_save_filename
                EF_df.to_csv(EF_csv_file, index=False)

            logger.info("Saved!")
        
        if compute_ef:
            return IoU_df, Dice_df, EF_df
        else:
            return IoU_df, Dice_df




