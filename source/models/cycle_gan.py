from glob import glob
import logging, sys, time, json, gc
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
from source.echo_utils import load_data_echonet, get_train_dataset, get_video_as_dataset, save_video
from source.models.echo_ae import EchoAutoencoderModel
from source.models.ops import Conv
from source.shape_model_utils import overwrite_vtkpoly, frames_to_vid, echo_mesh_4ch_overlay, save_feats_as_vtps, echo_edf_mesh_edf
from source.shape_model_utils import compute_volumes_feats, generate_seg_maps, compute_ef_mesh_vid, mesh_echo_iou_and_dice, mesh_ef_disks_method
from source.shape_model_utils import save_mesh_vid

from source.constants import ROOT_LOGGER_STR, ECHO_DATA_PARAMS, CONRAD_DATA_PARAMS, CONRAD_VIEWS, CONRAD_SEG_MAPS_SIZE

parallel_processes = []  # cpu processes for videos and volume plots generation

# get the logger
logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

LEOMED_RUN = "--leomed" in sys.argv  # True if running on LEOMED


# ------------------------------------------------------- Generators --------------------------------------------------------
class EchoToMeshGen(Model):

    def __init__(self, dense_layers, latent_space_mesh_dim, name='Gen_G', **kwargs):
        super(EchoToMeshGen, self).__init__(name=name, **kwargs)

        # make sure the last dense layer outputs a vector of size "nb of mesh's shape params"
        assert dense_layers[-1]['units'] == (latent_space_mesh_dim - 2)

        self.dense_layers = []
        self.batch_norms = []
        self.activations = []
        for i, dense_layer in enumerate(dense_layers):
            # get if batch_norm used
            batch_norm = dense_layer.get("batch_norm", False)
            # remove the key batch_norm from dense_layer dict (not using pop to be able to call save_me without issues)
            dense_layer_new = {}
            for k in dense_layer:
                if k != 'batch_norm':
                    dense_layer_new[k] = dense_layer[k]
            dense_layer = dense_layer_new
            # get activation function
            act_str = dense_layer['activation']['name'].lower()  # get activation in lowercase
            if act_str == 'leakyrelu':
                activation = LeakyReLU(alpha=dense_layer['activation']['alpha'])
            elif act_str == 'prelu':
                activation = PReLU()
            else:
                activation = tf.keras.activations.get(dense_layer['activation']['name'])

            # Linear Dense
            dense_layer['activation'] = 'linear'
            dense = Dense(**dense_layer)
            self.dense_layers.append(dense)

            if batch_norm:  # apply batch_norm
                self.batch_norms.append(BatchNormalization())
            else:  # pass through
                self.batch_norms.append(None)

            self.activations.append(activation)

        # value to add to echo's phase to get the mesh phase
        self.echo_to_mesh_phase_diff = tf.Variable(0., trainable=True, name="echo_to_mesh_phase_converter")

        # self.dense = Dense(latent_space_dim_echo, activation='sigmoid', name='gen_e_dense')

    def call(self, inputs, training=False):
        """
        inputs is expected to be of shape [batch_size, latent_space_echo_dim]
        """
        freqs = inputs[:, 0, None]
        echo_phases = inputs[:, 1, None]
        h = inputs[:, 2:]  # echo shape params

        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](h)
            if self.batch_norms[i] is not None:
                h = self.batch_norms[i](h, training=training)
            h = self.activations[i](h)
        mesh_phases = echo_phases + self.echo_to_mesh_phase_diff

        output = tf.concat([freqs, mesh_phases, h], axis=1)

        return output


class MeshToEchoGen(Model):

    def __init__(self, dense_layers, latent_space_echo_dim, name='Gen_F', **kwargs):
        super(MeshToEchoGen, self).__init__(name=name, **kwargs)

        # make sure the last dense layer outputs a vector of size "nb of echo's shape params"
        assert dense_layers[-1]['units'] == (latent_space_echo_dim - 2)

        self.dense_layers = []
        self.batch_norms = []
        self.activations = []
        for i, dense_layer in enumerate(dense_layers):
            # get if batch_norm used
            batch_norm = dense_layer.get("batch_norm", False)
            # remove the key batch_norm from dense_layer dict (not using pop to be able to call save_me without issues)
            dense_layer_new = {}
            for k in dense_layer:
                if k != 'batch_norm':
                    dense_layer_new[k] = dense_layer[k]
            dense_layer = dense_layer_new
            # get activation function
            act_str = dense_layer['activation']['name'].lower()  # get activation in lowercase
            if act_str == 'leakyrelu':
                activation = LeakyReLU(alpha=dense_layer['activation']['alpha'])
            elif act_str == 'prelu':
                activation = PReLU()
            else:
                activation = tf.keras.activations.get(dense_layer['activation']['name'])

            # Linear Dense
            dense_layer['activation'] = 'linear'
            dense = Dense(**dense_layer)
            self.dense_layers.append(dense)

            if batch_norm:  # apply batch_norm
                self.batch_norms.append(BatchNormalization())
            else:  # pass through
                self.batch_norms.append(None)

            self.activations.append(activation)

        # value to add to mesh's phase to get the echo phase
        self.mesh_to_echo_phase_diff = tf.Variable(0., trainable=True, name="mesh_to_echo_phase_converter")

    def call(self, inputs, training=False):
        """
        inputs is expected to be of shape [batch_size, latent_space_mesh_dim]
        """
        freqs = inputs[:, 0, None]
        mesh_phases = inputs[:, 1, None]
        h = inputs[:, 2:]  # mesh shape params

        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](h)
            if self.batch_norms[i] is not None:
                h = self.batch_norms[i](h, training=training)
            h = self.activations[i](h)
        echo_phases = mesh_phases + self.mesh_to_echo_phase_diff  # 1 - mesh_phases because mesh phases are subtracted whereas they should be added

        output = tf.concat([freqs, echo_phases, h], axis=1)

        return output


# ---------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------ Discriminators -----------------------------------------------------
class MeshLatentDisc(Model):

    def __init__(self, dense_layers, name='Disc_MeshLatent', **kwargs):
        super(MeshLatentDisc, self).__init__(name=name, **kwargs)

        assert dense_layers[-1]['units'] == 1  # one value output, logit where sigmoid(logit) = softmax(logit) = probability of being real

        self.dense_layers = []
        self.batch_norms = []
        self.activations = []
        for i, dense_layer in enumerate(dense_layers):
            # get if batch_norm used
            batch_norm = dense_layer.get("batch_norm", False)
            # remove the key batch_norm from dense_layer dict (not using pop to be able to call save_me without issues)
            dense_layer_new = {}
            for k in dense_layer:
                if k != 'batch_norm':
                    dense_layer_new[k] = dense_layer[k]
            dense_layer = dense_layer_new
            # get activation function
            act_str = dense_layer['activation']['name'].lower()  # get activation in lowercase
            if act_str == 'leakyrelu':
                activation = LeakyReLU(alpha=dense_layer['activation']['alpha'])
            elif act_str == 'prelu':
                activation = PReLU()
            else:
                activation = tf.keras.activations.get(dense_layer['activation']['name'])

            # Linear Dense
            dense_layer['activation'] = 'linear'
            dense = Dense(**dense_layer)
            self.dense_layers.append(dense)

            if batch_norm:  # apply batch_norm
                self.batch_norms.append(BatchNormalization())
            else:  # pass through
                self.batch_norms.append(None)

            self.activations.append(activation)

    def call(self, inputs, training=False):
        """
        inputs is expected to be of shape [batch_size, latent_space_mesh_dim]
        """
        h = inputs[:, 2:]  # apply discriminator on shape params only
        # Question: apply discriminator only on shape params + phase or whole vector? (For now latter)
        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](h)
            if self.batch_norms[i] is not None:
                h = self.batch_norms[i](h, training=training)
            h = self.activations[i](h)

        return h  # return logit, sigmoid(logit) = softmax(logit) = probability of being real


class EchoLatentDisc(Model):

    def __init__(self, dense_layers, name='Disc_EchoLatent', **kwargs):
        super(EchoLatentDisc, self).__init__(name=name, **kwargs)

        assert dense_layers[-1]['units'] == 1  # one value output, logit where sigmoid(logit) = softmax(logit) = probability of being real

        self.dense_layers = []
        self.batch_norms = []
        self.activations = []
        for i, dense_layer in enumerate(dense_layers):
            # get if batch_norm used
            batch_norm = dense_layer.get("batch_norm", False)
            # remove the key batch_norm from dense_layer dict (not using pop to be able to call save_me without issues)
            dense_layer_new = {}
            for k in dense_layer:
                if k != 'batch_norm':
                    dense_layer_new[k] = dense_layer[k]
            dense_layer = dense_layer_new
            # get activation function
            act_str = dense_layer['activation']['name'].lower()  # get activation in lowercase
            if act_str == 'leakyrelu':
                activation = LeakyReLU(alpha=dense_layer['activation']['alpha'])
            elif act_str == 'prelu':
                activation = PReLU()
            else:
                activation = tf.keras.activations.get(dense_layer['activation']['name'])

            # Linear Dense
            dense_layer['activation'] = 'linear'
            dense = Dense(**dense_layer)
            self.dense_layers.append(dense)

            if batch_norm:  # apply batch_norm
                self.batch_norms.append(BatchNormalization())
            else:  # pass through
                self.batch_norms.append(None)

            self.activations.append(activation)

    def call(self, inputs, training=False):
        """
        inputs is expected to be of shape [batch_size, latent_space_echo_dim]
        """
        h = inputs[:, 2:]  # apply discriminator on shape params only
        # Question: apply discriminator only on shape params + phase or whole vector? (For now latter)
        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](h)
            if self.batch_norms[i] is not None:
                h = self.batch_norms[i](h, training=training)
            h = self.activations[i](h)

        return h  # return logit, sigmoid(logit) = softmax(logit) = probability of being real


# ---------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------- Discriminators (Old) --------------------------------------------------
class MeshDiscriminatorOneFrame(Layer):

    def __init__(self, L, D, params, name='discriminator_m', **kwargs):
        super(DiscriminatorS, self).__init__(name=name, **kwargs)

        self.params = params
        self.conv_params = params['conv_layers']

        # down-sampling matrices
        self.D = [tf.Variable(down.toarray(),
                              name=f"disc_s_d_sampling_mat_{i}",
                              trainable=False,
                              dtype=tf.float32)
                  for i, down in enumerate(D)]

        self.D_b = [tf.Variable(
            tf.zeros([d.shape[0], p['channels']]),
            trainable=False,
            name=f"disc_s_d_sampling_bias_{i}",
            dtype=tf.float32)
            for i, (d, p) in enumerate(zip(D, self.conv_params))]

        # input matrices to conv layers
        self.L = [tf.convert_to_tensor(lap.toarray(),
                                       dtype=tf.float32,
                                       name=f"disc_s_laplacian_{i}")
                  for i, lap in enumerate(L)]

        # Graph-Convolution
        self.conv_layers = []
        for i, layer_params in enumerate(self.conv_params):
            layer = utils.get_conv_layer(layer_params.get('name'))
            self.conv_layers.append(layer(**layer_params))

        self.fc = Dense(1, name="disc_s_dense")

    def call(self, inputs, training=False):
        """

        :param inputs: a mesh consisting of adj. matrix (NxN) and
        feature matrix (NxF)
        :param training: if it's training or validation
        :return decision: true or wrong mesh
        """

        adj, features = inputs
        for i in range(len(self.D)):
            assert features.shape[1] == self.L[i].shape[0], \
                f"incompatible shapes between feature matrix {features.shape}" \
                f"and graph structure matrix {self.L[i].shape}"

            features = self.conv_layers[i]((features, self.L[i]))

            d = tf.repeat(tf.expand_dims(self.D[i], axis=0),
                          tf.shape(features)[0], axis=0)  # match batch dim

            features = tf.matmul(d, features) + self.D_b[i]

        features = tf.reshape(
            features,
            shape=[-1, self.conv_params[-1]['channels'] * self.D[-1].shape[0]])

        decision = self.fc(features)

        return decision


class EchoDiscriminatorDecoderOneFrame(Layer):

    def __init__(self, name='discriminator_e', **kwargs):
        super(DiscriminatorE, self).__init__(name=name, **kwargs)

        # Convolution
        # 8 filters (channels) conv with filter sizes 4x4 and stride 2x2, no batch norm
        self.conv1 = Conv(8, 4, 2, activation='relu', batch_normalisation=False, name='disc_e_conv1')
        # 16 filters (channels) conv with filter sizes 4x4 and stride 2x2, no batch norm
        self.conv2 = Conv(16, 4, 2, activation='relu', batch_normalisation=False, name='disc_e_conv2')
        # 16 filters (channels) conv with filter sizes 4x4 and stride 2x2, no batch norm
        self.conv3 = Conv(16, 4, 2, activation='relu', batch_normalisation=False, name='disc_e_conv3')
        # 16 filters (channels) conv with filter sizes 4x4 and stride 2x2, no batch norm
        self.conv4 = Conv(16, 4, 2, activation='relu', batch_normalisation=False, name='disc_e_conv4')
        # probability of being true or fake
        self.dense = Dense(1, name='disc_e_dense_1')  # linear activation

    def call(self, inputs, training=False):
        """
        inputs is expected to be of shape [batch_size, height, width, n_views]
        """

        h = self.conv1(inputs, training=training)  # (batch_size, 55, 55, 8)
        h = self.conv2(h, training=training)  # (batch_size, 26, 26, 16)
        h = self.conv3(h, training=training)  # (batch_size, 12, 12, 16)
        h = self.conv4(h, training=training)  # (batch_size, 5, 5, 16)

        h = tf.reshape(h, shape=[-1, int(np.prod(h.get_shape()[1:]))])  # flatten: (batch_size, 400)

        decision = self.dense(h)  # logit = l where softmax(l) = p = probability of being real, 1 - p = probability of being fake
        # sigmoid(x) = 1/(1+e^-x)

        return decision


# ---------------------------------------------------------------------------------------------------------------------------

class CycleGan(Model):

    def __init__(self, echo_ae, mesh_ae, mesh_ef_pred, log_dir, echo_data_dir, training_params, model_params, save_metrics=True, name='cycle_gan', **kwargs):

        super(CycleGan, self).__init__(name=name, **kwargs)

        # save experiment path in self
        self.log_dir = log_dir  # current experiment log dir (datetime dir)
        self.echo_data_dir = echo_data_dir

        # save training and model parameters from config file in self
        self.training_params = training_params
        self.model_params = model_params

        # save models
        self.echo_ae = echo_ae
        self.mesh_ae = mesh_ae
        self.mesh_ef_pred = mesh_ef_pred

        # build the CycleGan model
        self.build_model(model_params)

        # https://rafayak.medium.com/how-do-tensorflow-and-keras-implement-binary-classification-and-the-binary-cross-entropy-function-e9413826da7
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # self.loss_obj = tf.keras.losses.MeanSquaredError()

        if save_metrics:
            self.create_metrics_writers()

        logger.info("Cycle GAN model initialized and built.")

    def build_model(self, model_params):
        latent_space_mesh_dim = 18
        latent_space_echo_dim = 128
        self.mesh_gen = EchoToMeshGen(dense_layers=model_params['mesh_gen']['dense_layers'], latent_space_mesh_dim=latent_space_mesh_dim)
        self.echo_gen = MeshToEchoGen(dense_layers=model_params['echo_gen']['dense_layers'], latent_space_echo_dim=latent_space_echo_dim)
        self.mesh_disc = MeshLatentDisc(dense_layers=model_params['mesh_disc']['dense_layers'])
        self.echo_disc = EchoLatentDisc(dense_layers=model_params['echo_disc']['dense_layers'])

        # call each model to init its variables
        fake_mesh_latent = self.mesh_gen(tf.zeros(shape=(1, latent_space_echo_dim)))
        fake_echo_latent = self.echo_gen(tf.zeros(shape=(1, latent_space_mesh_dim)))
        self.mesh_disc(fake_mesh_latent)
        self.echo_disc(fake_echo_latent)

    def create_metrics_writers(self):

        train_metrics = ['cycle_loss_lambda', 'ef_loss_lambda', 'total_loss', 'echo_ae_loss', 'echo_ae_rec_error', 'mesh_disc_loss', 'mesh_disc_real_loss',
                         'mesh_disc_fake_loss', 'echo_disc_loss', 'echo_disc_real_loss', 'echo_disc_fake_loss', 'mesh_gen_loss', 'echo_gen_loss',
                         'mesh_gen_total_loss', 'echo_gen_total_loss', 'total_cycle_loss', 'mesh_cycle_loss', 'echo_cycle_loss', 'ef_loss',
                         'ef_loss_echo_to_mesh', 'ef_loss_mesh_to_mesh', 'mean_heart_rate']
        val_metrics = ['ef_loss', 'ef_loss_echo_to_mesh', 'ef_loss_mesh_to_mesh']
        val_exps_metrics = ['ef_loss', 'ef_loss_echo_to_mesh']

        self._val_metrics = dict()
        self._val_exps_metrics = dict()
        self._train_metrics = dict()

        train_log_dir = self.log_dir / 'metrics' / 'train'
        val_log_dir = self.log_dir / 'metrics' / 'validation'
        val_exps_log_dir = self.log_dir / 'metrics' / 'validation_exps'
        self._train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
        self._val_summary_writer = tf.summary.create_file_writer(str(val_log_dir))
        self._val_exps_summary_writer = tf.summary.create_file_writer(str(val_exps_log_dir))

        for metric in train_metrics:
            self._train_metrics[metric] = tf.keras.metrics.Mean(metric)

        for metric in val_metrics:
            self._val_metrics[metric] = tf.keras.metrics.Mean(metric)

        for metric in val_exps_metrics:
            self._val_exps_metrics[metric] = tf.keras.metrics.Mean(metric)

    @tf.function
    def call(self, real_echo_latents, real_mesh_latents, training=False):

        out_echo_to_mesh = self.translate_echo_to_mesh(real_echo_latents, training=training)
        out_mesh_to_echo = self.translate_mesh_to_echo(real_mesh_latents, training=training)

        return (out_echo_to_mesh, out_mesh_to_echo)

    def translate_echo_to_mesh(self, real_echo_latents, training=False):
        # ECHO -> MESH
        # translate echo latent to mesh latent with freq and phase
        fake_mesh_latents = self.mesh_gen(real_echo_latents, training=training)
        # translate back mesh latent to echo latent with freq and phase
        cycled_echo_latents = self.echo_gen(fake_mesh_latents, training=training)
        # pass the fake_mesh_latents through mesh decoder and get corresponding mesh feats, do not train decoder

        return fake_mesh_latents, cycled_echo_latents

    def translate_mesh_to_echo(self, real_mesh_latents, training=False):
        # MESH -> ECHO
        # translate mesh latent to echo latent with freq and phase
        fake_echo_latents = self.echo_gen(real_mesh_latents, training=training)
        # translate back echo latent to mesh latent with freq and phase
        cycled_mesh_latents = self.mesh_gen(fake_echo_latents, training=training)

        return fake_echo_latents, cycled_mesh_latents

    # --------------------------------------------------- Computing losses ----------------------------------------------
    # TODO: consider identity loss (apply generator functions to target sample)

    def mse(self, y_true, y_pred):
        mse = tf.reduce_mean(losses.mean_squared_error(y_true, y_pred))
        return mse

    def mae(self, y_true, y_pred):
        mae = tf.reduce_mean(losses.mean_absolute_error(y_true, y_pred))
        return mae

    def _discriminator_loss(self, real_logits, generated_logits):
        real_loss = self.loss_obj(tf.ones_like(real_logits), real_logits)

        generated_loss = self.loss_obj(tf.zeros_like(generated_logits), generated_logits)

        true_vals = tf.concat([tf.ones_like(real_logits), tf.zeros_like(generated_logits)], axis=0)
        pred_vals = tf.concat([real_logits, generated_logits], axis=0)

        all_data_loss = self.loss_obj(true_vals, pred_vals)

        return all_data_loss, real_loss, generated_loss

    def _generator_loss(self, generated_logits):
        return self.loss_obj(tf.ones_like(generated_logits), generated_logits)

    def _cycle_loss(self, real, cycled):

        cycle_loss_str = self.training_params['cycle_loss_str']
        cycle_loss = 0
        if cycle_loss_str == 'l1':
            cycle_loss = tf.reduce_mean(losses.mean_absolute_error(real, cycled))
        elif cycle_loss_str == 'l2':
            cycle_loss = tf.reduce_mean(losses.mean_squared_error(real, cycled))
        else:
            raise NotImplementedError()

        return cycle_loss

    def _ef_loss(self, true_echo_efs, true_mesh_efs, fake_mesh_latents, cycled_mesh_latents):
        # ef losses
        ef_loss = 0.0
        ef_loss_echo_to_mesh = 0
        ef_loss_mesh_to_mesh = 0
        # convert to range [0.0, 1.0]
        true_echo_efs = true_echo_efs / 100.0
        true_mesh_efs = true_mesh_efs / 100.0
        # add ef loss
        fake_mesh_pred_efs = tf.math.abs(self.mesh_ef_pred(fake_mesh_latents))
        cycled_mesh_pred_efs = tf.math.abs(self.mesh_ef_pred(cycled_mesh_latents))

        ef_loss_str = self.training_params["ef_loss_str"]
        # skip ef_loss if ef_loss_str != "l1" or "l2"
        if ef_loss_str == "l1":
            ef_loss_echo_to_mesh += self.mae(true_echo_efs, fake_mesh_pred_efs)
            ef_loss_mesh_to_mesh += self.mae(true_mesh_efs, cycled_mesh_pred_efs)
        elif ef_loss_str == "l2":
            ef_loss_echo_to_mesh += self.mse(true_echo_efs, fake_mesh_pred_efs)
            ef_loss_mesh_to_mesh += self.mse(true_mesh_efs, cycled_mesh_pred_efs)

        return ef_loss_echo_to_mesh, ef_loss_mesh_to_mesh

    def _loss(self, inputs, cycle_loss_lambda):
        # split the input based on values that go together
        real_meshes_logits, fake_meshes_logits = inputs[0], inputs[1]
        real_echos_logits, fake_echos_logits = inputs[2], inputs[3]
        real_echo_latents, cycled_echo_latents = inputs[4], inputs[5]
        real_mesh_latents, cycled_mesh_latents = inputs[6], inputs[7]

        # compute discriminator loss on meshes and on echos
        mesh_disc_loss, mesh_disc_real_loss, mesh_disc_fake_loss = self._discriminator_loss(real_meshes_logits, fake_meshes_logits)
        echo_disc_loss, echo_disc_real_loss, echo_disc_fake_loss = self._discriminator_loss(real_echos_logits, fake_echos_logits)
        # compute generator loss on meshes and echos
        mesh_gen_loss = self._generator_loss(fake_meshes_logits)
        echo_gen_loss = self._generator_loss(fake_echos_logits)
        # compute cycle loss on echos and meshes
        echo_cycle_loss = self._cycle_loss(real_echo_latents, cycled_echo_latents)
        mesh_cycle_loss = self._cycle_loss(real_mesh_latents, cycled_mesh_latents)
        # total cycle loss        
        total_cycle_loss = cycle_loss_lambda * echo_cycle_loss + cycle_loss_lambda * mesh_cycle_loss

        # mesh generator tries to minimize "mesh_gen_loss" (computed with mesh_disc on fake logits) to fool mesh discriminator
        # to output "real" on fake generated meshes
        # it also tries to minimize "echo_cycle_loss" so that it produces better "fake meshes" from "real echos"
        # it also tries to minimize "mesh_cycle_loss" so that it produces better "fake meshes" from "fake echos"
        mesh_gen_total_loss = mesh_gen_loss + total_cycle_loss
        # echo generator tries to minimize "echo_gen_loss" (computed with echo_disc on fake logits) to fool echo discriminator
        # to output "real" on fake generated echos
        # it also tries to minimize "mesh_cycle_loss" so that it produces better "fake echos" from "real meshes"
        # it also tries to minimize "echo_cycle_loss" so that it produces better "fake echos" from "fake meshes"
        echo_gen_total_loss = echo_gen_loss + total_cycle_loss
        # total loss of generators
        total_gen_loss = mesh_gen_loss + echo_gen_loss + total_cycle_loss

        # total loss
        total_loss = total_gen_loss + mesh_disc_loss + echo_disc_loss

        return total_loss, total_gen_loss, mesh_disc_loss, mesh_disc_real_loss, mesh_disc_fake_loss, echo_disc_loss, echo_disc_real_loss, echo_disc_fake_loss, mesh_gen_loss, echo_gen_loss, mesh_gen_total_loss, echo_gen_total_loss, total_cycle_loss, mesh_cycle_loss, echo_cycle_loss

    def _reconstruction_error(self, true_echo_frames, pred_echo_frames, row_lengths):
        # compute reconstruction error
        loss = 0
        which_loss = self.training_params['echo_rec_loss_str']
        true_echo_frames = tf.RaggedTensor.from_row_lengths(true_echo_frames, row_lengths=row_lengths)
        pred_echo_frames = tf.RaggedTensor.from_row_lengths(pred_echo_frames, row_lengths=row_lengths)
        if which_loss == 'l1':
            batch_error = tf.reduce_mean(tf.abs(true_echo_frames - pred_echo_frames), axis=[-1, -2, -3])
            loss = tf.reduce_mean(batch_error)
        elif which_loss == 'l2':
            batch_error = tf.reduce_mean((true_echo_frames - pred_echo_frames) ** 2, axis=[-1, -2, -3])
            loss = tf.reduce_mean(batch_error)
        else:
            raise NotImplementedError()
        return loss

    def _parameter_regularisation(self, params):
        # regularize the heart shape params only, not the trajectory params
        # we use L2 regularization (i,e sum squared of these params)
        batch_reg = tf.reduce_mean(params[:, 2:] ** 2, axis=-1)
        return tf.reduce_mean(batch_reg)

    # -------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- Training ----------------------------------------------
    @tf.function
    def _train_step(self, real_echo_latents, real_mesh_latents, true_echo_efs, true_mesh_efs, cycle_loss_lambda, ef_loss_lambda):

        # weights to be updated for each model
        mesh_gen_vars = self.mesh_gen.trainable_variables  # variables of the echo to mesh generator
        echo_gen_vars = self.echo_gen.trainable_variables  # variables of the mesh to echo generator
        mesh_disc_vars = self.mesh_disc.trainable_variables  # variables of the mesh discriminator
        echo_disc_vars = self.echo_disc.trainable_variables  # variables of the echo discriminator
        # echo_ae_vars = self.echo_ae.trainable_variables # variables of the echo ae encoder

        with tf.GradientTape(watch_accessed_variables=False) as mesh_gen_tape, tf.GradientTape(
                watch_accessed_variables=False) as echo_gen_tape, tf.GradientTape(watch_accessed_variables=False) as mesh_disc_tape, tf.GradientTape(
                watch_accessed_variables=False) as echo_disc_tape, tf.GradientTape(watch_accessed_variables=False) as echo_ae_tape:
            # specify to each tape wich vars it should watch
            mesh_gen_tape.watch(mesh_gen_vars)
            echo_gen_tape.watch(echo_gen_vars)
            mesh_disc_tape.watch(mesh_disc_vars)
            echo_disc_tape.watch(echo_disc_vars)
            # echo_ae_tape.watch(echo_ae_vars)

            # row_lengths = echo_times.row_lengths()
            # real_echo_latents = self.echo_ae.encode(echo_times, echo_frames, return_freq_phase=True, training=True)
            # rec_echo_frames = self.echo_ae.decode2(real_echo_latents, echo_times.values, row_lengths, passed_freqs_phases=True, training=True)
            # run model end-to-end in train mode:
            # --> translate echo_latent to mesh_latent and cycled_echo_latent
            # ---> tranlate mesh_latent to echo_latent and cycled_mesh_latent
            out_echo_to_mesh, out_mesh_to_echo = self.call(real_echo_latents, real_mesh_latents, training=True)
            fake_mesh_latents, cycled_echo_latents = out_echo_to_mesh
            fake_echo_latents, cycled_mesh_latents = out_mesh_to_echo
            # get mesh discriminator output on both real and fake echo latents
            real_meshes_logits = self.mesh_disc(real_mesh_latents, training=True)
            fake_meshes_logits = self.mesh_disc(fake_mesh_latents, training=True)
            # get echo discriminator output on both real and fake echo latents
            real_echos_logits = self.echo_disc(real_echo_latents, training=True)
            fake_echos_logits = self.echo_disc(fake_echo_latents, training=True)

            # gather inputs to the loss computing function
            inputs = (real_meshes_logits, fake_meshes_logits,
                      real_echos_logits, fake_echos_logits,
                      real_echo_latents, cycled_echo_latents,
                      real_mesh_latents, cycled_mesh_latents)

            # compute losses
            losses = self._loss(inputs, cycle_loss_lambda)
            total_loss, total_gen_loss, mesh_disc_loss, mesh_disc_real_loss, mesh_disc_fake_loss, echo_disc_loss, echo_disc_real_loss, echo_disc_fake_loss, mesh_gen_loss, echo_gen_loss, mesh_gen_total_loss, echo_gen_total_loss, total_cycle_loss, mesh_cycle_loss, echo_cycle_loss = losses
            # compute ef_loss
            # ef_loss, ef_loss_echo_to_mesh, ef_loss_mesh_to_mesh = 0, 0, 0
            ef_loss_echo_to_mesh, ef_loss_mesh_to_mesh = self._ef_loss(true_echo_efs, true_mesh_efs, fake_mesh_latents, cycled_mesh_latents)
            ef_loss = ef_loss_echo_to_mesh

            # add ef_loss
            mesh_gen_total_loss += ef_loss_lambda * ef_loss_echo_to_mesh
            echo_gen_total_loss += ef_loss_lambda * ef_loss_mesh_to_mesh

            total_loss += ef_loss_lambda * (ef_loss_echo_to_mesh + ef_loss_mesh_to_mesh)

            # echo ae loss
            # echo_ae_rec_error = self._reconstruction_error(echo_frames.values, rec_echo_frames, row_lengths)
            # echo_ae_reg = self._parameter_regularisation(real_echo_latents)
            # echo_ae_loss = echo_ae_rec_error + echo_ae_reg
            # echo_ae_loss += ef_loss
            # echo_ae_loss += total_cycle_loss
            # total_loss += (echo_ae_rec_error + echo_ae_reg)

            # experimental losses (to help mesh_gen decrease ef_loss)
            mesh_gen_l = mesh_gen_total_loss
            echo_gen_l = echo_gen_total_loss
            mesh_disc_l = mesh_disc_loss
            echo_disc_l = echo_disc_loss
            # echo_ae_l = echo_ae_loss

        # get gradients for weights of each model
        mesh_gen_grads = mesh_gen_tape.gradient(mesh_gen_l, mesh_gen_vars)
        echo_gen_grads = echo_gen_tape.gradient(echo_gen_l, echo_gen_vars)
        mesh_disc_grads = mesh_disc_tape.gradient(mesh_disc_l, mesh_disc_vars)
        echo_disc_grads = echo_disc_tape.gradient(echo_disc_l, echo_disc_vars)
        # echo_ae_grads = echo_ae_tape.gradient(echo_ae_l, echo_ae_vars)

        # apply gradients for each model using the corresponding optimizer
        self.mesh_gen_optimizer.apply_gradients(zip(mesh_gen_grads, mesh_gen_vars))
        self.echo_gen_optimizer.apply_gradients(zip(echo_gen_grads, echo_gen_vars))
        self.mesh_disc_optimizer.apply_gradients(zip(mesh_disc_grads, mesh_disc_vars))
        self.echo_disc_optimizer.apply_gradients(zip(echo_disc_grads, echo_disc_vars))
        # self.echo_ae_optimizer.apply_gradients(zip(echo_ae_grads, echo_ae_vars))

        mean_heart_rate = tf.reduce_mean(real_echo_latents[:, 0] * 60)

        losses = {'total_loss': total_loss,
                  'mesh_disc_loss': mesh_disc_loss,
                  'mesh_disc_real_loss': mesh_disc_real_loss,
                  'mesh_disc_fake_loss': mesh_disc_fake_loss,
                  'echo_disc_loss': echo_disc_loss,
                  'echo_disc_real_loss': echo_disc_real_loss,
                  'echo_disc_fake_loss': echo_disc_fake_loss,
                  'mesh_gen_loss': mesh_gen_loss,
                  'echo_gen_loss': echo_gen_loss,
                  'mesh_gen_total_loss': mesh_gen_total_loss,
                  'echo_gen_total_loss': echo_gen_total_loss,
                  'total_cycle_loss': total_cycle_loss,
                  'mesh_cycle_loss': mesh_cycle_loss,
                  'echo_cycle_loss': echo_cycle_loss,
                  'ef_loss_echo_to_mesh': ef_loss_echo_to_mesh,
                  'ef_loss_mesh_to_mesh': ef_loss_mesh_to_mesh,
                  'ef_loss': ef_loss,
                  'echo_ae_loss': 0,
                  'echo_ae_rec_error': 0,
                  'mean_heart_rate': mean_heart_rate
                  }

        return losses

    @tf.function
    def _val_step(self, real_echo_latents, real_mesh_latents, true_echo_efs, true_mesh_efs):
        # real_echo_latents = self.echo_ae.encode(echo_times, echo_frames, return_freq_phase=True, training=False)
        out_echo_to_mesh, out_mesh_to_echo = self.call(real_echo_latents, real_mesh_latents, training=False)
        fake_mesh_latents, _ = out_echo_to_mesh
        _, cycled_mesh_latents = out_mesh_to_echo

        # compute ef_loss
        ef_loss_echo_to_mesh, ef_loss_mesh_to_mesh = self._ef_loss(true_echo_efs, true_mesh_efs, fake_mesh_latents, cycled_mesh_latents)

        ef_loss = ef_loss_echo_to_mesh

        losses = {'ef_loss': ef_loss,
                  'ef_loss_echo_to_mesh': ef_loss_echo_to_mesh,
                  'ef_loss_mesh_to_mesh': ef_loss_mesh_to_mesh
                  }

        return losses

    def fit(self, mesh_datasets_enc, echo_datasets_enc, echo_datasets, echo_filenames, echo_datasets_edfs):
        global parallel_processes

        opt_early_stopping_metric = np.inf
        patience = self.training_params['patience']
        max_patience = self.training_params['max_patience']
        learning_rate_decay = self.training_params['decay_rate']

        count = epoch = epoch_step = global_step = 0
        opt_weights = self.get_weights()

        optimizers_params = self.training_params['optimizers']

        # log model optimizers params
        logger.info("\nOptimizers params:")
        logger.info(json.dumps(optimizers_params, sort_keys=False, indent=4))
        logger.info("\n")

        # create optimizers
        self.mesh_gen_optimizer = Adam(**optimizers_params['mesh_gen'])
        self.echo_gen_optimizer = Adam(**optimizers_params['echo_gen'])
        self.mesh_disc_optimizer = Adam(**optimizers_params['mesh_disc'])
        self.echo_disc_optimizer = Adam(**optimizers_params['echo_disc'])
        self.echo_ae_optimizer = Adam(**optimizers_params['echo_ae'])

        num_steps_summaries = self.training_params['num_steps_summaries']
        num_steps_val = self.training_params['num_steps_val']
        max_steps = self.training_params['max_steps']

        # log number of trainable weights before training
        logger.info("Start training...\n")

        logged_weights = False
        early_stopping = False
        run_exps = False

        t1_steps = time.time()  # start epoch timer

        self.epoch = epoch  # save epoch nb in self

        # get train and val datasets
        # echo_train, echo_val = echo_datasets["train"], echo_datasets["val"]
        mesh_train_enc, mesh_val_enc = mesh_datasets_enc["train"], mesh_datasets_enc["val"]
        echo_train_enc, echo_val_enc = echo_datasets_enc["train"], echo_datasets_enc["val"]

        echo_ef_index = ECHO_DATA_PARAMS.index("EF")
        save_original_echos = True
        train_g = True

        cycle_loss_lambda = float(self.training_params["cycle_loss_lambda"])
        ef_loss_lambda = float(self.training_params["ef_loss_lambda"])

        all_exps = ["Render", "Echo_rec", "Overlay", "EF_train", "EF_val", "IoU"]

        for b, (echo_b, mesh_b) in enumerate(zip(echo_train_enc, mesh_train_enc)):
            if global_step == max_steps or early_stopping:
                break

            # # echo batch
            # echo_times, echo_frames, echo_params = echo_b
            # echo_params = echo_params.to_tensor(default_value=0.0)
            # echo_efs = echo_params[:, echo_ef_index, None] # efs of shape (-1, 1) are in range [0, 100]
            # mesh latents batch
            mesh_latents, mesh_efs = mesh_b  # efs of shape (-1, 1) are in range [0, 100]
            # echo latents batch
            echo_latents, echo_efs = echo_b

            losses = self._train_step(echo_latents, mesh_latents, echo_efs, mesh_efs, tf.constant(cycle_loss_lambda), tf.constant(ef_loss_lambda))
            self._log_metrics(losses, self._train_metrics)
            early_stopping_train_loss = self._train_metrics['ef_loss'].result()

            if not logged_weights:
                # with self._train_summary_writer.as_default():
                #     echo_latents = self.echo_ae.encode(echo_times, echo_frames, return_freq_phase=True, training=False)
                #     tf.summary.graph(self.call.get_concrete_function(echo_latents, mesh_latents, tf.constant(True)).graph)
                self.log_weights()
                logged_weights = True

            if global_step % num_steps_summaries == 0:
                self._train_metrics['cycle_loss_lambda'](cycle_loss_lambda)  # log cycle_loss_lambda with train metrics
                self._train_metrics['ef_loss_lambda'](ef_loss_lambda)  # log ef_loss_lambda with train metrics

                # stop "steps timer" log train metrics and log "steps time"
                t2_steps = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2_steps - t1_steps)
                self.write_summaries(self._train_summary_writer, self._train_metrics, global_step, "Train")
                logger.info("{} steps done in {}:{}:{}\n".format(num_steps_summaries, h, m, s))

                # start steps timer again
                t1_steps = time.time()

                # reset metrics
                self.reset_metrics(train=True)

                with open('cycle_gan_run_exps.txt') as f:
                    lines = f.readlines()
                    processed_lines = []
                    for line in lines:
                        line = line.rstrip("\n")
                        processed_lines.append(line)
                    lines = processed_lines

                for exp in all_exps:
                    if exp in lines:
                        run_exps = True
                        break

                if run_exps:
                    logger.info(f"\nRunning plots and visualization exps...\n")
                else:
                    logger.info(f"\nSkipping plots and visualization exps...\n")

            if global_step % num_steps_val == 0:
                # Perform validation step
                logger.info("Computing validation error...")
                t1_val = time.time()  # start validation timer
                early_stopping_val_loss = 0
                for echo_val_b, mesh_val_b in zip(echo_val_enc, mesh_val_enc):
                    # # echo batch
                    # echo_times_val, echo_frames_val, echo_params_val = echo_val_b
                    # echo_params_val = echo_params_val.to_tensor(default_value=0.0)
                    # echo_efs_val = echo_params_val[:, echo_ef_index, None]
                    # mesh latents batch
                    mesh_latents_val, mesh_efs_val = mesh_val_b
                    # echo latents batch
                    echo_latents_val, echo_efs_val = echo_val_b
                    losses = self._val_step(echo_latents_val, mesh_latents_val, echo_efs_val, mesh_efs_val)
                    self._log_metrics(losses, self._val_metrics)
                    early_stopping_val_loss = self._val_metrics['ef_loss'].result()

                # stop validation timer, log validation metrics and log epoch time
                t2_val = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2_val - t1_val)
                self.write_summaries(self._val_summary_writer, self._val_metrics, global_step, "Validation")
                logger.info("Validation done in {}:{}:{}\n".format(h, m, s))

                # early stopping loss based on train data if ef lambda is not used.
                if ef_loss_lambda == 0:
                    early_stopping_loss = float(early_stopping_train_loss)
                else:
                    early_stopping_loss = float(early_stopping_val_loss)
                # if new validation loss worse than previous one:
                if early_stopping_loss >= opt_early_stopping_metric:
                    count += 1
                    logger.info(f"Validation EF loss did not improve from {opt_early_stopping_metric}. Counter: {count}")

                    if count == max_patience:
                        logger.info("Early Stopping")
                        early_stopping = True

                else:  # validation loss improved
                    logger.info(f"Validation loss ({early_stopping_loss}) improved, saving model.")
                    opt_early_stopping_metric = float(early_stopping_loss)  # convert from mean (metric) to float
                    opt_weights = self.get_weights()
                    count = 0

                    # save best model
                    self.save_me()

                    # run_exps = True
                    # lines = ["EF_val", "IoU"]

                # save last model
                self.save_me("last")

                # reset metrics
                self.reset_metrics(val=True)

            if run_exps:
                run_exps = False  # reset boolean
                self.run_experiments(lines, echo_datasets, echo_filenames, echo_datasets_edfs, global_step,
                                     write_summaries=True,
                                     save_original_echos=save_original_echos)
                save_original_echos = False  # only save once

                self.reset_metrics(val_exp=True)
                # start steps timer again
                t1_steps = time.time()

            global_step += 1  # increment nb total of steps

        # reset to optimal weights
        logger.info("Set optimal weights.")
        self.set_weights(opt_weights)
        lines = ["EF_val", "Render"]
        self.run_experiments(lines, echo_datasets, echo_filenames, echo_datasets_edfs, global_step,
                             write_summaries=True,
                             save_original_echos=save_original_echos)

    # ---------------------------------------------------- Experiments --------------------------------------------------
    def run_experiments(self, exps_list, echo_datasets, echo_filenames, echo_datasets_edfs, global_step, write_summaries=True, save_original_echos=True):
        if "Render" in exps_list or "Echo_rec" in exps_list or "Overlay" in exps_list:
            # Translate echo to mesh then render mesh
            set_names = ["train_render"]
            for set_name in set_names:
                logger.info(f"Visualizations on {set_name} dataset...")
                echo_dataset_render = echo_datasets[set_name]
                echo_dataset_filenames_render = echo_filenames[set_name]
                echo_dataset_edfs_render = echo_datasets_edfs[set_name]
                if "Render" in exps_list:
                    self.render_mesh_vid_from_echo(echo_dataset_render, echo_dataset_filenames_render, set_name, global_step, framerate_ds_factor=5)
                if "Echo_rec" in exps_list:
                    self.reconstruct_echo_vid(echo_dataset_render, echo_dataset_filenames_render, set_name, global_step, save_original=save_original_echos)

        if "EF_train" in exps_list:
            # compute EF vol and draw correlation plot on part of train_dataset
            echo_dataset_train_viz = echo_datasets["train_viz"]
            echo_filenames_train_viz = echo_filenames["train_viz"]
            self.echo_vs_mesh_ef(echo_dataset_train_viz, echo_filenames_train_viz, "train_viz", save=True, gen_plots=True, global_step=global_step)
        if "EF_val" in exps_list:
            # compute EF vol and draw correlation plot on val dataset
            echo_dataset_val = echo_datasets["val"]
            echo_filenames_val = echo_filenames["val"]
            self.echo_vs_mesh_ef(echo_dataset_val, echo_filenames_val, "val", save=True, write_summaries=write_summaries, gen_plots=True,
                                 global_step=global_step)


    # ----------------------------------------------- Mesh on Echo Overlay ----------------------------------------------
    def overlay_mesh_on_echo(self, echo_dataset, echo_dataset_filenames, echo_dataset_edfs, set_name, global_step):
        edf_pairs = self.echo_mesh_edf_pairs(echo_dataset, echo_dataset_filenames, echo_dataset_edfs, output_dir=None)
        overlay_out_dir = self.log_dir / "visualizations" / set_name / "echo_to_mesh_overlay" / f"GS{str(global_step).zfill(3)}"
        self.echo_vid_to_mesh_vid_edf_aligned(echo_dataset, echo_dataset_filenames, edf_pairs, output_dir=overlay_out_dir)

    def echo_mesh_edf_pairs(self, echo_dataset, echo_dataset_filenames, echo_edfs, output_dir):
        # convert each echo in the dataset to a mesh and computes the volume peaks of the mesh lv volumes
        # to determine the mesh EDF.
        # returns and saves to file the echo/mesh edf pairs for this model

        if output_dir is not None:
            echo_mesh_edf_pairs_file = output_dir / "EDF_pairs.csv"
            if echo_mesh_edf_pairs_file.exists():  # if EDF pairs file exists, read it
                logger.info("\nFound EDF pairs file, reading it...")
                echo_mesh_edf_pairs = pd.read_csv(echo_mesh_edf_pairs_file)
                # filter edf pairs to get only those of filenames in the list "echo_dataset_filenames"
                found_edf_pairs = echo_mesh_edf_pairs[echo_mesh_edf_pairs['FileName'].isin(echo_dataset_filenames)]
                if found_edf_pairs.shape[0] == len(echo_dataset_filenames):  # if found as many edf pairs as requested, return the edf pairs
                    logger.info("All EDF pairs found in file, returning read values...")
                    return found_edf_pairs
                else:
                    logger.info("Not all EDF pairs found, recomputing on whole dataset...")

        # otherwise, recompute over the whole dataset
        logger.info("\nComputing EDF pairs...")

        # dataset values
        mesh_dataset_scale = self.mesh_ae.data_handler.dataset_scale
        mesh_dataset_min = self.mesh_ae.data_handler.dataset_min
        mesh_ref_poly = self.mesh_ae.data_handler.reference_poly

        # parallel jobs
        max_parallel_jobs = mp.cpu_count()
        parallel_jobs = []
        manager = mp.Manager()
        return_dict = manager.dict()

        # vars for batch size
        batch_size = -1
        t_start = time.time()
        t1 = time.time()

        for b, (echo_times, echo_frames, echo_params) in enumerate(echo_dataset):  # b'th batch

            # get row lengths and row limits
            row_lengths = echo_times.row_lengths()
            row_limits = tf.cumsum(row_lengths).numpy()
            if batch_size == -1:  # get batch_size
                batch_size = row_lengths.shape[0]
                logger.info(f"Batch size: {batch_size}")

            # translate echo vids to mesh vids (transform relevant arrays to numpy)
            real_echo_latents = self.echo_ae.encode(echo_times, echo_frames, return_freq_phase=True, training=False)
            fake_mesh_latents, _ = self.translate_echo_to_mesh(real_echo_latents)
            fake_mesh_feats = self.mesh_ae.decode(fake_mesh_latents, echo_times.values, row_lengths, training=False).numpy()

            # process videos in parallel, each video in a parallel job
            i = 0
            for k, j in enumerate(row_limits):  # k'th video in the batch "b"
                mesh_feats = fake_mesh_feats[i:j]

                global_index = b * batch_size + k  # index of the video in the whole dataset

                echo_filename = echo_dataset_filenames[global_index]
                echo_edf = echo_edfs[global_index]

                args = (echo_edf, mesh_feats, mesh_ref_poly, mesh_dataset_scale, mesh_dataset_min, True, echo_filename, return_dict)
                p = mp.Process(target=echo_edf_mesh_edf, args=args)
                parallel_jobs.append(p)
                p.start()

                i = j

            # wait for parallel jobs to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, max_parallel_jobs)

            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done 1 batch of {batch_size} videos in {h}:{m}:{s}")
            t1 = time.time()

            if b % 10 == 0:  # print progress
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b} after {h}:{m}:{s}")

        # wait for all parallel jobs to complete
        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
        logger.info(f"Done all batches in {h}:{m}:{s}")

        mesh_edfs = []
        for filename in echo_dataset_filenames:
            mesh_edfs.append(return_dict[filename][1])

        edf_pairs = pd.DataFrame({"FileName": echo_dataset_filenames, "Echo_EDF": echo_edfs, "Mesh_EDF": mesh_edfs})

        if output_dir is not None:
            logger.info("Saving to csv...")
            output_dir.mkdir(parents=True, exist_ok=True)
            edf_pairs.to_csv(echo_mesh_edf_pairs_file, index=False)
            logger.info("Done!")

        return edf_pairs

    def echo_vid_to_mesh_vid_edf_aligned_temporally(self, echo_dataset, diff_edfs):

        batch_size = -1
        echo_ef_index = ECHO_DATA_PARAMS.index("EF")

        for b, (echo_times, echo_frames, echo_params) in enumerate(echo_dataset):  # b'th batch

            # get row lengths and row limits
            row_lengths = echo_times.row_lengths().numpy()
            row_limits = np.cumsum(row_lengths)
            if batch_size == -1:  # get batch_size
                batch_size = row_lengths.shape[0]
                logger.info(f"Batch size: {batch_size}")

            # create new timesteps for generating mesh video aligned with echo video
            i = 0
            mesh_times = []
            global_indices = []
            for k, j in enumerate(row_limits):  # k'th video in the batch "b"
                timesteps = echo_times.values[i:j]
                timestep = timesteps[1]
                final_timestep = timesteps[-1]

                global_index = b * batch_size + k  # index of the video in the whole dataset
                global_indices.append(global_index)
                diff_edf = diff_edfs[global_index]

                new_timesteps = timesteps
                if diff_edf > 0:  # echo edf larger than mesh edf
                    additional_timesteps = tf.cast(tf.range(-diff_edf, 0, 1), tf.float32) * timestep
                    to_remove = additional_timesteps.shape[0]
                    timesteps_clipped = timesteps[:-to_remove]
                    new_timesteps = tf.concat([additional_timesteps, timesteps_clipped], axis=0)
                elif diff_edf < 0:  # mesh edf larger than echo edf
                    additional_timesteps = tf.cast(tf.range(1, -diff_edf + 1, 1), tf.float32) * timestep + final_timestep
                    to_remove = additional_timesteps.shape[0]
                    timesteps_clipped = timesteps[to_remove:]
                    new_timesteps = tf.concat([timesteps_clipped, additional_timesteps], axis=0)
                mesh_times.append(new_timesteps)

                i = j

            mesh_times = tf.RaggedTensor.from_row_lengths(tf.concat(mesh_times, axis=0), row_lengths)

            # translate echo vids to mesh vids aligned in time (transform relevant arrays to numpy)
            real_echo_latents = self.echo_ae.encode(echo_times, echo_frames, return_freq_phase=True, training=False)
            fake_mesh_latents, _ = self.translate_echo_to_mesh(real_echo_latents)
            fake_mesh_feats = self.mesh_ae.decode(fake_mesh_latents, mesh_times.values, row_lengths, training=False).numpy()
            times_values = echo_times.values.numpy()
            echo_frames_values = echo_frames.values.numpy()

            echo_params = echo_params.to_tensor(default_value=0.0)
            echo_efs = echo_params[:, echo_ef_index].numpy()

            yield global_indices, row_limits, fake_mesh_feats, times_values, echo_frames_values, echo_efs

    def echo_vid_to_mesh_vid_edf_aligned(self, echo_dataset, echo_filenames, edf_pairs, output_dir=None, save_vtps=False, render_mesh_video=False):
        if output_dir is None:
            output_dir = self.log_dir

        logger.info("\nGenerating mesh slice on echo overlay videos aligned...")

        # read volume tracings file
        volume_tracings = utils.get_volume_tracings_df(self.echo_data_dir)

        # get "echo_edf - mesh_edf" in same order as filenames
        diff_edfs = []
        echo_edfs = {}
        for filename in echo_filenames:
            row = edf_pairs[edf_pairs["FileName"] == filename]
            diff_edfs.append(int(row["Echo_EDF"]) - int(row["Mesh_EDF"]))
            echo_edfs[filename] = int(row["Echo_EDF"])

        # mesh data handler
        mesh_data_handler = self.mesh_ae.data_handler

        # parallel jobs
        max_parallel_jobs = mp.cpu_count()
        parallel_jobs = []

        t_start = time.time()
        t1 = time.time()
        output_size = (250, 250)
        for b, (global_indices, row_limits, fake_mesh_feats, times_values, echo_frames_values, echo_efs) in enumerate(
                self.echo_vid_to_mesh_vid_edf_aligned_temporally(echo_dataset, diff_edfs)):
            batch_size = len(global_indices)

            i = 0
            for k, j in enumerate(row_limits):  # k'th video in the batch "b"
                echo_frames = echo_frames_values[i:j]
                mesh_feats = fake_mesh_feats[i:j]
                times = times_values[i:j]
                vid_duration = times[-1]

                global_index = global_indices[k]  # index of the video in the whole dataset
                echo_ef = echo_efs[k]

                echo_filename = echo_filenames[global_index]
                echo_edf = echo_edfs[echo_filename]
                echo_lax = utils.get_echo_long_axis_points(echo_filename, volume_tracings, new_size=output_size)[echo_edf]

                overlay_output_dir = output_dir
                overlay_vid_prefix = f"vid_{global_index}_name_{echo_filename}_EF_{echo_ef:.2f}"

                # echo_mesh_4ch_overlay(echo_filename, echo_frames, echo_edf, echo_lax, mesh_feats, mesh_data_handler, vid_duration, output_dir, overlay_vid_prefix, output_size)
                # exit(0)
                args = (echo_filename, echo_frames, echo_edf, echo_lax, mesh_feats, mesh_data_handler, vid_duration, overlay_output_dir, overlay_vid_prefix,
                        output_size)
                p = mp.Process(target=echo_mesh_4ch_overlay, args=args)
                parallel_jobs.append(p)
                p.start()

                if save_vtps:
                    vtps_output_dir = output_dir / "mesh_3d" / echo_filename
                    prefix = f"Mesh"

                    args = (mesh_feats, mesh_data_handler.reference_poly, vtps_output_dir, prefix)
                    p = mp.Process(target=save_feats_as_vtps, args=args)
                    parallel_jobs.append(p)
                    p.start()

                if render_mesh_video:
                    vid_out_dir = output_dir / "rendered_mesh_3d"
                    vid_name = f"vid_{global_index}_name_{echo_filename}"

                    args = (mesh_feats, mesh_data_handler, vid_duration, vid_out_dir, vid_name)
                    p = mp.Process(target=save_mesh_vid, args=args)
                    parallel_jobs.append(p)
                    p.start()

                i = j

            # wait for parallel jobs to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, max_parallel_jobs)

            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done 1 batch of {batch_size} videos in {h}:{m}:{s}")
            t1 = time.time()

            if b % 10 == 0:  # print progress
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b} after {h}:{m}:{s}")

        # wait for all parallel jobs to complete
        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
        logger.info(f"Done all batches in {h}:{m}:{s}")

    # -------------------------------------------------- Mesh Rendering -------------------------------------------------
    def render_mesh_vid_from_echo(self, echo_dataset, echo_filenames, set_name, global_step, framerate_ds_factor=1, output_dir=None, save_vtps=False):
        if output_dir is None:
            output_dir = self.log_dir / "visualizations" / "3d_mesh_from_echo" / f"GS{str(global_step).zfill(3)}"

        logger.info("\nRendering mesh videos from echo videos...")
        # mesh data handler
        mesh_data_handler = self.mesh_ae.data_handler

        # parallel jobs
        max_parallel_jobs = mp.cpu_count()
        parallel_jobs = []

        t_start = time.time()
        t1 = time.time()
        output_size = (250, 250)
        batch_size = -1
        for b, (echo_times, echo_frames, echo_params) in enumerate(echo_dataset):  # b'th batch

            # translate echo vids to mesh vids aligned in time (transform relevant arrays to numpy)
            real_echo_latents = self.echo_ae.encode(echo_times, echo_frames, return_freq_phase=True, training=False)
            fake_mesh_latents, _ = self.translate_echo_to_mesh(real_echo_latents)

            # decrease framerate of decoded fake meshes for faster processing
            echo_times_tensor = echo_times.to_tensor(default_value=-1.0)  # convert to tensor with default value -1
            max_len = echo_times_tensor.shape[1]
            indices = list(range(0, max_len, framerate_ds_factor))  # indices to take, take 1 frame for every "framerate_ds_factor" frames
            fake_mesh_times = tf.gather(echo_times_tensor, indices=indices, axis=1)  # downsampled echo times
            fake_mesh_times = tf.RaggedTensor.from_tensor(fake_mesh_times, padding=-1)  # convert back to RaggedTensor

            fake_mesh_feats = self.mesh_ae.decode(fake_mesh_latents, fake_mesh_times.values, fake_mesh_times.row_lengths(), training=False).numpy()

            # get row lengths and row limits
            row_lengths = fake_mesh_times.row_lengths().numpy()
            row_limits = np.cumsum(row_lengths)
            if batch_size == -1:  # get batch_size
                batch_size = row_lengths.shape[0]
                logger.info(f"Batch size: {batch_size}")

            # convert to numpy
            fake_mesh_times = fake_mesh_times.values.numpy()

            i = 0
            for k, j in enumerate(row_limits):  # k'th video in the batch "b"
                mesh_feats = fake_mesh_feats[i:j]
                mesh_times = fake_mesh_times[i:j]
                global_index = b * batch_size + k  # index of the video in the whole dataset
                echo_filename = echo_filenames[global_index]
                vid_duration = mesh_times[-1]

                vid_out_dir = output_dir / "vids"
                vid_name = f"vid_{global_index}_name_{echo_filename}"

                args = (mesh_feats, mesh_data_handler, vid_duration, vid_out_dir, vid_name)
                p = mp.Process(target=save_mesh_vid, args=args)
                parallel_jobs.append(p)
                p.start()

                if save_vtps:
                    vtps_out_dir = output_dir / "vtps" / echo_filename
                    prefix = f"Mesh"

                    args = (mesh_feats, mesh_data_handler.reference_poly, vtps_out_dir, prefix)
                    p = mp.Process(target=save_feats_as_vtps, args=args)
                    parallel_jobs.append(p)
                    p.start()

                i = j

            # wait for parallel jobs to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, max_parallel_jobs)

            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done 1 batch of {batch_size} videos in {h}:{m}:{s}")
            t1 = time.time()

            if b % 10 == 0:  # print progress
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b} after {h}:{m}:{s}")

        # wait for all parallel jobs to complete
        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
        logger.info(f"Done all batches in {h}:{m}:{s}")

    # ----------------------------------------------- Echo Reconstruction -----------------------------------------------
    def reconstruct_echo_vid(self, echo_dataset, echo_filenames, set_name, global_step, output_dir=None, save_original=False):
        if output_dir is None:
            output_dir = self.log_dir / "visualizations" / set_name / "echo_reconstructions"
            output_dir_echo = output_dir / "OriginalEchos"
            output_dir_echo_rec = output_dir / f"GS{str(global_step).zfill(3)}"

        logger.info("\nEncoding and reconstructing echo videos...")

        # parallel jobs
        max_parallel_jobs = mp.cpu_count()
        parallel_jobs = []

        t_start = time.time()
        t1 = time.time()
        batch_size = -1
        for b, (echo_times, echo_frames, echo_params) in enumerate(echo_dataset):  # b'th batch

            # translate echo vids to mesh vids aligned in time (transform relevant arrays to numpy)
            echo_latents = self.echo_ae.encode(echo_times, echo_frames, return_freq_phase=True, training=False)
            echo_vids_rec = self.echo_ae.decode2(echo_latents, echo_times.values, echo_times.row_lengths(), passed_freqs_phases=True, training=False)

            # get row lengths and row limits
            row_lengths = echo_times.row_lengths().numpy()
            row_limits = np.cumsum(row_lengths)
            if batch_size == -1:  # get batch_size
                batch_size = row_lengths.shape[0]
                logger.info(f"Batch size: {batch_size}")

            # convert to numpy
            echo_vids = echo_frames.values.numpy()
            echo_vids_rec = echo_vids_rec.numpy()
            echo_times = echo_times.values.numpy()

            i = 0
            for k, j in enumerate(row_limits):  # k'th video in the batch "b"
                echo_vid = echo_vids[i:j]
                echo_vid_rec = echo_vids_rec[i:j]
                echo_vid_times = echo_times[i:j]
                global_index = b * batch_size + k  # index of the video in the whole dataset
                echo_filename = echo_filenames[global_index]
                vid_duration = echo_vid_times[-1]

                vid_name = f"vid_{global_index}_name_{echo_filename}"

                args = (output_dir_echo_rec, echo_vid_rec, vid_duration, vid_name, True, False)
                p = mp.Process(target=frames_to_vid, args=args)
                parallel_jobs.append(p)
                p.start()

                if save_original:
                    args = (output_dir_echo, echo_vid, vid_duration, vid_name, True, False)
                    p = mp.Process(target=frames_to_vid, args=args)
                    parallel_jobs.append(p)
                    p.start()

                i = j

            # wait for parallel jobs to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, max_parallel_jobs)

            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done 1 batch of {batch_size} videos in {h}:{m}:{s}")
            t1 = time.time()

            if b % 10 == 0:  # print progress
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b} after {h}:{m}:{s}")

        # wait for all parallel jobs to complete
        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
        logger.info(f"Done all batches in {h}:{m}:{s}")

    # ----------------------------------------------------- IoU/Dice ----------------------------------------------------
    def dice_and_IoU_echo_mesh_aligned(self, echo_dataset, echo_filenames, edf_pairs, echo_data_dir, save_vtps=False):
        # read volume tracings file
        volume_tracings = pd.read_csv(str(echo_data_dir / 'EchoNet-Dynamic/VolumeTracings.csv'))  # file with volume tracing data as a pandas df

        # get "echo_edf - mesh_edf" in same order as filenames (for all echos in dataset)
        diff_edfs = []
        echo_edfs = []
        for filename in echo_filenames:
            row = edf_pairs[edf_pairs["FileName"] == filename]
            diff_edfs.append(int(row["Echo_EDF"]) - int(row["Mesh_EDF"]))
            echo_edfs.append(int(row["Echo_EDF"]))

        # for each echo filename, the tracing_data is a pair of triple (one for EDF and one for ESF)
        # where triple = (tracing_frame, lax, frame_nb)
        output_size = (112, 112)
        echo_tracings_data, skipped_echos = utils.generate_echo_tracings(echo_filenames, echo_edfs, volume_tracings, output_size)

        # mesh dataset handler
        mesh_data_handler = self.mesh_ae.data_handler

        # parallel jobs
        manager = mp.Manager()
        return_dict = manager.dict()
        max_parallel_jobs = mp.cpu_count()
        parallel_jobs = []

        t_start = time.time()
        t1 = time.time()
        for b, (global_indices, row_limits, fake_mesh_feats, times_values, echo_frames_values, echo_efs) in enumerate(
                self.echo_vid_to_mesh_vid_edf_aligned_temporally(echo_dataset, diff_edfs)):
            batch_size = len(global_indices)

            if b % 10 == 0:  # print progress
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b} after {h}:{m}:{s}")

            i = 0
            for k, j in enumerate(row_limits):  # k'th video in the batch "b"
                all_mesh_feats = fake_mesh_feats[i:j]
                global_index = global_indices[k]  # index of the video in the whole dataset
                echo_filename = echo_filenames[global_index]
                if echo_filename in skipped_echos:  # skip echo video if no tracing data available
                    i = j
                    continue

                # get the echo EDF and ESF tracing data
                echo_tracing_data = echo_tracings_data[echo_filename]
                echo_edf_tracing_data, echo_esf_tracing_data = echo_tracing_data

                echo_edf_tracing_frame, echo_edf_lax, edf = echo_edf_tracing_data
                echo_esf_tracing_frame, echo_esf_lax, esf = echo_esf_tracing_data

                echo_frames = [echo_edf_tracing_frame, echo_esf_tracing_frame]
                echo_laxes = [echo_edf_lax, echo_esf_lax]

                mesh_feats = np.array([all_mesh_feats[edf], all_mesh_feats[esf]])

                output_dir = self.log_dir / "echo_mesh_lv_tracings"

                # mesh_echo_iou_and_dice(echo_filename, echo_frames, echo_laxes, mesh_feats, mesh_data_handler, output_size, False, False, None, output_dir)
                # exit(0)
                args = (echo_filename, echo_frames, echo_laxes, mesh_feats, mesh_data_handler, output_size, False, True, return_dict, output_dir)
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

        edf_IoUs, esf_IoUs = [], []
        edf_Dices, esf_Dices = [], []
        filenames = []
        for echo_filename in echo_filenames:
            if echo_filename in skipped_echos:
                continue

            IoUs, Dices = return_dict[echo_filename]
            edf_IoUs.append(IoUs[0])
            esf_IoUs.append(IoUs[1])
            edf_Dices.append(Dices[0])
            esf_Dices.append(Dices[1])
            filenames.append(echo_filename)

        logger.info("Saving computed IoU and Dice values...")

        # Save IoU values
        IoU_csv_file = self.log_dir / "IoUs.csv"
        IoU_df = pd.DataFrame({"FileName": filenames, "EDF_IoU": edf_IoUs, "ESF_IoU": esf_IoUs})
        IoU_df.to_csv(IoU_csv_file, index=False)

        # Save Dice values
        Dice_csv_file = self.log_dir / "Dices.csv"
        Dice_df = pd.DataFrame({"FileName": filenames, "EDF_Dice": edf_Dices, "ESF_Dice": esf_Dices})
        Dice_df.to_csv(Dice_csv_file, index=False)

        logger.info("Saved!")

    def dice_and_IoU_echo_mesh_volume_based(self, echo_dataset,
                                            echo_filenames,
                                            echo_edfs,
                                            set_name=None,
                                            framerate_ds_factor=1,
                                            mean_mesh_iou_dice=True,
                                            compute_ef=False,
                                            save=True,
                                            iou_save_filename="IoUs.csv",
                                            dice_save_filename="Dice.csv",
                                            ef_save_filename="EF.csv"):

        logger.info("\nComputing IoU and Dice values...")

        volume_tracings = utils.get_volume_tracings_df(self.echo_data_dir)

        # for each echo filename, the tracing_data is a pair of triple (one for EDF and one for ESF)
        # where triple = (tracing_frame, lax, frame_nb)
        output_size = (112, 112)
        echo_tracings_data, skipped_echos = utils.generate_echo_tracings(echo_filenames, echo_edfs, volume_tracings, output_size)

        # mesh dataset handler
        mesh_data_handler = self.mesh_ae.data_handler

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
        for b, (echo_times, echo_frames, echo_params) in enumerate(echo_dataset):  # b'th batch

            # translate echo vids to mesh vids aligned in time (transform relevant arrays to numpy)
            real_echo_latents = self.echo_ae.encode(echo_times, echo_frames, return_freq_phase=True, training=False)
            fake_mesh_latents, _ = self.translate_echo_to_mesh(real_echo_latents)

            # decrease framerate of decoded fake meshes for faster processing
            echo_times_tensor = echo_times.to_tensor(default_value=-1.0)  # comvert to tensor with default value -1
            max_len = echo_times_tensor.shape[1]
            indices = list(range(0, max_len, framerate_ds_factor))  # indices to take, take 1 frame for every "framerate_ds_factor" frames
            fake_mesh_times = tf.gather(echo_times_tensor, indices=indices, axis=1)  # downsampled echo times
            fake_mesh_times = tf.RaggedTensor.from_tensor(fake_mesh_times, padding=-1)  # convert back to RaggedTensor

            fake_mesh_feats = self.mesh_ae.decode(fake_mesh_latents, fake_mesh_times.values, fake_mesh_times.row_lengths(), training=False).numpy()

            echo_params = echo_params.to_tensor(default_value=0.0)
            echo_efs = echo_params[:, echo_ef_index].numpy()

            # get row lengths and row limits
            row_lengths = fake_mesh_times.row_lengths().numpy()
            row_limits = np.cumsum(row_lengths)
            if batch_size == -1:  # get batch_size
                batch_size = row_lengths.shape[0]
                logger.info(f"Batch size: {batch_size}")

            i = 0
            for k, j in enumerate(row_limits):  # k'th video in the batch "b"
                mesh_feats = fake_mesh_feats[i:j]
                global_index = b * batch_size + k  # index of the video in the whole dataset
                echo_filename = echo_filenames[global_index]
                if echo_filename in skipped_echos:  # skip echo video if no tracing data available
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
                args = (
                echo_filename, echo_frames, echo_laxes, mesh_feats, mesh_data_handler, mean_mesh_iou_dice, output_size, True, compute_ef, True, return_dict,
                output_dir)
                p = mp.Process(target=mesh_echo_iou_and_dice, args=args)
                parallel_jobs.append(p)
                p.start()

                i = j

            # wait for parallel jobs to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, max_parallel_jobs)

            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done 1 batch of {batch_size} videos in {h}:{m}:{s}")
            t1 = time.time()

            if b % 10 == 0:  # print progress
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b} after {h}:{m}:{s}")

        # wait for all parallel jobs to complete
        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
        logger.info(f"Done all batches in {h}:{m}:{s}")

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

            if echo_filename not in list(return_dict.keys()):
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

        IoU_df = pd.DataFrame(
            {"FileName": filenames, "EDF_IoU": edf_IoUs, "ESF_IoU": esf_IoUs, "Mean_Mesh_EDF_IoU": edf_IoUs_MM, "Mean_Mesh_ESF_IoU": esf_IoUs_MM})
        Dice_df = pd.DataFrame(
            {"FileName": filenames, "EDF_Dice": edf_Dices, "ESF_Dice": esf_Dices, "Mean_Mesh_EDF_Dice": edf_Dices_MM, "Mean_Mesh_ESF_Dice": esf_Dices_MM})
        EF_df = pd.DataFrame(
            {"FileName": filenames, "EF_Vol": efs_vol, "EF_Biplane": efs_biplane, "EF_Echo": efs_echo, "Mesh_EDV_3D": edvs_3d, "Mesh_ESV_3D": esvs_3d,
             "Mesh_EDV_Biplane": edvs_biplane, "Mesh_ESV_Biplane": esvs_biplane})

        if save:
            logger.info("Saving computed IoU and Dice values...")

            # Save IoU values
            iou_output_dir = self.log_dir / "iou_data" / set_name
            iou_output_dir.mkdir(parents=True, exist_ok=True)
            IoU_csv_file = iou_output_dir / iou_save_filename
            IoU_df.to_csv(IoU_csv_file, index=False)

            # Save Dice values
            dice_output_dir = self.log_dir / "dice_data" / set_name
            dice_output_dir.mkdir(parents=True, exist_ok=True)
            Dice_csv_file = dice_output_dir / dice_save_filename
            Dice_df.to_csv(Dice_csv_file, index=False)

            if compute_ef:
                logger.info("Saving computed EF values...")
                ef_output_dir = self.log_dir / "ef_data" / set_name
                ef_output_dir.mkdir(parents=True, exist_ok=True)
                EF_csv_file = ef_output_dir / ef_save_filename
                EF_df.to_csv(EF_csv_file, index=False)

            logger.info("Saved!")

        return IoU_df, Dice_df, EF_df

    # -------------------------------------------------- Mesh vs Echo EF ------------------------------------------------
    def echo_vs_mesh_ef(self, echo_dataset, echo_filenames, set_name, save=True, output_dir=None, save_filename="EchovsMeshEF_Vol.csv", gen_plots=False,
                        global_step=0, write_summaries=False):
        logger.info("\nEFs Echo to Mesh...")
        # mesh dataset params
        mesh_data_handler = self.mesh_ae.data_handler

        # parallel jobs
        manager = mp.Manager()
        return_dict = manager.dict()
        max_parallel_jobs = mp.cpu_count()
        parallel_jobs = []

        t_start = time.time()
        t1 = time.time()
        batch_size = -1
        echo_ef_index = ECHO_DATA_PARAMS.index("EF")
        all_echo_efs = []
        all_mesh_efs_pred = []
        count = 0
        for b, (echo_times, echo_frames, echo_params) in enumerate(echo_dataset):  # b'th batch

            # decrease nb of timesteps to reduce time needed to process the mesh video
            echo_times_tensor = echo_times.to_tensor(default_value=-1.0)  # comvert to tensor with default value -1
            max_len = echo_times_tensor.shape[1]
            indices = list(range(0, max_len, 5))  # indices to take, take every 5th frame
            fake_mesh_times = tf.gather(echo_times_tensor, indices=indices, axis=1)  # downsampled echo times
            fake_mesh_times = tf.RaggedTensor.from_tensor(fake_mesh_times, padding=-1)  # convert back to RaggedTensor

            # translate echo vids to mesh vids aligned in time (transform relevant arrays to numpy)
            real_echo_latents = self.echo_ae.encode(echo_times, echo_frames, return_freq_phase=True, training=False)
            fake_mesh_latents, _ = self.translate_echo_to_mesh(real_echo_latents)
            fake_mesh_feats = self.mesh_ae.decode(fake_mesh_latents, fake_mesh_times.values, fake_mesh_times.row_lengths(), training=False).numpy()
            mesh_efs_pred = self.mesh_ef_pred(fake_mesh_latents).numpy() * 100

            echo_params = echo_params.to_tensor(default_value=0.0)
            echo_efs = echo_params[:, echo_ef_index].numpy()

            # get row lengths and row limits
            row_lengths = fake_mesh_times.row_lengths()
            row_limits = tf.cumsum(row_lengths).numpy()
            if batch_size == -1:  # get batch_size
                batch_size = row_limits.shape[0]
                logger.info(f"Batch size: {batch_size}")

            i = 0
            for k, j in enumerate(row_limits):  # k'th video in the batch "b"
                mesh_feats = fake_mesh_feats[i:j]
                echo_ef = echo_efs[k]
                mesh_ef_pred = mesh_efs_pred[k][0]

                all_echo_efs.append(echo_ef)
                all_mesh_efs_pred.append(mesh_ef_pred)

                ef_disks_params = {"view_pair": ("4CH", "2CH"), "slicing_ref_frame": "EDF"}
                # ef_disks_params = None

                # ef_vol, ef_biplane = compute_ef_mesh_vid(mesh_feats, mesh_data_handler, ef_disks_params)
                # logger.info(f"EF vol: {ef_vol}")
                # logger.info(f"EF biplane: {ef_biplane}")
                # exit(0)
                args = (mesh_feats, mesh_data_handler, ef_disks_params, True, return_dict, count)
                p = mp.Process(target=compute_ef_mesh_vid, args=args)
                parallel_jobs.append(p)
                p.start()

                i = j
                count += 1

            # wait for parallel jobs to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, max_parallel_jobs)

            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done 1 batch of {batch_size} videos in {h}:{m}:{s}")
            t1 = time.time()

            if b % 10 == 0:  # print progress
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b} after {h}:{m}:{s}")

        # wait for all parallel jobs to complete
        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
        logger.info(f"Done all batches in {h}:{m}:{s}")

        all_mesh_efs_vol = []
        all_mesh_efs_biplane = []

        to_delete_idx = []
        for i in range(count):
            if i not in list(return_dict.keys()):
                to_delete_idx.append(i)
                continue

            ef_vol, ef_biplane = return_dict[i]
            all_mesh_efs_vol.append(ef_vol)
            all_mesh_efs_biplane.append(ef_biplane)

        all_echo_efs = np.asarray([aeefs for i, aeefs in enumerate(all_echo_efs) if i not in to_delete_idx])
        all_mesh_efs_pred = np.asarray([amefs for i, amefs in enumerate(all_mesh_efs_pred) if i not in to_delete_idx])
        echo_filenames = np.asarray([efn for i, efn in enumerate(echo_filenames) if i not in to_delete_idx])

        all_mesh_efs_vol = np.array(all_mesh_efs_vol)
        all_mesh_efs_biplane = np.array(all_mesh_efs_biplane)
        all_echo_efs = np.array(all_echo_efs)
        all_mesh_efs_pred = np.array(all_mesh_efs_pred)

        EFs_df = pd.DataFrame(
            {"FileName": echo_filenames, "EF_Echo": all_echo_efs, "EF_Vol": all_mesh_efs_vol, "EF_Biplane": all_mesh_efs_biplane, "EF_Pred": all_mesh_efs_pred})

        if write_summaries:
            all_mesh_efs_biplane = all_mesh_efs_biplane / 100.0
            all_echo_efs = all_echo_efs / 100.0

            ef_loss_echo_to_mesh = np.mean(np.abs(all_echo_efs - all_mesh_efs_biplane))
            ef_loss = ef_loss_echo_to_mesh

            losses = {'ef_loss_echo_to_mesh': ef_loss_echo_to_mesh,
                      'ef_loss': ef_loss
                      }

            self._log_metrics(losses, self._val_exps_metrics)
            self.write_summaries(self._val_exps_summary_writer, self._val_exps_metrics, global_step, "Validation_exps")
        # Save EF values
        if save:
            if output_dir is None:
                output_dir = self.log_dir / "ef_data" / set_name
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Saving computed EF values...")
            csv_file = output_dir / save_filename
            EFs_df.to_csv(csv_file, index=False)

        if gen_plots:
            if output_dir is None:
                output_dir = self.log_dir

            plot_y_equals_x = True
            # plot EF_Echo vs EF_Vol scatter plot
            ef_plots_dir = output_dir / "ef_plots" / set_name
            plot_dir = ef_plots_dir / "EF_Vol_scatter"
            filename = f"EF_Vol_GS_{global_step}.png"
            x_data, y_data = np.array(EFs_df["EF_Echo"]), np.array(EFs_df["EF_Vol"])
            x_label, y_label = "Echo EF", "Mesh EF Volume"
            title = f"Echo vs Mesh EF (volume) scatter plot - GS {global_step}"
            utils.scatter_plot(plot_dir, filename, x_data, y_data, x_label, y_label, title, plot_y_equals_x=plot_y_equals_x)

            # plot EF_Echo vs EF_Pred scatter plot
            plot_dir = ef_plots_dir / "EF_Biplane_scatter"
            filename = f"EF_Biplane_GS_{global_step}.png"
            x_data, y_data = np.array(EFs_df["EF_Echo"]), np.array(EFs_df["EF_Biplane"])
            x_label, y_label = "Echo EF", "Mesh EF Biplane"
            title = f"Echo vs Mesh EF (biplane) scatter plot - GS {global_step}"
            utils.scatter_plot(plot_dir, filename, x_data, y_data, x_label, y_label, title, plot_y_equals_x=plot_y_equals_x)

            # plot EF_Echo vs EF_Pred scatter plot
            plot_dir = ef_plots_dir / "EF_Pred_scatter"
            filename = f"EF_Pred_GS_{global_step}.png"
            x_data, y_data = np.array(EFs_df["EF_Echo"]), np.array(EFs_df["EF_Pred"])
            x_label, y_label = "Echo EF", "Mesh EF Pred"
            title = f"Echo vs Mesh EF (ef pred) scatter plot - GS {global_step}"
            utils.scatter_plot(plot_dir, filename, x_data, y_data, x_label, y_label, title, plot_y_equals_x=plot_y_equals_x)

            # plot EF_Vol vs EF_Pred scatter plot
            plot_dir = ef_plots_dir / "EF_Mesh_scatter"
            filename = f"EF_Mesh_GS_{global_step}.png"
            x_data, y_data = np.array(EFs_df["EF_Biplane"]), np.array(EFs_df["EF_Pred"])
            x_label, y_label = "Mesh EF Biplane", "Mesh EF Pred"
            title = f"Mesh EF Biplane vs Mesh EF Pred scatter plot - GS {global_step}"
            utils.scatter_plot(plot_dir, filename, x_data, y_data, x_label, y_label, title, plot_y_equals_x=plot_y_equals_x)

        return EFs_df

    def echo_vs_mesh_ef_disks_method(self, echo_dataset, echo_filenames, edf_pairs, echo_edfs, echo_esfs):

        # get "echo_edf - mesh_edf" in same order as filenames
        diff_edfs = []
        for filename in echo_filenames:
            row = edf_pairs[edf_pairs["FileName"] == filename]
            diff_edfs.append(int(row["Echo_EDF"]) - int(row["Mesh_EDF"]))

        # mesh dataset params
        mesh_dataset_scale = self.mesh_ae.data_handler.dataset_scale
        mesh_dataset_min = self.mesh_ae.data_handler.dataset_min
        mesh_ref_poly = self.mesh_ae.data_handler.reference_poly

        # parallel jobs to compute video efs, 1 process per video ---> faster processing
        manager = mp.Manager()
        return_dict = manager.dict()
        max_parallel_jobs = mp.cpu_count()
        parallel_jobs = []

        t_start = time.time()
        all_echo_efs = {}
        output_dir = self.log_dir / "visuals"

        for b, (global_indices, row_limits, fake_mesh_feats, times_values, echo_frames_values, echo_efs) in enumerate(
                self.echo_vid_to_mesh_vid_edf_aligned_temporally(echo_dataset, diff_edfs)):
            t1 = time.time()
            batch_size = len(global_indices)

            if b % 10 == 0:  # print progress
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b} after {h}:{m}:{s}")

            i = 0
            for k, j in enumerate(row_limits):  # k'th video in the batch "b"
                mesh_feats = fake_mesh_feats[i:j]
                global_index = global_indices[k]  # index of the video in the whole dataset
                echo_ef = echo_efs[k]
                echo_filename = echo_filenames[global_index]
                edf = echo_edfs[global_index]
                esf = echo_esfs[global_index]

                all_echo_efs[echo_filename] = echo_ef
                view_pair = ("4CH", "2CH")

                # mesh_ef_disks_method(echo_filename, view_pair, mesh_feats, edf, esf, mesh_ref_poly, mesh_dataset_scale, mesh_dataset_min, False, None)
                # exit(0)
                args = (echo_filename, view_pair, mesh_feats, edf, esf, mesh_ref_poly, mesh_dataset_scale, mesh_dataset_min, True, return_dict)
                p = mp.Process(target=mesh_ef_disks_method, args=args)
                parallel_jobs.append(p)
                p.start()

                i = j

            # wait for all parallel jobs of batch to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done 1 batch of {batch_size} videos in {h}:{m}:{s}")

        # wait for all parallel jobs of batch to complete
        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

        t_end = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t_end - t_start)
        logger.info(f"Done computing all efs in {h}:{m}:{s}")

        echo_efs = []
        mesh_efs = []
        filenames = []
        for echo_filename in return_dict:
            echo_efs.append(all_echo_efs[echo_filename])
            mesh_efs.append(return_dict[echo_filename])
            filenames.append(echo_filename)

        for echo_filename in echo_filenames:
            if echo_filename not in filenames:
                logger.info(f"Not found: {echo_filename}")

        # correlation plot between echo and mesh ejection fractions computed with method of disks
        plots_dir = self.log_dir / 'plots'
        ef_plots_dir = plots_dir / "ejection_fraction_disks"
        ef_plots_dir.mkdir(parents=True, exist_ok=True)
        plt.clf()
        filename = "EjectionFractionPlotDisks.png"
        plt.plot(echo_efs, mesh_efs, 'bo')
        plt.title(f'Ejection Fraction correlation plot between echos and corresponding meshes - disks method')
        min_echo_ef = min(echo_efs)
        max_echo_ef = max(echo_efs)
        min_mesh_ef = min(mesh_efs)
        max_mesh_ef = max(mesh_efs)
        plt.xlabel(f'Echo Ejection Fraction\nmin val: {min_echo_ef:.5f}, max val: {max_echo_ef:.5f}')
        plt.ylabel(f'Mesh Ejection Fraction\nmin val: {min_mesh_ef:.5f}, max val: {max_mesh_ef:.5f}')
        plt.savefig(ef_plots_dir / filename, bbox_inches='tight')

    # -------------------------------------------- Logging, Saving and Loading ------------------------------------------
    def log_weights(self):
        # log number of trainable weights
        trainable_variable = [v for v in self.echo_gen.trainable_variables]
        trainable_variable.extend([v for v in self.echo_ae.trainable_variables])
        trainable_variable.extend([v for v in self.mesh_gen.trainable_variables])
        trainable_variable.extend([v for v in self.echo_disc.trainable_variables])
        trainable_variable.extend([v for v in self.mesh_disc.trainable_variables])
        logger.info(f"\nTrainable variables names: {[v.name for v in trainable_variable]}")
        n_trainable = np.sum([np.prod(v.get_shape().as_list()) for v in trainable_variable])
        logger.info(f"Number of trainable variables: {n_trainable}\n")

    def _log_metrics(self, losses, metrics):
        for metric in losses:
            metrics[metric](losses[metric])

    @staticmethod
    def write_summaries(summary_writer, metrics, global_step, log_str, is_dict=False):
        # get the metrics values
        metrics_values = {}
        if is_dict:
            metrics_values = metrics
        else:
            for metric, value in metrics.items():
                metrics_values[metric] = value.result()

        with summary_writer.as_default():
            for metric, value in metrics_values.items():
                tf.summary.scalar(metric, value, step=global_step)

        # print metrics
        strings = ['%s: %.5e' % (k, v) for k, v in metrics_values.items()]
        logger.info(f"\nGlobal Step: {global_step} | {log_str}: {' - '.join(strings)} ")

    def reset_metrics(self, train=False, val=False, val_exp=False):
        metrics = []
        if train:
            metrics.extend(self._train_metrics.values())
        if val:
            metrics.extend(self._val_metrics.values())
        if val_exp:
            metrics.extend(self._val_exps_metrics.values())
        for metric in metrics:
            metric.reset_states()

    def save_me(self, name="best"):
        trained_model_path = self.log_dir / "trained_models" / "cycleGAN"
        self.save_weights(str(trained_model_path) + f"_{name}")

        logger.info(f"Model saved to file {trained_model_path}_{name}")

    # -------------------------------------------------- Not Used (Old) -------------------------------------------------
    def save_mesh_video_seg_maps_and_volumes_over_time(self, feats, times, output_dir, prefix, index, true_meshes):
        # dataset values
        ref_poly = self.mesh_ae.data_handler.reference_poly
        dataset_scale = self.mesh_ae.data_handler.dataset_scale
        dataset_min = self.mesh_ae.data_handler.dataset_min

        # get video duration
        vid_duration = times[-1]

        # save mesh as vtps and video
        output_dir, filenames = self.save_vtp_from_feats(feats, output_dir, prefix)  # convert mesh feats to vtps
        vtps_to_vid(output_dir, filenames, vid_duration)  # convert vtps to video
        # save mesh's seg map as video
        seg_map_output_dir = Path(str(output_dir).replace("/reconstruction", "/reconstruction_vid")).parent
        seg_map_prefix = output_dir.name
        views = {'2ch': CONRAD_VIEWS['2ch'],
                 '4ch': CONRAD_VIEWS['4ch']
                 }  # views to generate for
        feats_to_seg_map_vid(ref_poly, feats, dataset_scale, dataset_min, vid_duration, seg_map_output_dir, seg_map_prefix, views=views)

        # save mesh volumes per time for fake meshes
        if not true_meshes:
            # compute volumes for all chambers
            feats = (feats * dataset_scale) + dataset_min  # denormalize before computing volumes
            volumes = compute_volumes_feats(feats, ref_poly)  # compute volumes for all available components in mesh
            for c in volumes:  # for each component
                v_vs_t_dir = seg_map_output_dir / "volumes_over_time" / c  # output to same folder as seg map videos
                v_vs_t_dir.mkdir(parents=True, exist_ok=True)
                plt.clf()

                t = times
                volumes_c = volumes[c]  # volumes of this component

                plt.plot(t, volumes_c)
                plt.title(f'{c} volumes over time (fake mesh)')
                plt.xlabel('Times')
                plt.ylabel('Fake volume')

                # save plot
                mesh_seq_nb = str(index).zfill(3)
                filename = f"VolumeOverTimeMesh{mesh_seq_nb}_{c}.png"
                plt.savefig(v_vs_t_dir / filename, bbox_inches='tight')
