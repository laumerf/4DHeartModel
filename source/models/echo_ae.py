import os
from datetime import datetime
from itertools import chain
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import cv2 as cv

import source.utils as utils
from source.echo_utils import load_data_echonet, get_train_dataset, get_val_dataset, save_video
from source.models.ops import Conv, DeConv


# Model definition

# TODO: implement logger

class Encoder(Layer):

    def __init__(self, latent_space_dim, input_noise=None,
                 name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.input_noise = input_noise

        # Convolution
        self.conv1 = Conv(8, 4, 2, activation='relu',
                          batch_normalisation=False, name='conv1')
        self.conv2 = Conv(16, 4, 2, activation='relu',
                          batch_normalisation=False, name='conv2')
        self.conv3 = Conv(16, 4, 2, activation='relu',
                          batch_normalisation=False, name='conv3')
        self.conv4 = Conv(16, 4, 2, activation='relu',
                          batch_normalisation=False, name='conv4')

        # TODO (fabian): consider fully convolutional
        # TODO (fabian): other activation function
        self.dense = Dense(latent_space_dim, activation='sigmoid',
                           name='dense')

    def call(self, inputs, training=False):
        """
        inputs is expected to be of shape [batch_size, height, width, n_views]
        """

        # Add Gaussian noise
        if self.input_noise is not None:
            h = GaussianNoise(self.input_noise)(inputs, training=training)
        else:
            h = inputs

        h = self.conv1(h, training=training)  # (32, 55, 55, 8)
        h = self.conv2(h, training=training)  # (32, 26, 26, 16)
        h = self.conv3(h, training=training)  # (32, 12, 12, 16)
        h = self.conv4(h, training=training)  # (32, 5, 5, 16)

        h = tf.reshape(h, shape=[-1, int(np.prod(h.get_shape()[1:]))])

        latent = self.dense(h)

        output = latent

        return output


class Decoder(Layer):

    def __init__(self, latent_space_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.latent_space_dim = latent_space_dim

        self.dense = Dense(5 * 5 * 16, activation='relu', name='dense',
                           input_shape=(latent_space_dim,))

        self.deconv1 = DeConv(32, 4, 2, activation='relu',
                              batch_normalisation=False, name='deconv1')
        self.deconv2 = DeConv(16, 4, 2, activation='relu',
                              batch_normalisation=False, name='deconv2')
        self.deconv3 = DeConv(8, 4, 2, activation='relu',
                              batch_normalisation=False, output_padding=1,
                              name='deconv3')
        self.deconv4 = DeConv(1, 4, 2, activation='relu',
                              batch_normalisation=False, name='deconv4')

    def call(self, inputs, training=False):
        h = inputs
        h = self.dense(h)

        h = tf.reshape(h, shape=[-1, 5, 5, 16])
        h = self.deconv1(h, training=training)
        h = self.deconv2(h, training=training)  # (32, 12, 12, 16)
        h = self.deconv3(h, training=training)  # (32, 26, 26, 16)
        h = self.deconv4(h, training=training)  # (32, 55, 55, 16)

        output = h

        return output


class EchoAutoencoderModel(Model):

    def __init__(self, model_params, data_params, training_params,
                 input_noise=None, log_dir=None,
                 name='eae', **kwargs):

        super(EchoAutoencoderModel, self).__init__(name=name, **kwargs)

        self._validation_metrics = dict()
        self._train_metrics = dict()

        self.log_dir = log_dir
        self.latent_space_dim = model_params['latent_space_dim']
        self.batch_size = data_params['batch_size']
        self.shuffle_buffer = data_params['shuffle_buffer']
        self.n_echo_videos = data_params['n_videos']
        self.learning_rate = training_params['learning_rate']
        self.input_noise = input_noise
        self.data_dir = data_params['data_dir']
        self.trained_model_path = self.log_dir / "trained_models" / "EAE"
        self.steps_per_epoch = training_params["num_steps_per_epoch"]
        self.patience = training_params["patience"]
        self.max_epochs = training_params["num_epochs"]
        self.test_freq_n_epochs = training_params["test_frequency"]

        self.encoder = Encoder(latent_space_dim=self.latent_space_dim,
                               input_noise=input_noise)

        self.decoder = Decoder(latent_space_dim=self.latent_space_dim)

    def call(self, inputs, training=False):

        latent = self.encoder(inputs, training=training)

        # reconstructions
        reconstructions = self.decoder(latent, training=training)

        return latent, reconstructions

    def decode(self, inputs):

        x_rec = self.decoder(inputs, training=False)
        return x_rec

    @staticmethod
    def _reconstruction_error(y_true, y_pred):

        mse = tf.reduce_mean((y_true - y_pred) ** 2, axis=[-3, -2, -1])
        return mse

    def _loss(self, true_frames, pred_frames):

        # flatten sequences and calculate reconstruction error
        reconstruction_error = self._reconstruction_error(true_frames,
                                                          pred_frames)

        # total loss
        loss = reconstruction_error

        return loss, reconstruction_error

    @tf.function
    def _train_step(self, frames, optimizer):

        with tf.GradientTape() as tape:
            latent = self.encoder(frames, training=True)
            pred_frames = self.decoder(latent, training=True)

            loss, reconstruction_error = self._loss(frames, pred_frames)

        # update model weights
        variables = self.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        # logging
        self._train_metrics['loss'](loss)
        self._train_metrics['reconstruction_error'](reconstruction_error)

    @tf.function
    def _evaluate(self, frames):

        latent = self.encoder(frames, training=False)
        reconstructed_frames = self.decoder(latent, training=False)

        reconstruction_error = self._reconstruction_error(frames,
                                                          reconstructed_frames)

        # logging
        self._validation_metrics['reconstruction_error'](reconstruction_error)

    def fit(self, **kwargs):

        # Load data from AMI/TTS and EchoNet datasets
        data_info, files = load_data_echonet(path=self.data_dir,
                                             number_of_videos=self.n_echo_videos)

        ids = list(files.keys())
        files = np.array([files[id_]for id_ in ids])

        # Train Validation files
        train_files = files[0:int(0.8 * len(files))]
        val_files = files[int(0.8 * len(files)):]

        train_dataset = get_train_dataset(train_files,
                                          batch_size=self.batch_size,
                                          n_shuffle=self.shuffle_buffer)
        val_dataset = get_val_dataset(val_files)

        # setup summary writers
        train_log_dir = str(self.log_dir) + '/train'
        val_log_dir = str(self.log_dir) + '/validation'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        optimizer = Adam(lr=self.learning_rate)

        opt_weights = self.get_weights()
        opt_early_stopping_metric = np.inf

        count = 0
        epoch = 0

        # setup train metrics
        train_metrics = ['loss', 'reconstruction_error']
        for metric in train_metrics:
            self._train_metrics[metric] = tf.keras.metrics.Mean(metric)

        # setup validation metrics
        val_metrics = ['reconstruction_error']
        for metric in val_metrics:
            self._validation_metrics[metric] = tf.keras.metrics.Mean(metric)

        step = 0
        for frame in train_dataset:
            self._train_step(frame, optimizer)
            step += 1

            # epoch finished
            if step == self.steps_per_epoch:
                # reset step
                step = 0
                epoch += 1
                count += 1

                # update learning rate
                if count > self.patience - int(self.patience / 2):
                    lr = optimizer.learning_rate * 0.9
                    print(f"Reduced learning rate to {lr}")
                    optimizer.learning_rate = lr

                # log train metrics
                with train_summary_writer.as_default():
                    for metric in train_metrics:
                        tf.summary.scalar(metric,
                                          self._train_metrics[metric].result(),
                                          step=epoch)

                # print train metrics
                train_strings = [
                    '%s: %.3e' % (k, self._train_metrics[k].result())
                    for k in train_metrics]
                print('{}: Epoch {}: Train: '.format(datetime.now(),
                                                     epoch) + ' - '.join(
                    train_strings), flush=True)

                # validate
                for frames in val_dataset:
                    self._evaluate(frames)

                # log validation metrics
                with val_summary_writer.as_default():
                    for metric in val_metrics:
                        tf.summary.scalar(metric, self._validation_metrics[
                            metric].result(), step=epoch)

                # reset early stopping counter on improvement
                if self._validation_metrics['reconstruction_error'].result() < opt_early_stopping_metric:
                    opt_early_stopping_metric = self._validation_metrics['reconstruction_error'].result()
                    opt_weights = self.get_weights()
                    count = 0
                    name = 'best'  # could also be epoch...
                    print(f"Saving best model to {self.trained_model_path}")
                    self.save_weights(str(self.trained_model_path) + f"_{name}")
                else:
                    print(f"Validation loss did not improve from "
                          f"{opt_early_stopping_metric}. "
                          f"Counter: {count}/{self.patience}")

                # print validation metrics
                val_strings = [
                    '%s: %.3e' % (k, self._validation_metrics[k].result()) for
                    k in
                    val_metrics]
                print('{}: Epoch {}: Validation: '.format(datetime.now(),
                                                          epoch) + ' - '.join(
                    val_strings), flush=True)

                # log input and reconstruction to tensorboard
                if epoch % self.test_freq_n_epochs == 0 or epoch == 1:
                    for frames in val_dataset:
                        outputs = self.decoder(self.encoder(frames,
                                                            training=False),
                                               training=False)

                        frame = frames[0, ..., 0]
                        output = outputs[0, ..., 0]

                        plot = utils.plot_input_output(frame.numpy(),
                                                       output.numpy())
                        image = utils.plot_to_image(plot)
                        with val_summary_writer.as_default():
                            tf.summary.image("Input-Output", data=image,
                                             step=epoch)
                        break

                # reset metrics
                for metric in chain(self._train_metrics.values(),
                                    self._validation_metrics.values()):
                    metric.reset_states()

            # stop training
            if count > self.patience or epoch > self.max_epochs:
                print(f"Training stopped at epoch {epoch}!")
                break  # leave training loop

        # reset to optimal weights
        print("Load optimal weights")
        self.set_weights(opt_weights)

    def extract_features(self, videos=None):

        eval_dir = self.log_dir / 'features'
        os.makedirs(eval_dir, exist_ok=True)

        # Load data from AMI/TTS and EchoNet datasets
        data_info, files = load_data_echonet(
            path=self.echo_autoencoder.data_dir,
            number_of_videos=1, videos=videos)

        ids = list(files.keys())
        files = np.array([files[id_] for id_ in ids])

        series = []
        for video_file in files:
            data = np.load(video_file)
            video_name = video_file.split('/')[-1].split('.')[0]
            frames = data['frames']

            frames = np.array(
                [cv.resize(x, (112, 112)) for x in frames]).astype(
                'float32')

            # normalize per video (maybe to load file function)
            frames -= np.mean(frames)
            frames /= np.std(frames)

            frames = frames[..., np.newaxis]

            time_series = self.encoder(frames, training=False)
            series.append({'file': video_name, 'video': time_series})

        filepath = os.path.join(eval_dir, "series.npz")
        np.savez(filepath, series=series)

        return series

    def reconstruct(self, videos):

        eval_dir = self.log_dir / 'reconstruction'
        os.makedirs(eval_dir, exist_ok=True)

        # Load data from AMI/TTS and EchoNet datasets
        data_info, files = load_data_echonet(
            path=self.data_dir,
            number_of_videos=1, videos=videos)

        ids = list(files.keys())
        files = np.array([files[id_] for id_ in ids])

        videos = []
        for file in files:
            data = np.load(file)

            frames = data['frames']
            video_name = file.split('/')[-1].split('.')[0]
            frames = np.array(
                [cv.resize(x, (112, 112)) for x in frames]).astype(
                'float32')

            # normalize per video
            frames -= np.mean(frames)
            frames /= np.std(frames)

            frames = frames[..., np.newaxis]

            time_series = self.encoder(frames, training=False)
            reconstructed_video = self.decoder(time_series, training=False)
            reconstructed_video = np.array(reconstructed_video[..., 0].numpy())

            videos.append({'file': file, 'video': reconstructed_video})
            # save generated echo
            file_path = os.path.join(eval_dir, f"{video_name}.avi")
            save_video(frames=reconstructed_video, file_path=file_path)

        filepath = os.path.join(eval_dir, "reconstructed_videos.npz")
        np.savez(filepath, videos=videos)

        return videos
