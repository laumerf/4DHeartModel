from datetime import datetime
from itertools import chain
import numpy as np
import time
import multiprocessing as mp
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import *

from source.utils_dhb import *
import source.utils as utils

from source.shape_model_utils import frames_to_vid

ECHO_DATA_PARAMS = ["EF","ESV","EDV","FrameHeight","FrameWidth","FPS","NumberOfFrames"]

# -------------------------------------------------------- Utility function------------------------------------------------------       
def random_subsequence_start_end(N, min_length):
    """
    Random start and end indices for a random subsequence for a sequences of length N.
    """
    n_subsequences = ((N-min_length)**2 + N - min_length)/2
    start = np.random.choice(np.arange(N - min_length), p=np.arange(N - min_length, 0, step=-1)/n_subsequences)
    end = np.random.choice(np.arange(start+min_length, N))
    return start, end
# -------------------------------------------------------------------------------------------------------------------------------

# Model definition

# ----------------------------------------------- Custom Conv+BatchNorm Layer ----------------------------------------------------
class Conv(Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, batch_normalisation=False, name=None):
        
        # call super's init (super = Layer class)
        super(Conv, self).__init__()

        # save params and layers in self
        self.activation = activation
        self.batch_normalisation = batch_normalisation

        self.conv = Conv2D(filters, kernel_size, activation=None, strides=strides, padding=padding, name=name, use_bias=(not batch_normalisation))
        
        if self.batch_normalisation:
            self.bn = BatchNormalization()

    def call(self, inputs, training=False):

        # apply conv then batch norm then apply the activation
        h = inputs
        h = self.conv(h)
        if self.batch_normalisation:
            h = self.bn(h, training=training)
        h = Activation(self.activation)(h)
        return h
# -------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------- Custom Deconv+BatchNorm Layer -------------------------------------------------
class DeConv(Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, batch_normalisation=False, output_padding=None, name=None):
        
        # call super's init (super = Layer class)
        super(DeConv, self).__init__()

        # save params and layers in self
        self.activation = activation
        self.batch_normalisation = batch_normalisation

        self.conv = Conv2DTranspose(filters, kernel_size, strides=strides, activation=None, padding=padding, output_padding=output_padding, name=name, use_bias=(not batch_normalisation))
        
        if self.batch_normalisation:
            self.bn = BatchNormalization()

    def call(self, inputs, training=False):

        # apply deconv then batch norm then apply the activation
        h = inputs
        h = self.conv(h)
        if self.batch_normalisation:
            h = self.bn(h, training=training)
        h = Activation(self.activation)(h)
        return h
# -------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------- Encoder -----------------------------------------------------------
class Encoder(Layer):
    
    def __init__(self, latent_space_dim, hidden_dim, input_noise=None, name='encoder', **kwargs):
        
        super(Encoder, self).__init__(name=name, **kwargs)

        self._supports_ragged_inputs = True
                
        self.latent_space_dim = latent_space_dim
        self.input_noise = input_noise
        
        # Custom Convolution Layers (but no batch norm used): relu activations
        self.conv1 = Conv(8, 4, 2, activation='relu', batch_normalisation=False, name='conv1')
        self.conv2 = Conv(16, 4, 2, activation='relu', batch_normalisation=False, name='conv2')
        self.conv3 = Conv(16, 4, 2, activation='relu', batch_normalisation=False, name='conv3')
        self.conv4 = Conv(16, 4, 2, activation='relu', batch_normalisation=False, name='conv4')
        
        # Elman Net (LSTM and Dense layers): Bidirectional LSTM Layer with tanh activation for the output and sigmoid activation for the recurrent state
        # Dense layer with linear activation

        # LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
        # Recall: LSTM computes 2 vectors: the output vector of the current timestep (given the input vector and the cell state of the previous timestep)
        # and the cell state of the current timestep (passed to the next timestep)
        # here the cell state vector is also called hidden state and has size "hidden_dim"
        # activation and recurrent_activation used here are the defaults
        
        # we use bidirectional LSTM to process each video from first frame to last frame (forwards processing) 
        # then from last frame to first frame (backwards processing)
        # so we get a sequence of "nb_frames" output vectors in the forwards processing step (one per frame)
        # and we get a sequence of "nb_frames" output vectors in the  backwards processing step (one per frame)
        # so in the forward processing: the output vector of timestep 0, aka outF_0, is the first output
        # while in the backwards processing: the output vector of timestep 0, aka outB_0, is the last output
        # the final output of timestep 0, aka out_0, is the concat of outF_0 and outB_0

        self.lstm = Bidirectional(LSTM(hidden_dim, activation='tanh', recurrent_activation='sigmoid'))
        self.dense = Dense(latent_space_dim, activation='linear', name='dense')

    
    # function to encode input i,e transform to latent space then feed to LSTM to get params
    def call(self, inputs, training=False):

        """
        inputs is expected to be a tuple of 2 RaggedTensors
        """

        # RaggedTensor: https://www.tensorflow.org/guide/ragged_tensor
        # Ragged tensors are the TensorFlow equivalent of nested variable-length lists
        # here the videos have variable lengths since not all videos have the same number of frames
        
        # Ragged Tensor: https://www.tensorflow.org/api_docs/python/tf/RaggedTensor
        # A RaggedTensor is a tensor with one or more ragged dimensions, which are dimensions whose 
        # slices may have different lengths (to get a slice along a dimension "d", we take the i'th element of that dimension "d" e,g
        # tensor "data", then a slice along the 4th dim would be data[:, :, :, i, :, ..., : ], so we take all elems of other dimensions (i,e ":")
        # and only the i'th elem of the dimension "d" i,e choose a dimension i , take all elems along that dimension)
        # For example, the 2D tensor rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []] is ragged
        # since the slices (rt[0, :], ..., rt[4, :]) have different lengths.
        # Dimensions whose slices all have the same length are called uniform dimensions

        # get the batch of times and the batch of frames
        # each of "times" and "sequences" has batch_size elems with each element having different shape
        # i,e for "times", each elem has shape "nb_frames" represented here by (None, )
        # and for "sequences", each elem has shape "nb_frames x 112 x 112" represented here by (None, 112, 112)
        # so times.shape is (batch_size, None)
        # and sequences.shape is (batch_size, None, 112, 112)
        times, sequences = inputs

        # RaggedTensor: https://www.tensorflow.org/api_docs/python/tf/RaggedTensor
        # Internally, each RaggedTensor consists of 2 parts:
        # 1. A "values" tensor, which concatenates the "variable-length rows" into a flattened list, i,e it concatenates
        # the 1st dim into a list of values.
        # For example, the values tensor for [[3, 1, 4, 1], [], [5, 9, 2], [6], []] is [3, 1, 4, 1, 5, 9, 2, 6].
        # 2. A "row_splits" vector, which indicates how those flattened values are divided into rows. 
        # In particular, the values for row rt[i] are stored in the slice rt.values[rt.row_splits[i]:rt.row_splits[i+1]].

        # in this case, the rows is the "batch dim" where the i'th elem is a video, so "sequences.values" is a sequence of frames (concatenates frames
        # of all videos) where the start and end of each the i'th video is stored in sequences.row_splits[i] and sequences.row_splits[i+1] respectively
        # so the first 2 dimensions of shape "batch_size" and "None" are merged into 1 dimension of shape "None" (since nb of frames not fixed and unknown
        # initially i,e None, so the resulting dimension is also None)
        # thus the tensor has h.shape is (None, 112, 112, 1)
        h = tf.expand_dims(sequences.values, -1)
        # print(h.shape)

        # Add Gaussian noise to the input (by default None)
        if self.input_noise is not None:
            h = GaussianNoise(self.input_noise)(h, training=training)
        
        # apply custom conv layers in succession
        # Note here that the convolution happens only in the last 3 dimensions (height, width, channels)
        # so frames of the same video and even of different videos are not combined
        # each frame gets convolved with the conv filter gets outputted as a result,
        # so given the input [frame1_vid1, frame2_vid1, ..., frameF_vidB] (B here is batch_size), the result is
        # [conv(frame1_vid1), conv(frame2_vid1), ..., conv(frameF_vidB)]
        # so the dimension of size "None" stays the same and doesn't get changed, e,g if we had 3 videos of 20, 30 and 40 frames
        # so the None is 20+30+40 = 90, then the output's "None" dimension is also 90
        # so the frames and the results stay separated, so it's clear which output features are from which frame of which video
        h = self.conv1(h, training=training)
        h = self.conv2(h, training=training)
        h = self.conv3(h, training=training)
        h = self.conv4(h, training=training)
        # print(h.shape)
        # flattening: final conv layer's output has shape (None, 5, 5, 16), so we flatten it to have shape (None, 5*5*16) = (None, 400)
        # so for the 1st dim of "h" has the concatenated frames of all videos and the 2nd dim has the latent vector of size 400 for each frame
        h = tf.reshape(h, shape=[-1, 5*5*16])

        # times.shape is (batch_size, None) so times.values.shape is (None,)
        # we add a dimension to the end of times.values so we get shape (None, 1)
        # then we concat h of shape (None, 400) with of result of shape (None, 1) along the axis=1
        # i,e for each frame's latent vector, we add to the end of the vector the timestep of that frame,
        # so the resulting vector of size 401 is [frame's latent vector, timestep]
        # thus the resulting "h" has shape (None, 401)
        h = tf.concat([h, tf.expand_dims(times.values, axis=-1)], axis=1)
        # print(h.shape)

        # we split the sequence of (latent) frames into videos again using "limits = times.row_splits"
        # since "limits[i]" and "limits[i+1]" gives us the limits of the i'th video.
        # so get back a RaggedTensor of shape (batch_size, None, 401)
        h = tf.RaggedTensor.from_row_splits(h, row_splits=times.row_splits)
                
        # RNN
        # transform the ragged tensor into a tensor, i,e the dimension of shape "None" will 
        # have shape M="max_frames" where M is the number of frames of the video with max frames
        # so all videos will have M frames, smaller videos are padded in the end with black frames 
        # (since default value is 0.0)
        h_padded = h.to_tensor(default_value=0.0)
        # h_mask is used to tell the LSTM to ignore some values (frames) for each input video
        # so given a sequence of frames (video), the LSTM only processes unmasked frames (i,e where the mask value is True)
        # the masked frames are the added black frames when padding
        # here "h.row_lengths()" gives the number of frames per video as a list
        # so h_mash has shape (batch_size, M)
        # e,g tf.sequence_mask([1, 3, 2], 5) # [[True, False, False, False, False],
                                             #  [True, True, True, False, False],
                                             #  [True, True, False, False, False]]
        h_mask = tf.sequence_mask(h.row_lengths(), maxlen=tf.shape(h_padded)[1])

        # apply LSTM: we only get the final output of the LSTM (not the whole sequence of outputs, one per frame)
        # the LSTM hidden_dim is 128, but since we're using a Bi-LSTM, we get a final output of the forward pass (i,e after processing
        # 1st frame to last frame), this output has size 128
        # and we get the final output of the backwards pass (i,e after processing last frame to 1st frame), this output has size 128 too
        # so the for each video, the LSTM outputs a vector of size 256 which is the concat of the 2 outputs (forwards and backwards passes)
        # so the resulting "h.shape" is (batch_size, 256)
        h = self.lstm(h_padded, mask=h_mask, training=training)
        
        # apply Dense layer (with linear activation) to get the DeepHeartBeat params
        # output shape "h.shape" is (batch_size, 128)
        h = self.dense(h)

        # apply shape denses
        freqs = tf.exp(h[:, 0, None])
        shifts = h[:, 1, None]
        shape_params = h[:, 2:]

        # apply exp to pace
        h = tf.concat([freqs, shifts, shape_params], axis=1)

        # get output params of DeepHeartBeat and return them
        output = h

        
        return output
# -------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------- Decoder -----------------------------------------------------------
class Decoder(Layer):
    
    def __init__(self, latent_space_dim, name='decoder', **kwargs):
        
        super(Decoder, self).__init__(name=name, **kwargs)

        self.latent_space_dim = latent_space_dim
        
        # Keras Dense layer with relu activation
        self.dense = Dense(5*5*16, activation='relu', name='dense', input_shape=(latent_space_dim,))

        # Custom Deconv Layers (but no batch norm used): relu activated except last which is sigmoid activated
        self.deconv1 = DeConv(16, 4, 2, activation='relu', batch_normalisation=False, name='deconv1')
        self.deconv2 = DeConv(16, 4, 2, activation='relu', batch_normalisation=False, name='deconv2')
        self.deconv3 = DeConv(8, 4, 2, activation='relu', batch_normalisation=False, output_padding=1, name='deconv3')
        self.deconv4 = DeConv(1, 4, 2, activation='sigmoid', batch_normalisation=False, name='deconv4')
        
    def call(self, inputs, training=False):
        
        h = inputs
        # apply dense layer to get tensor of shape (None, 400)
        h = self.dense(h)

        # reshape to (None, 5, 16, 16)
        h = tf.reshape(h, shape=[-1, 5, 5, 16])
        # apply deconv layers
        h = self.deconv1(h, training=training)
        h = self.deconv2(h, training=training)
        h = self.deconv3(h, training=training)
        h = self.deconv4(h, training=training)

        # h has shape (None, 112, 112, 1), we squeeze it remove the dimensions of size 1 (i,e the last one)
        # so result has shape (None, 112, 112)
        h = tf.squeeze(h, axis=-1)
        
        # return result
        output = h
        
        return output
# -------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------- Trajectory ---------------------------------------------------------
class Trajectory(Layer):
    
    def __init__(self, name='trajectory', **kwargs):
        
        super(Trajectory, self).__init__(name=name, **kwargs)
        
    def call(self, inputs):
        
        parameters, times = inputs
        
        # get the heart beating pace and shift of each video
        freq = parameters[:, 0, None]
        shift = parameters[:, 1, None]

        # to turn the pace into frequency, we apply exponential to it
        # we compute t = 2*pi*freq_i*(t-tau_i) for each video i
        # then compute the values of the 1st and 2nd axis of the latent space (e1 and e2)
        # to impose a circular trajectory
        t = freq*(times - shift)*2*np.pi
        e1 = tf.sin(t)
        e2 = tf.cos(t)
        # replace the first 2 params of the parameters vector by the computed params e1 and e2
        l = tf.concat([e1, e2, parameters[:, 2:]], axis=1)

        return l
# -------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------- AutoEncoder ---------------------------------------------------------
class EchocardioModel(Model):
    
    def __init__(self, logger, latent_space_dim, batch_size=8, hidden_dim=32, learning_rate=8e-4, input_noise=None, log_dir=None, name='cardio', **kwargs):
        
        super(EchocardioModel, self).__init__(name=name, **kwargs)

        # save params in self
        self._supports_ragged_inputs = True
        
        self.params_dim = latent_space_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.input_noise = input_noise
        
        # init Encoder, Trajectory and Decoder
        self.encoder = Encoder(latent_space_dim=latent_space_dim, hidden_dim=hidden_dim, input_noise=input_noise)
        self.trajectory = Trajectory()
        self.decoder = Decoder(latent_space_dim=latent_space_dim)

        # save params in self
        self._sigma_squared = tf.Variable(1.0, trainable=False)

        self.log_dir = log_dir
        self.logger = logger

        # setup train metrics
        self._train_metrics = dict()
        train_metrics = ['loss', 'reconstruction_error', 'regularisation',
                         'freq_regularisation', 'heart_rate', 'regularisation_strength',
                         'reconstruction_stddev']
        for metric in train_metrics:
            self._train_metrics[metric] = tf.keras.metrics.Mean(metric)

        # setup validation metrics (these metrics compute the mean over the batch, e,g we feed it a sequence of validation errors and it computes the mean)
        self._validation_metrics = dict()
        val_metrics = ['reconstruction_error', 'heart_rate']
        for metric in val_metrics:
            self._validation_metrics[metric] = tf.keras.metrics.Mean(metric)

    def call(self, inputs, training=False):
        
        # inputs is tuple of 2 batches: batch of times sequence and batch of frames sequence
        # i'th elem of each batch correspond to the i'th video
        times, sequences = inputs

        # get the batch of video parameters of shape (batch_size, 128)
        params = self.encoder((times, sequences), training=training)

        # flatten batch of times and add a dim of size 1 at the end, so times_list.shape is (total_times, 1)
        times_list = tf.expand_dims(times.values, axis=-1)
        # repeat params of video i "nb_frames_i" times where "nb_frames_i" is the number of frames in video i
        # so times_list and params_list have the same 1st dim = "total_nb_frames" = "total_times"
        # we repeat along axis 0, so for each video i of the batch, we repeat it's parameters (i'th row of "params" Tensor)
        # "nb_frames_i" times along axis 0
        # So to get the 0th param of all videos (repeated as many times as the nb_frames 
        # of the corresponding video), we use params[:, 0]
        params_list = tf.repeat(params, times.row_lengths(), axis=0)

        # compute the latent trajectory params (replace 0th and 1st params of each video by e1 and e2 (repeated))
        # latent still has latent.shape = (total_times, 128)
        latents = self.trajectory((params_list, times_list))

        # turn into RaggedTensor of shape (batch_size, None, 128)
        # where for each video in "batch_size" axis, we have: None represents the nb_frames (i,e nb_timesteps) for that video
        # this RaggedTensor is not fed to the decoder but only returned
        latent_trajectories = tf.RaggedTensor.from_row_splits(latents, row_splits=times.row_splits)

        # decoder gets fed the Tensor "latents" of shape (total_times, 128)
        reconstructions = self.decoder(latents, training=training)
        # same operations as with the RaggedTensor "latent_trajectories", the "reconstructions" RaggedTensor is returned
        reconstructions = tf.RaggedTensor.from_row_splits(reconstructions, row_splits=times.row_splits)
        
        return params, latent_trajectories, reconstructions

    def encode(self, times, frames, return_freq_phase=False, training=False):

        # get the batch of video parameters of shape (batch_size, 128)
        params = self.encoder((times, frames), training=training)
        if return_freq_phase:
            # compute freq and phases from pace and shifts
            freqs = params[:, 0, None]
            shifts = params[:, 1, None]
            phases = freqs * shifts # QUESTION: Should constrain to [0, 1] or some other range of length 1?
            # update params to contain freq and phase
            params = tf.concat([freqs, phases, params[:, 2:]], axis=1)
        return params

    def decode(self, inputs):

        x_rec = self.decoder(inputs, training=False)
        return x_rec
    
    def decode2(self, params, times_values, row_lengths, passed_freqs_phases=False, training=False):
        if passed_freqs_phases:
            freqs = params[:, 0, None]
            phases = params[:, 1, None]
            shifts = tf.divide(phases, freqs)
            params = tf.concat([freqs, shifts, params[:, 2:]], axis=1)
        
        params_list = tf.repeat(params, row_lengths, axis=0)
        times_list = tf.expand_dims(times_values, axis=-1)

        latents = self.trajectory((params_list, times_list))

        reconstructions = self.decoder(latents, training=training)
        
        return reconstructions

    def _freq_regularisation(self, params):
        # penalise frequencies (heart rates) below 0.3bps and above 3.5bps
        freqs = params[:, 0]
        return tf.reduce_mean(tf.nn.relu(0.3 - freqs) + tf.nn.relu(freqs - 3.5))

    def _parameter_regularisation(self, params):
        # regularize the heart shape params only, not the trajectory params
        # we use L2 regularization (i,e sum squared of these params)
        return tf.reduce_mean(tf.reduce_mean(params[:,2:]**2, axis=-1))
    
    def _reconstruction_error(self, y_true, y_pred):
        # compute mean squared error
        mse = tf.reduce_mean(tf.reduce_mean((y_true - y_pred)**2, axis=[-3, -2, -1]))
        return mse
    
    def mse(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        return mse
    
    def mae(self, y_true, y_pred):
        mae = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))
        return mae
    
    def _loss(self, true_sequences, pred_sequences, params, echo_efs):
        
        # flatten sequences (frames) and calculate reconstruction error
        reconstruction_error = self._reconstruction_error(true_sequences, pred_sequences)

        # regularisation of the shape params
        regularisation = self._parameter_regularisation(params)

        # frequency regularization
        freq_regularization = self._freq_regularisation(params)

        # total (weighted) loss
        loss = reconstruction_error/self._sigma_squared + regularisation + freq_regularization
        
        return loss, reconstruction_error, regularisation, freq_regularization
    
    @tf.function
    def _train_step(self, times, sequences, echo_efs, optimizer):

        # Perform the forward pass and get the reconstructions and the losses
        with tf.GradientTape() as tape:
            params = self.encoder((times, sequences), training=True)
            times_list = tf.expand_dims(times.values, axis=-1)
            params_list = tf.repeat(params, times.row_lengths(), axis=0)
            latents = self.trajectory((params_list, times_list))
            reconstructed_sequences_seq = self.decoder(latents, training=True)
            reconstructed_sequences = tf.RaggedTensor.from_row_splits(reconstructed_sequences_seq, row_splits=times.row_splits)

            loss, reconstruction_error, regularisation, freq_regularisation = self._loss(sequences, reconstructed_sequences, params, echo_efs)
        
        # get the heart rates from the heart frequencies
        heart_rates = params[:, 0]*60
            
        # update model weights
        variables = self.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        # logging
        self._train_metrics['loss'](loss)
        self._train_metrics['reconstruction_error'](reconstruction_error)
        self._train_metrics['regularisation'](regularisation)
        self._train_metrics['freq_regularisation'](freq_regularisation)
        self._train_metrics['heart_rate'](heart_rates)
        self._train_metrics['regularisation_strength'](self._sigma_squared)
        
        means = tf.expand_dims(tf.reduce_mean(reconstructed_sequences, axis=1), axis=1)
        reconstruction_var = tf.reduce_mean(tf.reduce_mean((reconstructed_sequences - means)**2, axis=1))
        reconstruction_stddev = tf.sqrt(reconstruction_var)
        self._train_metrics['reconstruction_stddev'](reconstruction_stddev)

        # update regularisation weight
        update_rate = 0.99
        self._sigma_squared.assign(update_rate*self._sigma_squared + (1 - update_rate)*tf.reduce_mean(reconstruction_error))

    @tf.function
    def _evaluate(self, times, sequences, echo_efs):
        """
            Perform model validation/evaluation
        """
        # forward pass
        params = self.encoder((times, sequences), training=False)
        times_list = tf.expand_dims(times.values, axis=-1)
        params_list = tf.repeat(params, times.row_lengths(), axis=0)
        latents = self.trajectory((params_list, times_list))
        reconstructed_sequences = self.decoder(latents, training=False)

        # compute reconstruction error
        reconstruction_error = self._reconstruction_error(sequences.values, reconstructed_sequences)
        heart_rates = params[:, 0]*60
        
        # logging
        self._validation_metrics['reconstruction_error'](reconstruction_error)
        self._validation_metrics['heart_rate'](heart_rates)
    
    def fit(self, train_files, val_files, trained_model_path):
        
        logger = self.logger

        n_train_subjects = len(train_files)        
        n_val_subjects = len(val_files)

        val_batch_size = 16

        logger.info(f"{n_train_subjects} train samples")
        logger.info(f"{n_val_subjects} val samples")

        # Dataset generator
        def train_dataset_generator(files):

            n = len(files)
            while True: # infinite loop
                # load a train video randomly from train set
                ix = np.random.randint(0, n)
                filepath = files[ix]
                data = np.load(filepath)
                times = data['times']
                frames = data['frames']
                params = data['params']

                minimum_length = np.ceil(2.0/times[1]).astype('int') # nb frames of the video that make up a 2.0 seconds duration
                if not times.shape[0] <= minimum_length: # if video duration not less than 2.0 seconds (i,e greater than 2.0 seconds)
                    # cut video (i,e extract sub-sequence of frames between indices "start" and "end") to get a sub-video of duration
                    # at least 2.0 seconds, i,e the video shows same heart, with same frequency but different shift
                    # indices "start" and "end" are chosen randomly and "end - start" is at least "minimum_length" 
                    start, end = random_subsequence_start_end(times.shape[0], minimum_length)
                    times = times[start:end] - times[start] # extract subsequence of times, shift it to get first value = 0
                    frames = frames[start:end] # extract corresponding subsequence of frames
                
                frames = (frames/255).astype('float32') # normalize
                yield times, frames, params

        def val_dataset_generator(files):

            for filepath in files: # load validation files (times + frames)
                data = np.load(filepath)
                times = data['times']
                frames = data['frames']
                params = data['params']
                frames = (frames/255).astype('float32') # normalize
                yield times, frames, params
        
        # get Train and Val datasets
        logger.info("Loading train and validation datasets...")

        dataset_output_types = (tf.float32, tf.float32, tf.float32)
        dataset_output_shapes = (tf.TensorShape([None]), tf.TensorShape([None, 112, 112]), tf.TensorShape([7,]))
        train_dataset = tf.data.Dataset.from_generator(train_dataset_generator, dataset_output_types, dataset_output_shapes, args=(train_files,))
        val_dataset = tf.data.Dataset.from_generator(val_dataset_generator, dataset_output_types, dataset_output_shapes, args=(val_files,))

        # messy batching, as RaggedTensors not fully supported by Tensorflow's Dataset
        train_dataset = train_dataset.map(lambda x, y, z: (tf.expand_dims(x, 0), tf.expand_dims(y, 0), tf.expand_dims(z, 0)))
        train_dataset = train_dataset.map(lambda x, y, z: (tf.RaggedTensor.from_tensor(x), tf.RaggedTensor.from_tensor(y), tf.RaggedTensor.from_tensor(z)))
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.map(lambda x, y, z: (tf.squeeze(x, axis=1), tf.squeeze(y, axis=1), tf.squeeze(z, axis=1)))

        val_dataset = val_dataset.map(lambda x, y, z: (tf.expand_dims(x, 0), tf.expand_dims(y, 0), tf.expand_dims(z, 0)))
        val_dataset = val_dataset.map(lambda x, y, z: (tf.RaggedTensor.from_tensor(x), tf.RaggedTensor.from_tensor(y), tf.RaggedTensor.from_tensor(z)))
        val_dataset = val_dataset.batch(val_batch_size)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.map(lambda x, y, z: (tf.squeeze(x, axis=1), tf.squeeze(y, axis=1), tf.squeeze(z, axis=1)))
        
        logger.info("Loaded Datasets!")

        # setup summary writers
        train_log_dir = str(self.log_dir) + '/train'
        val_log_dir = str(self.log_dir) + '/validation'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        # optimizer
        optimizer = Adam(learning_rate=self.learning_rate)

        opt_weights = self.get_weights()
        opt_early_stopping_metric = np.inf

        # train params: patience, count, epoch, max_epochs
        patience = 10
        count = 0
        epoch = 0
        max_epochs = 100

        # TRAIN LOOP
        logger.info("\nStart Training...")

        echo_ef_index = ECHO_DATA_PARAMS.index("EF")

        while count < patience and epoch <= max_epochs:
            epoch += 1
            count += 1
            steps_per_epoch = 200 # steps per epoch

            t1_steps = time.time()

            # Perform a train epoch
            for times, sequences, echo_params in train_dataset.take(steps_per_epoch):
                echo_params = echo_params.to_tensor(default_value=0.0)
                echo_efs = echo_params[:, echo_ef_index, None] / 100.0
                self._train_step(times, sequences, echo_efs, optimizer)
            
            t2_steps = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2_steps - t1_steps)
            logger.info("\n{} steps done in {}:{}:{}\n".format(steps_per_epoch, h, m, s))
            
            # log train metrics
            with train_summary_writer.as_default():
                for metric in self._train_metrics.keys():
                    tf.summary.scalar(metric, self._train_metrics[metric].result(), step=epoch)

            # print train metrics
            train_strings = ['%s: %.3e' % (k, self._train_metrics[k].result()) for k in self._train_metrics.keys()]
            logger.info(f'Epoch {epoch}: Train: ' + ' - '.join(train_strings))

            t1_val = time.time()
            # validate
            for times, sequences, echo_params in val_dataset:
                echo_params = echo_params.to_tensor(default_value=0.0)
                echo_efs = echo_params[:, echo_ef_index, None] / 100.0
                self._evaluate(times, sequences, echo_efs)
            
            t2_val = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2_val - t1_val)
            logger.info("Validation done in {}:{}:{}\n".format(h, m, s))

            # log validation metrics
            with val_summary_writer.as_default():
                for metric in self._validation_metrics.keys():
                    tf.summary.scalar(metric, self._validation_metrics[metric].result(), step=epoch)

            # print validation metrics
            val_strings = ['%s: %.3e' % (k, self._validation_metrics[k].result()) for k in self._validation_metrics.keys()]
            logger.info(f'Epoch {epoch}: Validation: ' + ' - '.join(val_strings))

            # reset early stopping counter on improvement and save optimal weights
            if self._validation_metrics['reconstruction_error'].result() < opt_early_stopping_metric:
                opt_early_stopping_metric = self._validation_metrics['reconstruction_error'].result()
                opt_weights = self.get_weights()
                count = 0
                logger.info(f"Validation loss improved, saving model under {trained_model_path}")
                self.save_weights(trained_model_path) # save trained model weights in model path
            else:
                logger.info(f"Validation loss did not improve from {opt_early_stopping_metric}. Counter: {count}")
            
            # reset metrics (reset the sequence of metrics computed for this epoch)
            for metric in chain(self._train_metrics.values(), self._validation_metrics.values()):
                metric.reset_states()

        # reset to optimal weights
        self.set_weights(opt_weights)

# -------------------------------------------------------------------------------------------------------------------------------

    def reconstruct_echo_vid(self, echo_dataset, echo_filenames, set_name, global_step, output_dir=None, save_original=False):

        logger = self.logger

        if output_dir is None:
            output_dir = self.log_dir / "visualizations" / set_name / "echo_reconstructions"
            output_dir_echo = output_dir / "OriginalEchos"
            output_dir_echo_rec = output_dir / f"GS{str(global_step).zfill(3)}"
        
        logger.info("\nEncoding and reconstructing echo videos...")

        # parallel jobs
        max_parallel_jobs = mp.cpu_count()
        parallel_jobs = []

        echo_ef_index = ECHO_DATA_PARAMS.index("EF")
        
        t_start = time.time()
        t1 = time.time()
        batch_size = -1
        for b, (echo_times, echo_frames, echo_params) in enumerate(echo_dataset): # b'th batch
            logger.info(f"Batch {b} start...")

            # translate echo vids to mesh vids aligned in time (transform relevant arrays to numpy)
            echo_latents = self.encode(echo_times, echo_frames, return_freq_phase=True, training=False)
            # replace ef value with different one
            echo_vids_rec = self.decode2(echo_latents, echo_times.values, echo_times.row_lengths(), passed_freqs_phases=True, training=False)

            # get row lengths and row limits
            row_lengths = echo_times.row_lengths().numpy()
            row_limits = np.cumsum(row_lengths)
            if batch_size == -1: # get batch_size
                batch_size = row_lengths.shape[0]
                logger.info(f"Batch size: {batch_size}")
            
            # convert to numpy
            echo_vids = echo_frames.values.numpy()
            echo_vids_rec = echo_vids_rec.numpy()
            echo_times = echo_times.values.numpy()
            echo_params = echo_params.to_tensor(default_value=0.0).numpy()
            
            i = 0
            for k, j in enumerate(row_limits): # k'th video in the batch "b"
                echo_vid = echo_vids[i:j]
                echo_vid_rec = echo_vids_rec[i:j]
                echo_vid_times = echo_times[i:j]
                global_index = b * batch_size + k # index of the video in the whole dataset
                echo_filename = echo_filenames[global_index]
                vid_duration = echo_vid_times[-1]
                echo_ef = echo_params[k, echo_ef_index]
                
                vid_name = f"vid_{global_index}_ef_{echo_ef:.2f}_name_{echo_filename}"

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
            
            if b % 10 == 0: # print progress
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b} after {h}:{m}:{s}")
        
        # wait for all parallel jobs to complete
        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)
        
        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
        logger.info(f"Done all batches in {h}:{m}:{s}")
