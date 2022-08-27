import logging, time, yaml, sys
from numpy.lib.function_base import diff
from itertools import chain, groupby
import tensorflow as tf
import numpy as np
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, LSTM, Bidirectional, BatchNormalization, PReLU
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import losses

from source.shape_model_utils import overwrite_vtkpoly
from source.shape_model_utils import save_mesh_vid
from source.shape_model_utils import compute_volumes_feats
from source.shape_model_utils import compute_ef_mesh_vid
import source.utils as utils
from source.constants import ROOT_LOGGER_STR, CONRAD_DATA_PARAMS, CONRAD_COMPONENTS

all_parallel_jobs = [] # cpu processes for videos and volume plots generation

# get the logger
logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

LEOMED_RUN = "--leomed" in sys.argv # True if running on LEOMED

if LEOMED_RUN:
    matplotlib.use('Agg')

# TODO: add to constants the number of vertices for each configuration of components.
# feats_values_shape = tf.TensorSpec(shape=(None, 16430, 3), dtype=tf.float32)
feats_values_shape = tf.TensorSpec(shape=(None, 15291, 3), dtype=tf.float32)


# --------------------------------------------------------- Encoder ---------------------------------------------------------
# using Model instead of Layer (issue: https://github.com/tensorflow/tensorflow/issues/38211)
class Encoder(Model):

    def __init__(self, L, D, params, name='encoder', freeze=False, **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        # save params in self
        self.params = params
        
        conv_params = params['conv_layers']

        self.D = [
            tf.Variable(down.toarray(), name=f"conv_d_sampling_mat_{i}", trainable=(self.params['trainable_down'] and not freeze), dtype=tf.float32)
            for i, down in enumerate(D)
            ]

        self.D_b = [
            tf.Variable(tf.zeros([d.shape[0], p['channels']]), trainable=(self.params['trainable_down'] and not freeze), name=f"conv_d_sampling_bias_{i}", dtype=tf.float32)
            for i, (d, p) in enumerate(zip(D, conv_params))
            ]
        
        # L = input matrices to each conv layer, aka enc_in. Convert these matrices to tensors. lap = laplacian matrix, a = adjacency matrix
        self.L = [
            tf.sparse.from_dense(a.toarray(), name=f"adjacency_{i}")
            for i, a in enumerate(L)
            ]

        # create and save successive Graph-Conv layers in list, as well as the input matrix (one per Graph conv)
        self.conv_layers = []
        self.L = [] # TF matrices input to the graph conv layer (Laplacian or Adjacency or ...)
        for i, layer_params in enumerate(conv_params):
            layer = utils.get_conv_layer(layer_params.get('name'))
            self.conv_layers.append(layer(**layer_params))
            # Save the input to graph conv layer
            if layer_params["name"] == "GeneralConv":
                l = tf.sparse.from_dense(L[i].toarray(), name=f"encoder_adjacency_{i}")
            else:
                l = tf.convert_to_tensor(L[i].toarray(), dtype=tf.float32)
            self.L.append(l)
        
        # create LSTM layer
        lstm_params = params['lstm_layer']
        lstm_name = None
        if "name" in lstm_params:
            lstm_name = lstm_params["name"]
        if lstm_name is None or lstm_name == "lstm":
            self.lstm = LSTM(**lstm_params)
        elif lstm_name == "bidir_lstm":
            self.lstm = Bidirectional(LSTM(**lstm_params))

        # create an save Dense layer
        self.lstm_fc = Dense(**params['lstm_dense_layer'])
        
        self.shape_fcs = []
        if "shape_params_dense_layers" in params:
            shape_fcs_params = params['shape_params_dense_layers']

            for layer_params in shape_fcs_params:
                self.shape_fcs.append(Dense(**layer_params))

    
    def call(self, inputs, training=False):
        """
        :param inputs: a mesh consisting of adj. matrix (NxN) and
        feature matrix (NxF)
        Recall, adjacency matrix only gives us which vertices are connected to which other vertices
        but the features matrix is the actual 3D shape since it gives us for each vertex it's 3 features: x, y and z
        so F = 3
        :param training: if it's training or validation
        :return features: latent features
        """

        features, times, row_lengths = inputs

        h = features

        for i in range(len(self.D)): # apply conv_d_sampling layers successively

            assert h.shape[1] == self.L[i].shape[0], \
                f"incompatible shapes between feature matrix {h.shape} and graph structure matrix {self.L[i].shape}"

            # Graph Conv
            h = self.conv_layers[i]((h, self.L[i])) # apply conv_d_sampling layer on the features
            # Down-sample
            # multiply each frame in the batch (sequence of frames of all videos of the batch) by the downsampling matrix
            # memory issues ?
            if i < 0:
                h = tf.map_fn(fn=lambda elem: tf.matmul(self.D[i], elem), elems=h)
            else:
                h = tf.matmul(self.D[i], h)

            h = h + self.D_b[i] # apply downsampling matrix to features then add bias matrix
        
        # flatten
        last_conv_channels = self.params['conv_layers'][-1]['channels']
        flat_dim = self.D[-1].shape[0]*last_conv_channels
        h = tf.reshape(h, shape=(-1, flat_dim))

        # add times to feature vectors
        h = tf.concat([h, tf.expand_dims(times, axis=-1)], axis=-1)
        # we also concat to each frame's feature vector the total nb of frames in it's corresponding video
        # nb_frames = tf.cast(tf.repeat(tf.expand_dims(times.row_lengths(), axis=-1), times.row_lengths(), axis=0), tf.float32)
        # h = tf.concat([h, nb_frames], axis=-1)

        # put features vectors of frames of same video together
        h = tf.RaggedTensor.from_row_lengths(h, row_lengths=row_lengths)

        # pad RaggedTensor with zeros to turn it to a tf Tensor
        h_padded = h.to_tensor(default_value=0.0)
        
        # pick which feature vectors to mask from the lstm (the ones used for padding)
        h_mask = tf.sequence_mask(row_lengths, maxlen=tf.shape(h_padded)[1])        

        # input sequences of feature vectors to lstm, get output vectors (one per video)
        h = self.lstm(h_padded, mask=h_mask, training=training)


        h = self.lstm_fc(h) # apply FC at the end of encoder

        freqs = h[:, 0, None]
        phases = h[:, 1, None]
        p = h[:, 2:] # shape params

        # apply fcs to shape params to further reduce it's dimentionality
        for fc in self.shape_fcs:
            p = fc(p)
        
        h = tf.concat([freqs, phases, p], axis=1)


        # exponentiate the pace to avoid negative values
        # pace = h[:, 0, None]
        # freqs = tf.exp(pace)
        # h = tf.concat([freqs, h[:, 1:]], axis=1)
        # pre_phases = h[:, 1, None]
        # phases = pre_phases
        # h = tf.concat([freqs, phases, h[:, 2:]], axis=1)

        return h
# ---------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------- Trajectory --------------------------------------------------------
# using Model instead of Layer (issue: https://github.com/tensorflow/tensorflow/issues/38211)
class Trajectory(Model):
    
    def __init__(self, name='trajectory', **kwargs):
        
        super(Trajectory, self).__init__(name=name, **kwargs)
    
    def call(self, inputs):
        
        parameters, times = inputs
        
        # get the heart beating freq and shift of each video
        freqs = parameters[:, 0, None]
        phases = parameters[:, 1, None]

        # phases = tf.math.floormod(phases, 1)

        # for each video i, compute the values of the 1st and 2nd axis of the latent space (e1 and e2)
        # to impose a circular trajectory
        t = freqs*times*2*np.pi + phases*2*np.pi
        e1 = tf.sin(t)
        e2 = tf.cos(t)
        # replace the first 2 params of the parameters vector by the computed params e1 and e2
        l = tf.concat([e1, e2, parameters[:, 2:]], axis=1)

        return l
# ---------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------- Decoder ---------------------------------------------------------
# using Model instead of Layer (issue: https://github.com/tensorflow/tensorflow/issues/38211)
class Decoder(Model):

    def __init__(self, L, U, data_handler, params, name='decoder', freeze=False, **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.params = params
        self.data_handler = data_handler

        # Grap conv layer params, each conv layer is preceded by an upsampling
        conv_params = params['conv_layers']
    
        # Reoder upsampling matrices from lowest to highest
        U = U[::-1]
        trainable_up = self.params['trainable_up'] and not freeze # whether or not the upsampling matrices are trainable 

        # input channels to the 1st (lowest) conv layer chosen to be same as it's output channels (could also choose another value)
        self.num_channels_input = conv_params[0]['channels']

        # Dense layer output features of shape v * c (v = vertices, c = channels) to be reshaped to 
        # output features matrix of shape v x c where v = nb_columns of U[0] for compatibility 
        # since we multiply U[0] * v, but c = nb_channels is chosen as above
        self.params['dense_layer']['units'] = U[0].shape[-1] * self.num_channels_input
        self.fc = Dense(**self.params['dense_layer'])

        self.L = [] # TF matrix input to the graph conv layer (Laplacian or Adjacency or ...)
        self.conv_layers = [] # Graph conv layers
        self.U = [] # TF upsampling matrices
        self.U_b = [] # TF upsampling biases

        for i in range(len(conv_params)): # for each conv layer
            # Create the upsampling layer
            u = tf.Variable(U[i].toarray(), name=f'conv_u_sampling_mat_{i}', trainable=trainable_up, dtype=tf.float32)
            u_b_dim = self.num_channels_input if i == 0 else conv_params[i-1]["channels"]
            u_b = tf.Variable(tf.zeros([U[i].shape[0], u_b_dim]), trainable=trainable_up, name=f"conv_u_sampling_bias_{i}",dtype=tf.float32)
            self.U.append(u)
            self.U_b.append(u_b)
            # Create the Graph conv layer
            layer_params = conv_params[i]
            layer = utils.get_conv_layer(layer_params["name"])
            self.conv_layers.append(layer(**layer_params))
            # Save the input to graph conv layer
            if layer_params["name"] == "GeneralConv":
                l = tf.sparse.from_dense(L[i].toarray(), name=f"decoder_adjacency_{i}")
            else:
                l = tf.convert_to_tensor(L[i].toarray(), dtype=tf.float32)
            self.L.append(l)
    
    def call(self, inputs, training=False):

        # apply dense layer and reshape input for following up_conv_layers
        latents = inputs
        features = self.fc(latents)
        features = tf.reshape(features, shape=(-1, self.U[0].shape[-1], self.num_channels_input))

        # apply up_sampling + graph conv
        for i in range(len(self.conv_layers)):
            if i < len(self.conv_layers) - 2:
                features = tf.matmul(self.U[i], features)
            else:
                features = tf.map_fn(fn=lambda elem: tf.matmul(self.U[i], elem), elems=features)
            
            # apply up_sampling
            features = features + self.U_b[i]

            # apply Graph Conv
            assert features.shape[1] == self.L[i].shape[0], \
                f"incompatible shapes between feature matrix {features.shape} and graph structure matrix {self.L[i].shape}"
            
            features = self.conv_layers[i]((features, self.L[i]))

        return features

# ---------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------- DHB AutoEncoder -----------------------------------------------------
class DHBMeshAutoencoderModel(Model):
    def __init__(self, data_handler, log_dir, training_params, model_params, save_metrics=True, freeze=False, build_MAE=True, **kwargs):
        # maybe add **kwargs if properly defined
        super(DHBMeshAutoencoderModel, self).__init__(name='autoencoder')

        # save data handler
        self.data_handler = data_handler

        # save experiment path in self
        self.log_dir = log_dir # current experiment log dir (datetime dir)
        
        # self.input_dim_mesh = data_handler.input_dim_mesh # None for CONRADData

        # save training and model parameters from config file in self
        self.training_params = training_params
        self.model_params = model_params

        # Prior
        self.use_laplacian_regularizer = False # by default not using Laplacian prior regularization
        # if training params in config file specify usage of Laplacian prior regularization, use it
        if 'compute_laplacian_prior' in training_params or 'laplacian_path' in training_params:
            self.use_laplacian_regularizer = True
        
        # build the Autoencoder model
        self.build_model(model_params, data_handler, freeze, build_MAE)
        # data_handler.built = True
        # data_handler.dataset_output_types = (tf.float32, tf.float32, tf.float32, tf.float32)
        
        # setup summary writers and metrics
        if save_metrics:
            self.create_metrics_writers()

        logger.info("DHB MAE model initialized and built.")

    def build_model(self, model_params, data_handler, freeze=False, build_MAE=True):

        # list of names of consecutive conv layers of encoder in config file. 
        # Each element of this list is a dict representing the params of the conv layer 
        # (layer name, nb channels, activation func, use bias ...)
        enc = [l['name'] for l in model_params['encoder']['conv_layers']]
        # likewise for the decoder
        dec = [l['name'] for l in model_params['decoder']['conv_layers']]

        _, _, D, U, enc_in, dec_in = data_handler.build_transform_matrices(enc, dec)
        
        if not build_MAE: # stop after setting ref_poly and dataset values of datahandler
            return
        
        
        # save down- and up-sampling matrices
        self.downsampling_matrices = D
        self.upsampling_matrices = U

        # init encoder and decoder
        self.encoder = Encoder(enc_in, D, params=model_params['encoder'], freeze=freeze)
        self.trajectory = Trajectory()
        self.decoder = Decoder(dec_in, U, data_handler, params=model_params['decoder'], freeze=freeze)

    def create_metrics_writers(self):
        
        train_metrics = ['loss', 'reconstruction_error', 'ef_loss', 'learning_rate']
        val_metrics = ['reconstruction_error', 'ef_loss']

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
    
    # ------------------------------------------------- Train and validation steps --------------------------------------
    # use @tf.function to turn this function into a TF Graph
    # use input signatures to avoid creating a new TF Graph every time the inputs shape changes (due to variable video lengths)

    @tf.function(
        input_signature=[
            feats_values_shape,
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.bool)
            ])
    def call(self, feats_values, times_values, row_lengths, training=False):
        """
            Run Model End-to-End and return reconstructed shapes
        """

        # get latent parameters
        params = self.encoder((feats_values, times_values, row_lengths), training=training)
        
        # repeat params of each video as many times as the nb timesteps of that video
        params_list = tf.repeat(params, row_lengths, axis=0)
        # flatten batch of times and add a dim of size 1 at the end, so times_list.shape is (total_times, 1)
        times_list = tf.expand_dims(times_values, axis=-1)

        # get trajectory latent vector (enforce circular trajectory)
        latents = self.trajectory((params_list, times_list))

        # get decoder reconstructions
        reconstructions = self.decoder(latents, training=training)

        return params, latents, reconstructions
    
    @tf.function(
        input_signature=[
            feats_values_shape,
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
            ])
    def _train_step(self, feats_values, times_values, true_efs, row_lengths):
        """
            feats_values: frames of all videos in the batch, in order from first frame of first video to last frame of last video.
            These frames are stacked as one large batch.
            times_values: like feats_values for the time points of each frame of each video
            row_lengths: nb of frames of each video
        """
        with tf.GradientTape() as tape:
            # run model end-to-end in train mode
            pred_params, _, reconstructions = self.call(feats_values, times_values, row_lengths, training=True)
            loss, rec_error, ef_loss = self._loss(feats_values, reconstructions, pred_params, true_efs, row_lengths)
        
        # update model weights
        variables = self.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        losses = { 'loss': loss,
                   'reconstruction_error': rec_error,
                   'ef_loss': ef_loss
        }

        return losses
    
    @tf.function(
        input_signature=[
            feats_values_shape,
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
            ])
    def _val_step(self, feats_values, times_values, true_efs, row_lengths):
        # run model end-to-end in non-training mode
        pred_params, _, reconstructions = self.call(feats_values, times_values, row_lengths, training=False)
        rec_error = self._reconstruction_error(feats_values, reconstructions, row_lengths)
        ef_loss = self._ef_loss(pred_params, true_efs)

        losses = { 'reconstruction_error': rec_error,
                   'ef_loss': ef_loss
        }

        return losses
    
    # ------------------------------------------------ Encoding and Decoding --------------------------------------------
    def encode(self, feats_values, times_values, row_lengths, training=False):
        return self.encoder((feats_values, times_values, row_lengths), training=training)

    def decode(self, params, times_values, row_lengths, training=False):
        # repeat params of each video as many times as the nb timesteps of that video
        params_list = tf.repeat(params, row_lengths, axis=0)
        # flatten batch of times and add a dim of size 1 at the end, so times_list.shape is (total_times, 1)
        times_list = tf.expand_dims(times_values, axis=-1)

        # get trajectory latent vector (enforce circular trajectory)
        latents = self.trajectory((params_list, times_list))

        # get decoder reconstructions
        reconstructions = self.decoder(latents, training=training)

        return reconstructions
    
    # --------------------------------------------------- Computing losses ----------------------------------------------
    def _loss(self, true_mesh_features, pred_mesh_features, pred_params, true_efs, row_lengths):

        rec_error = self._reconstruction_error(true_mesh_features, pred_mesh_features, row_lengths)
        freq_reg = self._freq_regularisation(pred_params)
        ef_loss = self._ef_loss(pred_params, true_efs)

        loss = rec_error + freq_reg + ef_loss
        
        return loss, rec_error, ef_loss

    def _reconstruction_error(self, true_mesh_features, pred_mesh_features, row_lengths):
        # compute reconstruction error
        loss = 0
        which_loss = self.training_params['loss_str']
        true_mesh_features = tf.RaggedTensor.from_row_lengths(true_mesh_features, row_lengths=row_lengths)
        pred_mesh_features = tf.RaggedTensor.from_row_lengths(pred_mesh_features, row_lengths=row_lengths)
        if which_loss == 'l1':
            batch_error = tf.reduce_mean(tf.abs(true_mesh_features - pred_mesh_features), axis=[-1, -2, -3])
            loss = tf.reduce_mean(batch_error)
        elif which_loss == 'l2':
            batch_error = tf.reduce_mean((true_mesh_features - pred_mesh_features)**2, axis=[-1, -2, -3])
            loss = tf.reduce_mean(batch_error)
        else:
            raise NotImplementedError()
        return loss
    
    def _freq_regularisation(self, params):
        # penalise frequencies (heart rates) below 0.3bps and above 3.5bps
        freqs = params[:, 0]
        return tf.reduce_mean(tf.nn.relu(0.5 - freqs) + tf.nn.relu(freqs - 3.5))
    
    def mse(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        return mse
    
    def mae(self, y_true, y_pred):
        mae = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))
        return mae

    def _ef_loss(self, params, true_efs):
        pred_efs = params[:, 2, None]
        if self.training_params["ef_loss_str"] == "l1":
            return self.mae(true_efs, pred_efs)
        elif self.training_params["ef_loss_str"] == "l2":
            return self.mse(true_efs, pred_efs)
        else:
            return tf.constant(0.0)
    # -------------------------------------------------------------------------------------------------------------------

    def log_weights(self):
        # log number of trainable weights
        logger.info(f"\nTrainable variables names: {[v.name for v in self.trainable_variables]}")
        n_trainable = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_variables])
        logger.info(f"Number of trainable variables: {n_trainable}\n")

    def fit(self, train_dataset, train_plotting_dataset, val_dataset, test_dataset, test_dataset_viz, last_train_params=None):
        global all_parallel_jobs

        opt_early_stopping_metric = np.inf
        count = epoch = epoch_step = global_step = 0
        lr = self.training_params['learning_rate']
        beta_1 = self.training_params['beta_1']
        beta_2 = self.training_params['beta_2']
        patience = self.training_params['patience']
        max_patience = self.training_params['max_patience']
        learning_rate_decay = self.training_params['decay_rate']

        opt_weights = self.get_weights()
        if last_train_params is not None: # load model at last epoch with last learning rate
            epoch = last_train_params['epoch']
            epoch_step = last_train_params['epoch_step']
            global_step = last_train_params['global_step']
            lr = last_train_params['learning_rate']
            opt_early_stopping_metric = last_train_params['best_val_metric']
            # update optimal weight after loading best model
            self.load_me(epoch, name='best')
            opt_weights = self.get_weights()
            # load last model (not necessarily best one) to continue training
            # self.load_me(epoch, name='last')
            logger.info(f"Continuing training at epoch {epoch}, epoch_step {epoch_step}, global_step {global_step}")
        
        trained_model_dir = self.log_dir / "trained_models"
        trained_model_dir.mkdir(parents=True, exist_ok=True)
        last_params_path = trained_model_dir / "last_params.yml"
        logger.info(f"lr: {lr}, beta_1: {beta_1}, beta_2: {beta_2}, decay rate: {learning_rate_decay}, patience: {patience}, max_patience: {max_patience}")
        self.optimizer = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2) # use Adam as optimizer

        num_steps = self.training_params['num_steps']
        max_epochs = epoch + self.training_params['num_epochs'] # continue for num_epochs additional epochs (at most)
        train_plot_freq = self.training_params['train_plot_freq']

        logger.info("Start training...\n")

        logged_weights = False # log weights after the first forward pass on the model (to take Keras weights into considerations as well)

        t1_steps = time.time() # start epoch timer
        
        reconstruct_test = False # boolean for when the validation loss improves to reconstruct and plot on test data
        self.epoch = epoch # save epoch nb in self

        early_stopping = False

        true_efs_biplane_index = CONRAD_DATA_PARAMS.index("EF_Biplane")

        while epoch < max_epochs:
            if early_stopping:
                break

            train_dataset_skipped = train_dataset.skip(epoch_step) # skip epoch step batches to start at the batch where the last
                                                                   # experiment stopped
            for feats, times, _, mesh_params in train_dataset_skipped: # loop over whole dataset = 1 epoch
                mesh_params = mesh_params.to_tensor(default_value=0.0).numpy()
                true_efs = mesh_params[:, true_efs_biplane_index, None] / 100.0
                # perform a train step and log metrics
                losses = self._train_step(feats.values, times.values, true_efs, times.row_lengths())
                self._log_metrics(losses, self._train_metrics)
                # self.call(feats.values, times.values, times.row_lengths(), training=False)
                # exit(0)

                if not logged_weights: # log weights
                    with self._train_summary_writer.as_default():
                        tf.summary.graph(self.call.get_concrete_function(feats.values, times.values, times.row_lengths(), tf.constant(False)).graph)
                    self.log_weights()
                    logged_weights = True
                
                global_step += 1 # increment nb total of steps
                epoch_step += 1 # increment nb of epoch steps
                # print(f"Global step: {global_step}, Epoch step: {epoch_step}")

                # if performed num_steps, then validate
                if global_step % num_steps == 0:
                    self._train_metrics['learning_rate'](lr) # log learning rate with train metrics

                    # stop "steps timer" log train metrics and log "steps time"
                    t2_steps = time.time()
                    h, m, s = utils.get_HH_MM_SS_from_sec(t2_steps - t1_steps)
                    self.write_summaries(self._train_summary_writer, self._train_metrics, epoch, epoch_step, global_step, "Train")
                    logger.info("{} steps done in {}:{}:{}\n".format(num_steps, h, m, s))
                    
                    # Validation at the end of every epoch.
                    logger.info("Computing validation error...")
                    t1_val = time.time() # start validation timer

                    for feats_v, times_v, _, mesh_params_v in val_dataset:
                        mesh_params_v = mesh_params_v.to_tensor(default_value=0.0).numpy()
                        true_efs_v = mesh_params_v[:, true_efs_biplane_index, None] / 100.0
                        # perform a validation step and log metrics
                        losses = self._val_step(feats_v.values, times_v.values, true_efs_v, times_v.row_lengths())
                        self._log_metrics(losses, self._val_metrics)
                    
                    # stop validation timer, log validation metrics and log epoch time
                    t2_val = time.time()
                    h, m, s = utils.get_HH_MM_SS_from_sec(t2_val - t1_val)
                    self.write_summaries(self._val_summary_writer, self._val_metrics, epoch, epoch_step, global_step, "Validation")
                    logger.info("Validation done in {}:{}:{}\n".format(h, m, s))

                    # get reconstruction error on validation data
                    val_rec_error = self._val_metrics['reconstruction_error'].result()
                    # if new validation loss worse than previous one:
                    if val_rec_error >= opt_early_stopping_metric:
                        count += 1
                        logger.info(f"Validation loss did not improve from {opt_early_stopping_metric}. Counter: {count}")

                        if count == max_patience:
                            early_stopping = True
                            break

                        # update learning rate after waiting "patience" validations, decay learning rate (by multiplying by the decay_rate) 
                        # to see if any improvement on validation loss can be seen
                        if patience > 0 and count % patience == 0:
                            lr = float(self.optimizer.learning_rate * learning_rate_decay) # new learning rate
                            logger.info(f"Reduced learning rate to {lr} using decay_rate {learning_rate_decay}")
                            self.optimizer.learning_rate = lr
                    else: # validation loss improved
                        logger.info("Validation loss improved, saving model.")
                        opt_early_stopping_metric = float(val_rec_error)
                        opt_weights = self.get_weights()

                        # save best model
                        self.save_me(epoch, name='best')

                        # reset counter
                        count = 0

                        reconstruct_test = True # reconstruct on test data (after saving last moel) since validation loss improved
                    
                    # save last model and last params to be able to continue training
                    self.save_me(epoch, name='last')
                    self.save_last_params(last_params_path, epoch, epoch_step, global_step, lr, opt_early_stopping_metric)

                    # reset metrics
                    self.reset_metrics()
                    
                    # Plotting and reconstructions
                    if reconstruct_test: # validation loss improved, reconstruct on test data
                        reconstruct_test = False # reset
                        # save reconstructions on test data whenever the validation loss improves
                        save_original = (global_step == num_steps)
                        self.reconstruct_test_dataset(test_dataset_viz, epoch, epoch_step, save_original, limit=-1)

                        # generate scatter plots (freq, phase and ef)
                        self.generate_freq_phase_ef_plots(test_dataset, "test", epoch, epoch_step)
                    
                    # # plot on train dataset once for every "train_plot_freq" validations
                    # if global_step % (train_plot_freq*num_steps) == 0 or global_step == num_steps:
                    #     self.generate_freq_phase_ef_plots(train_plotting_dataset, "train", epoch, epoch_step)
                    
                    # check after every num_steps for parallel jobs which terminated and close them, jeeping only alive jobs
                    all_parallel_jobs = utils.get_alive_jobs(all_parallel_jobs)

                    # start steps timer
                    t1_steps = time.time()

            # increment epoch nb and reset nb steps
            epoch += 1
            self.epoch = epoch # save epoch nb in self
            epoch_step = 0
            # save last train params (to update epoch and epoch_step)
            self.save_last_params(last_params_path, epoch, epoch_step, global_step, lr, opt_early_stopping_metric)
        
        # stop training
        logger.info(f"Training stopped! Early stopping: {early_stopping}")
        self.reconstruct_test_dataset(test_dataset_viz, epoch, epoch_step, limit=-1)

        # reset to optimal weights
        self.set_weights(opt_weights)

        utils.wait_parallel_job_completion(all_parallel_jobs, 1)
        
        logger.info(f"Finished model training at epoch {epoch}")

# --------------------------------------------- Metrics/Summaries + Load/Save Model ---------------------------------
    def _log_metrics(self, losses, metrics):
        for metric in losses:
            metrics[metric](losses[metric])
    
    def reset_metrics(self):
        metrics = chain(self._train_metrics.values(), self._val_metrics.values())
        for metric in metrics:
            metric.reset_states()
    
    @staticmethod
    def write_summaries(summary_writer, metrics, epoch, epoch_step, global_step, log_str):
        with summary_writer.as_default():
            for metric, value in metrics.items():
                tf.summary.scalar(metric, value.result(), step=global_step)

        # print train metrics
        strings = ['%s: %.5e' % (k, v.result()) for k, v in metrics.items()]
        logger.info(f"\nEpoch {epoch}, Step: {epoch_step} | {log_str}: {' - '.join(strings)} ")

    def save_me(self, epoch, name=None):
        if name is None:
            name = "E_" + str(epoch)
        
        trained_model_path = self.log_dir / "trained_models" / "CoMA_DHB"
        scale_path = self.log_dir / 'scales.yml'
        self.save_weights(str(trained_model_path) + f"_{name}")
        scales = self.data_handler.scales
        with scale_path.open(mode="w") as scale_file:
            yaml.dump(scales, scale_file)
        
        if name == "best":
            logger.info(f"Model saved to file {trained_model_path}")
    
    def save_last_params(self, path, epoch, epoch_step, global_step, lr, opt_early_stopping_metric):
        # save last train params:
        with path.open(mode="w") as last_params_file:
            last_train_params = {'epoch': epoch, 'epoch_step': epoch_step, 'global_step': global_step,
                'learning_rate': lr, 'best_val_metric': opt_early_stopping_metric}
            yaml.dump(last_train_params, last_params_file)

    def load_me(self, epoch_n, name=None):

        trained_model_path = self.log_dir / "trained_models" / "CoMA_DHB"
        scale_path = self.log_dir / 'scales.yml'
        if name is None:
            model_path = str(trained_model_path) + f"E_{epoch_n}"
        else:
            model_path = str(trained_model_path) + f"_{name}"

        self.load_weights(model_path).expect_partial()
        # with scale_path.open(mode="r") as scale_file:
        #     scales = yaml.safe_load(scale_file)
        # if scales != self.data_handler.scales:
        #     logger.error(f"Scales with which this model is trained are "
        #                  f"different from those of the data processing. "
        #                  f"Make sure you know what's happening. For now, "
        #                  f"the model's scales will be set to the data "
        #                  f"handler.")
        #     self.data_handler.set_scales(scales)
        logger.info(f"Model loaded from {model_path}")
# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------- Test data reconstruction -------------------------------------------
    def reconstruct_test_dataset(self, test_dataset, epoch, epoch_step, save_original=False, save_vtps=False, limit=3):
        t1 = time.time()

        global all_parallel_jobs
        logger.info(f"Making reconstructions of mesh videos of the test dataset in parallel...")

        # parallel jobs
        max_parallel_jobs = mp.cpu_count()
        parallel_jobs = []
        
        counter = 0
        rec_dir = self.log_dir / "reconstructions"
        for mesh_vids, times, _, _ in test_dataset: # for every batch of test mesh videos
            # reconstruct all the mesh videos in the batch
            _, _, rec_frames = self.call(mesh_vids.values, times.values, times.row_lengths(), training=False)
            # gather the frames of all mesh videos in the batch as well as their reconstructions and timesteps
            mesh_frames = mesh_vids.values.numpy()
            rec_frames = rec_frames.numpy()
            frame_times = times.values.numpy()
            
            row_lengths = times.row_lengths().numpy()
            row_limits = np.cumsum(row_lengths)

            i = 0
            for k, j in enumerate(row_limits): # k'th video in the batch
                if counter == limit:
                    break
                mesh_vid = mesh_frames[i:j]
                rec_vid = rec_frames[i:j]
                times_vid = frame_times[i:j]
                vid_duration = times_vid[-1]

                # make parallel jobs
                vid_name = f"ReconstructedMesh{str(counter).zfill(3)}"
                vid_out_dir = rec_dir / "videos" / f"E{str(epoch).zfill(3)}" / f"S{str(epoch_step).zfill(3)}"
                vtps_out_dir = rec_dir / "vtps" / f"E{str(epoch).zfill(3)}" / f"S{str(epoch_step).zfill(3)}" / vid_name
                
                # save_mesh_vid(rec_vid, self.data_handler, vid_duration, vid_out_dir, vid_name, True, True, vtps_out_dir)
                args=(rec_vid, self.data_handler, vid_duration, vid_out_dir, vid_name, True, save_vtps, vtps_out_dir)
                p = mp.Process(target=save_mesh_vid, args=args)
                parallel_jobs.append(p)
                p.start()

                if save_original:
                    vid_name = f"OriginalMesh{str(counter).zfill(3)}"
                    vid_out_dir = rec_dir / "videos" / "OriginalMeshes"
                    vtps_out_dir = rec_dir / "vtps" / "OriginalMeshes" / vid_name
                    
                    #save_mesh_vid(mesh_vid, self.data_handler, vid_duration, vid_out_dir, vid_name, True, True, vtps_out_dir)
                    args=(mesh_vid, self.data_handler, vid_duration, vid_out_dir, vid_name, True, save_vtps, vtps_out_dir)
                    p = mp.Process(target=save_mesh_vid, args=args)
                    parallel_jobs.append(p)
                    p.start()

                i = j
                counter += 1
            
            if counter == limit:
                break
        
        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 -t1)
        logger.info(f"Done in {h}: {m}: {s}")
        logger.info("Finishing videos generation in parallel")
        all_parallel_jobs.extend(parallel_jobs)

# -------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------- Plots generation -------------------------------------------
    def generate_freq_phase_ef_plots(self, dataset, set_name, epoch, epoch_step, compute_ef_biplane=False, save_plot_data=False, mod_pred_phases=False, verbose=False):
        logger.info(f"{set_name} plot generation...")
        t1 = time.time()

        plots_dir = self.log_dir / 'plots' / set_name
        plots_dir.mkdir(parents=True, exist_ok=True)

        true_freq_index = CONRAD_DATA_PARAMS.index("frequency")
        true_phase_index = CONRAD_DATA_PARAMS.index("cycle_shift")
        true_efs_vol_index = CONRAD_DATA_PARAMS.index("EF_Vol")
        true_efs_biplane_index = CONRAD_DATA_PARAMS.index("EF_Biplane")
        
        # true and pred values for plotting
        true_efs_vol = []
        pred_efs_vol = []
        true_efs_biplane = []
        pred_efs_biplane = []
        true_freqs = []
        pred_freqs = []
        true_phases = []
        pred_phases = []

        # parallel jobs to compute video efs, 1 process per video ---> faster processing
        manager = mp.Manager()
        return_dict = manager.dict()
        parallel_jobs = []
        max_parallel_jobs = 16

        # mesh data_handler
        mesh_data_handler = self.data_handler

        if compute_ef_biplane:
            # ef disks using 4CH and 2CH planes and EDF for slicing both ED and ES frames
            ef_disks_params = {}
            ef_disks_params["view_pair"] = ("4CH", "2CH")
            ef_disks_params["slicing_ref_frame"] = "EDF"
        else:
            ef_disks_params = None

        # count nb of videos processed
        counter = 0

        for b, (mesh_vids, times, all_volumes, true_params) in enumerate(dataset): # for every batch of the dataset
            pred_params, _, reconstructions = self.call(mesh_vids.values, times.values, times.row_lengths(), training=False) # run model end-to-end

            # convert to numpy
            reconstructions = reconstructions.numpy()
            true_params = true_params.to_tensor(default_value=0.0).numpy()
            pred_params = pred_params.numpy()

            # get row lengths and row limits
            row_lengths = times.row_lengths().numpy()
            row_limits = np.cumsum(row_lengths)

            if verbose:
                logger.info(f"Batch {b}")

            i = 0
            for k, j in enumerate(row_limits): # k'th video in the batch
                mesh_vid_rec = reconstructions[i:j] # the video's reconstruction

                # save true params
                true_freqs.append(true_params[k, true_freq_index])
                true_phases.append(true_params[k, true_phase_index])
                true_efs_vol.append(true_params[k, true_efs_vol_index])
                true_efs_biplane.append(true_params[k, true_efs_biplane_index])

                # save predicted freq and phase
                pred_freqs.append(pred_params[k, 0])
                pred_phases.append(pred_params[k, 1])

                # compute pred ef_vol and ef_biplane
                args=(mesh_vid_rec, mesh_data_handler, ef_disks_params, True, return_dict, counter)
                p = mp.Process(target=compute_ef_mesh_vid, args=args)
                parallel_jobs.append(p)
                p.start()
            
                i = j
                counter += 1
        
            # wait for some parallel jobs to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, max_parallel_jobs)
        
        # wait for all parallel jobs to complete
        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

        if verbose:
            logger.info("Done computing, plotting data....")

        # gather the computed pred ef_vol and ef_biplane
        for i in range(counter):
            ef_vol, ef_biplane = return_dict[i]
            pred_efs_vol.append(ef_vol)
            pred_efs_biplane.append(ef_biplane)
        
        if mod_pred_phases:
            modded_pred_phases = []
            for phase in pred_phases:
                while phase < 0:
                    phase += 1
                while phase >= 1:
                    phase -= 1
                modded_pred_phases.append(phase)

        # freq scatter plot
        freq_plots_dir = plots_dir / "freqs"
        filename = f"FreqPlot_E{str(epoch).zfill(3)}_S{str(epoch_step).zfill(3)}.png"
        x_data, y_data = true_freqs, pred_freqs
        x_label, y_label = "True Freq:", "Pred Freq:"
        title = f"Frequencies scatter plot E{epoch}, S{epoch_step}"
        utils.scatter_plot(freq_plots_dir, filename, x_data, y_data, x_label, y_label, title)

        # phase shift scatter plot
        phase_plots_dir = plots_dir / "phases"
        filename = f"PhasePlot_E{str(epoch).zfill(3)}_S{str(epoch_step).zfill(3)}.png"
        x_data, y_data = true_phases, pred_phases
        x_label, y_label = "True Phase:", "Pred Phase:"
        title = f"Phase shift scatter plot E{epoch}, S{epoch_step}"
        utils.scatter_plot(phase_plots_dir, filename, x_data, y_data, x_label, y_label, title, diff=True)

        if mod_pred_phases:
            # phase shift scatter plot
            modded_phase_plots_dir = plots_dir / "phases_mod"
            filename = f"PhasePlotMod_E{str(epoch).zfill(3)}_S{str(epoch_step).zfill(3)}.png"
            x_data, y_data = true_phases, modded_pred_phases
            x_label, y_label = "True Phase:", "Pred Phase (mod 1):"
            title = f"Phase shift scatter plot E{epoch}, S{epoch_step}"
            utils.scatter_plot(modded_phase_plots_dir, filename, x_data, y_data, x_label, y_label, title, diff=True)

        # ef_vol scatter plot
        ef_vol_plots_dir = plots_dir / "ef_vol"
        filename = f"EF_Vol_Plot_E{str(epoch).zfill(3)}_S{str(epoch_step).zfill(3)}.png"
        x_data, y_data = true_efs_vol, pred_efs_vol
        x_label, y_label = "True EF Vol:", "Pred EF Vol:"
        title = f"EF Vol scatter plot E{epoch}, S{epoch_step}"
        utils.scatter_plot(ef_vol_plots_dir, filename, x_data, y_data, x_label, y_label, title)

        if compute_ef_biplane:
            # ef_biplane scatter plot
            ef_biplane_plots_dir = plots_dir / "ef_biplane"
            filename = f"EF_Biplane_Plot_E{str(epoch).zfill(3)}_S{str(epoch_step).zfill(3)}.png"
            x_data, y_data = true_efs_biplane, pred_efs_biplane
            x_label, y_label = "True EF Biplane:", "Pred EF Biplane:"
            title = f"EF Biplane scatter plot E{epoch}, S{epoch_step}"
            utils.scatter_plot(ef_biplane_plots_dir, filename, x_data, y_data, x_label, y_label, title)

        if save_plot_data:
            # save freqs
            filename = f"TruePredFreqData_E{str(epoch).zfill(3)}_S{str(epoch_step).zfill(3)}.npz"
            np.savez_compressed(freq_plots_dir / filename, true_freqs=true_freqs, pred_freqs=pred_freqs)
            # save phase shifts
            filename = f"TruePredPhaseData_E{str(epoch).zfill(3)}_S{str(epoch_step).zfill(3)}.npz"
            np.savez_compressed(phase_plots_dir / filename, true_phases=true_phases, pred_phases=pred_phases)
            if mod_pred_phases:
                # save phase shifts modded
                filename = f"TruePredPhaseDataMod_E{str(epoch).zfill(3)}_S{str(epoch_step).zfill(3)}.npz"
                np.savez_compressed(modded_phase_plots_dir / filename, true_phases=true_phases, modded_pred_phases=modded_pred_phases)
            # save efs_vol
            filename = f"TruePredEFVolData_E{str(epoch).zfill(3)}_S{str(epoch_step).zfill(3)}.npz"
            np.savez_compressed(ef_vol_plots_dir / filename, true_efs_vol=true_efs_vol, pred_efs_vol=pred_efs_vol)
            if compute_ef_biplane:
                # save efs_biplane
                filename = f"TruePredEFBiplaneData_E{str(epoch).zfill(3)}_S{str(epoch_step).zfill(3)}.npz"
                np.savez_compressed(ef_biplane_plots_dir / filename, true_efs_biplane=true_efs_biplane, pred_efs_biplane=pred_efs_biplane)

        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
        logger.info(f"{set_name} plots generated in {h}:{m}:{s}\n")

# -------------------------------------------------------- Experiments ----------------------------------------------
    def plot_volumes_over_time(self, dataset, set_name, epoch, epoch_step, limit=10, save_plot_data=False, verbose=False):
        components = self.data_handler.components
        n_components = len(components)
        if limit > 0:
            logger.info(f"Plotting volumes over time on {limit} videos of {set_name} dataset...")
        else:
            logger.info(f"Plotting volumes over time on all videos of {set_name} dataset...")

        logger.info(f"Components: {components}")

        t1 = time.time()

        # parallel jobs to compute video volumes, 1 process per video ---> faster processing
        manager = mp.Manager()
        return_dict = manager.dict()
        parallel_jobs = []
        max_parallel_jobs = 16

        # mesh data_handler
        mesh_data_handler = self.data_handler
        dataset_scale = mesh_data_handler.dataset_scale
        dataset_min = mesh_data_handler.dataset_min
        ref_poly = mesh_data_handler.reference_poly

        all_true_volumes = {}
        all_times = {}
        counter = 0
        for b, (mesh_vids, times, vids_volumes, true_params) in enumerate(dataset): # for every batch of the dataset
            _, _, reconstructions = self.call(mesh_vids.values, times.values, times.row_lengths(), training=False) # run model end-to-end

            # convert to numpy
            reconstructions = reconstructions.numpy()
            reconstructions = (reconstructions * dataset_scale) + dataset_min

            if verbose:
                logger.info(f"Batch {b}")

            # split true volumes per video and per component
            volumes_row_lengths = vids_volumes.row_lengths().numpy()
            volumes_row_limits = np.cumsum(volumes_row_lengths)
            per_vid_volumes = np.split(vids_volumes.values.numpy(), volumes_row_limits)[:-1] # split volumes to get list of volumes for each video, ignore last empty list
            for j, vid_volumes in enumerate(per_vid_volumes):
                vid_volumes_split = np.split(vid_volumes, n_components)
                vid_comp_volumes = {}
                for comp_name, comp_vols in zip(components, vid_volumes_split):
                    vid_comp_volumes[comp_name] = comp_vols
                all_true_volumes[counter+j] = vid_comp_volumes

            # get row lengths and row limits and split times
            row_lengths = times.row_lengths().numpy()
            row_limits = np.cumsum(row_lengths)
            times_values = times.values.numpy()

            i = 0
            for k, j in enumerate(row_limits): # k'th video in the batch
                mesh_vid_rec = reconstructions[i:j] # the video's reconstruction
                vid_times = times_values[i:j]

                all_times[counter] = vid_times

                # compute pred volumes
                args=(mesh_vid_rec, ref_poly, components, True, return_dict, counter)
                p = mp.Process(target=compute_volumes_feats, args=args)
                parallel_jobs.append(p)
                p.start()
            
                i = j
                counter += 1
                if counter == limit:
                    break
            
            if counter == limit:
                break

            # wait for some parallel jobs to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, max_parallel_jobs)
        
        # wait for all parallel jobs to complete
        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

        if verbose:
            logger.info("Done computing, plotting data....")
        
        for i in range(counter): # for each video
            vid_true_volumes = all_true_volumes[i]
            vid_pred_volumes = return_dict[i]
            vid_times = all_times[i]
            for component in components: # plot volume over time for each component
                true_volumes_c = vid_true_volumes[component]
                pred_volumes_c = vid_pred_volumes[component]
                assert len(true_volumes_c) == len(pred_volumes_c)
                assert len(pred_volumes_c) == len(vid_times)
                
                plot_dir = self.log_dir / 'plots' / set_name / "volumes_over_time" / component
                if epoch is not None and epoch_step is not None:
                    plot_dir = plot_dir / f"E{str(epoch).zfill(3)}" / f"S{str(epoch_step).zfill(3)}"
                filename = f"VolumeOverTime_{i}.png"
                y_data_list = [true_volumes_c, pred_volumes_c]
                x_data_list = [vid_times, vid_times]
                labels = ["True Volume", "Pred Volume"]
                x_label = "time"
                y_label = "Volume"
                title = f"Volume over time of True and Predicted Mesh Video {i}"
                utils.plot_x_y_data(plot_dir, filename, y_data_list, x_data_list, labels, x_label, y_label, title)

                if save_plot_data:
                    filename = f"VolumesOverTime_{j}.npz"
                    np.savez_compressed(plot_dir / filename, true_volumes=true_volumes_c, t_true=vid_times, pred_volumes=pred_volumes_c, t_pred=vid_times)

        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
        logger.info(f"Volumes over time on {set_name} set generated in {h}:{m}:{s}\n")
    
    def interpolate_2_samples_latent(self, dataset, set_name, mesh_1_id, mesh_2_id, nb_intermediate=20, timestep=0, save_render_mesh=False):
        logger.info(f"Linearly interpolating between 2 meshes, mesh {mesh_1_id} and mesh {mesh_2_id} of {set_name} dataset...")
        ref_poly = self.data_handler.reference_poly
        batch_size = utils.get_mesh_dataset_batch_size(dataset)
        
        for feats, times, _, _ in dataset.skip(int(mesh_1_id // batch_size)): # encode the batch where mesh 1 is
            params = self.encode(feats.values, times.values, times.row_lengths(), training=False)
            mesh_1_params = params[int(mesh_1_id % batch_size)]
            break
        for feats, times, _, _ in dataset.skip(int(mesh_2_id // batch_size)): # encode the batch where mesh 2 is
            params = self.encode(feats.values, times.values, times.row_lengths(), training=False)
            mesh_2_params = params[int(mesh_2_id % batch_size)]
            break
        # fix freq and phase to be those of mesh 1
        mesh_2_params = tf.concat([mesh_1_params[0:2], mesh_2_params[2:]], axis=0)
        
        # nb_intermediate is nb of intermediate params vectors to generate which are linear interpolation between mesh_1 and mesh_2 params
        # generate list of mesh 1 params + latent params of the interpolations + mesh 2 params
        alphas = [i/(nb_intermediate+1) for i in range(nb_intermediate+2)] # first and last alpha are for mesh_1 and mesh_2 respectively
        interpolations_params = [mesh_1_params + alpha * (mesh_2_params - mesh_1_params) for alpha in alphas]
        interpolations_params = tf.stack(interpolations_params)
        
        # decode the latent params
        nb_shapes = interpolations_params.shape[0]
        times_values = tf.zeros(nb_shapes) + timestep # timestep of the decoded meshes (has to be floats)
        row_lengths = tf.ones(nb_shapes, tf.int32) # 1 timestep => 1 mesh per data sample => row_lengths = 1 (has to be ints)
        feats = self.decode(interpolations_params, times_values, row_lengths, training=False).numpy()
        
        # save meshes as vtps
        output_dir = self.log_dir / 'interpolations' / f'latent_interpolation_mesh_{str(mesh_1_id).zfill(3)}_to_mesh_{str(mesh_2_id).zfill(3)}'
        vtps_out_dir = output_dir / "vtps"
        
        logger.info(f"Saving interpolations vtps under {vtps_out_dir}")
        t1 = time.time()
        for i, mesh in enumerate(feats): # for each mesh in the mesh seq
            filename = f"ReconstructedMesh_{str(i).zfill(3)}"
            overwrite_vtkpoly(ref_poly, points=mesh, save=True, output_dir=vtps_out_dir, name=filename)
        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
        logger.info("Done saving interpolations vtps in {}:{}:{}\n".format(h, m, s))
        
        # save rendering of the mesh video
        if save_render_mesh:
            logger.info("Rendering interpolation meshes and saving as video")
            t1 = time.time()

            mesh_data_handler = self.data_handler
            vid_duration = (nb_intermediate + 2.0) / 2 # video duration so that each of the "nb_intermediate + 2" frames takes 1/2 second
            vid_out_dir = output_dir / "videos"
            vid_name = f'mesh_{str(mesh_1_id).zfill(3)}_to_mesh_{str(mesh_2_id).zfill(3)}'
            save_mesh_vid(feats, mesh_data_handler, vid_duration, vid_out_dir, vid_name)
            
            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info("Done generating interpolations pngs and videos in {}:{}:{}\n".format(h, m, s))

    def interpolate_latent_dimensions(self, dataset, set_name, mesh_1_id, mesh_2_id, nb_intermediate=20, timestep=0):
        logger.info(f"Linearly interpolating between 2 mesh latents dimentsions, mesh {mesh_1_id} and mesh {mesh_2_id} of {set_name} dataset...")
        ref_poly = self.data_handler.reference_poly
        batch_size = utils.get_mesh_dataset_batch_size(dataset)

        dataset_scale = self.data_handler.dataset_scale
        dataset_min = self.data_handler.dataset_min
        
        for feats, times, _, _ in dataset.skip(int(mesh_1_id // batch_size)): # encode the batch where mesh 1 is
            params = self.encode(feats.values, times.values, times.row_lengths(), training=False)
            mesh_1_params = params[int(mesh_1_id % batch_size)]
            break
        for feats, times, _, _ in dataset.skip(int(mesh_2_id // batch_size)): # encode the batch where mesh 2 is
            params = self.encode(feats.values, times.values, times.row_lengths(), training=False)
            mesh_2_params = params[int(mesh_2_id % batch_size)]
            break
        
        # start at phase 0
        mesh_1_params = tf.concat([mesh_1_params[0:1], tf.zeros(1), mesh_1_params[2:]], axis=0)

        alphas = [i/(nb_intermediate+1) for i in range(nb_intermediate+2)] # first and last alpha are for mesh_1 and mesh_2 respectively
        params_shape = mesh_2_params.shape[0]
        
        for i in range(2, params_shape): # for each shape param
            # create a vector that has the same params as mesh 1 except in one dimension of the shape params vectors
            # this would be the target vector to interpolate to it from mesh 1 params thus effectively moving only
            # along that 1 dimension
            single_param = tf.concat([mesh_1_params[0:i], mesh_2_params[i:i+1], mesh_1_params[i+1:]], axis=0)
            interpolations_params = [mesh_1_params + alpha * (single_param - mesh_1_params) for alpha in alphas]
            interpolations_params = tf.stack(interpolations_params)
        
            # decode the latent params
            nb_shapes = interpolations_params.shape[0]
            times_values = tf.zeros(nb_shapes) + timestep # timestep of the decoded meshes (has to be floats)
            row_lengths = tf.ones(nb_shapes, tf.int32) # 1 timestep => 1 mesh per data sample => row_lengths = 1 (has to be ints)
            feats = self.decode(interpolations_params, times_values, row_lengths, training=False)
            feats = (feats * dataset_scale) + dataset_min # denormalize

            mesh_1_rec = feats[0]
            diffs = feats - mesh_1_rec # difference between the interpolated reconstruction and the starting mesh (mesh 1) reconstruction
            distances = tf.reduce_sum(tf.square(diffs), axis=-1)

            # save meshes as vtps
            output_dir = self.log_dir / 'interpolations' / f'latent_dims_interpolation_mesh_{str(mesh_1_id).zfill(3)}_to_mesh_{str(mesh_2_id).zfill(3)}'
            vtps_out_dir = output_dir / "vtps"
            
            logger.info(f"Saving interpolations vtps under {vtps_out_dir}")
            t1 = time.time()

            for j, (mesh, dist) in enumerate(zip(feats, distances)): # for each mesh and it's diff (from mesh 1 rec)
                mesh, dist = mesh.numpy(), dist.numpy() # convert to numpy
                filename = f"ReconstructedMeshDim{str(i).zfill(3)}_{str(j).zfill(3)}"
                overwrite_vtkpoly(ref_poly, points=mesh, data_array_and_name=(dist, "dist"), save=True, output_dir=vtps_out_dir, name=filename)

            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done saving dimension {i} interpolations vtps in {h}:{m}:{s}\n")
    
    def interpolate_mesh_dims(self, mesh_feats, mesh_times, mesh_id, latents, nb_intermediate=20, timestep=0):
        logger.info(f"Linearly interpolating a mesh {mesh_id} latent dimensions...")
        ref_poly = self.data_handler.reference_poly
        dataset_scale = self.data_handler.dataset_scale
        dataset_min = self.data_handler.dataset_min

        row_lengths = [len(mesh_times)]
        mesh_params = self.encode(mesh_feats, mesh_times, row_lengths, training=False)
        logger.info(f"Encoded Mesh params: {mesh_params.numpy()}")
        # reconstruct original mesh at the same timestep as the interpolations
        times_values = mesh_times + timestep
        row_lengths = [len(mesh_times)]
        mesh_rec = self.decode(mesh_params, times_values, row_lengths, training=False)
        original_mesh_ef_vol, _ = compute_ef_mesh_vid(mesh_rec, self.data_handler)
        mesh_rec = (mesh_rec * dataset_scale) + dataset_min
        

        mesh_params = mesh_params[0]
        alphas = [i/nb_intermediate for i in range(nb_intermediate+1)] # generate "nb_intermediate" alphas, one per intermediate shape
        params_shape = mesh_params.shape[0]
        
        max_distances = []
        all_dims_vals = []
        all_efs = []
        save_dir = self.log_dir / 'interpolations' / "1_mesh_dims_interpolation"
        parallel_jobs = []
        for i in range(2, params_shape): # for each shape param "i"
            # if i != 8:
            #     continue
            # get the min value and max value of shape param "i" from the latents dataset then get the interpolation range
            min_val_i = latents[:, i].min()
            max_val_i = latents[:, i].max()
            mid_val_i = (min_val_i + max_val_i) / 2
            min_to_mid_dist = mid_val_i - min_val_i
            max_to_mid_dist = max_val_i - mid_val_i
            mult_fac = 3 # multiplicative factor for distance from mid
            # interpolation range
            min_interpol_val_i = tf.constant([float(mid_val_i - mult_fac * min_to_mid_dist)])
            max_interpol_val_i = tf.constant([float(mid_val_i + mult_fac * max_to_mid_dist)])
            logger.info(f"Dimension {i}: Interpolating between {min_interpol_val_i.numpy()} and {max_interpol_val_i.numpy()}")
            # create 2 vectors: the start and end vectors
            # the 2 vectors have same shape params as the input mesh except for the i'th dim
            # the start vector has value "min_interpol_val_i" for the i'th dim and the end vector has value "max_interpol_val_i" for the i'th dim
            start_mesh_params = tf.concat([mesh_params[0:i], min_interpol_val_i, mesh_params[i+1:]], axis=0)
            end_mesh_params = tf.concat([mesh_params[0:i], max_interpol_val_i, mesh_params[i+1:]], axis=0)
            interpolations_params = [start_mesh_params + alpha * (end_mesh_params - start_mesh_params) for alpha in alphas]
            dim_vals = []
            efs = []
            for params in interpolations_params:
                dim_val = params[i]
                dim_vals.append(dim_val)
                row_lengths = [len(mesh_times)]
                rec_feats = self.decode([params], mesh_times, row_lengths, training=False).numpy()
                ef_vol, _ = compute_ef_mesh_vid(rec_feats, self.data_handler)
                efs.append(ef_vol)

                vid_out_dir = save_dir / "reconstructions" / f"latent_dim_{i}" / f'mesh_{mesh_id}'
                vid_name = f"dim_value_{dim_val:.3f}"
                args=(rec_feats, self.data_handler, mesh_times[-1], vid_out_dir, vid_name)
                p = mp.Process(target=save_mesh_vid, args=args)
                parallel_jobs.append(p)
                p.start()
            
            all_dims_vals.append(dim_vals)
            all_efs.append(efs)
            continue
            interpolations_params = tf.stack(interpolations_params)
        
            # decode the latent params
            nb_shapes = interpolations_params.shape[0]
            times_values = tf.zeros(nb_shapes) + timestep # timestep of the decoded meshes (has to be floats)
            row_lengths = tf.ones(nb_shapes, tf.int32) # 1 timestep => 1 mesh per data sample => row_lengths = 1 (has to be ints)
            feats = self.decode(interpolations_params, times_values, row_lengths, training=False)
            feats = (feats * dataset_scale) + dataset_min # denormalize

            diffs = feats - mesh_rec # difference between the interpolated reconstruction and original mesh reconstruction
            distances = tf.reduce_sum(tf.square(diffs), axis=-1)

            # save meshes as vtps
            if isinstance(mesh_id, str): # mesh identifier is string
                rec_dir = self.log_dir / 'interpolations' / f'1_mesh_dims_interpolation_{mesh_id}'
            else: # mesh identifier is an int
                rec_dir = self.log_dir / 'interpolations' / f'1_mesh_dims_interpolation_{str(mesh_id).zfill(3)}'
            vtps_out_dir = rec_dir / "vtps"
            
            logger.info(f"Saving interpolations vtps under {vtps_out_dir}")
            t1 = time.time()

            max_dist_i = -1
            for j, (mesh, dist) in enumerate(zip(feats, distances)): # for each mesh and it's diff (from mesh 1 rec)
                mesh, dist = mesh.numpy(), dist.numpy() # convert to numpy
                max_dist_i = max(max_dist_i, np.max(dist))
                filename = f"ReconstructedMeshDim{str(i).zfill(3)}_{str(j).zfill(3)}"
                overwrite_vtkpoly(ref_poly, points=mesh, data_array_and_name=(dist, "dist"), save=True, output_dir=vtps_out_dir, name=filename)
            
            max_distances.append(max_dist_i)
            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done saving dimension {i} interpolations vtps in {h}:{m}:{s}\n")
        plot_dir = save_dir /"plots"
        filename = f'mesh_{mesh_id}.png'
        y_data_list = all_efs
        x_data_list = all_dims_vals
        labels = [f"EF_Vol_dim_{i+2}" for i in range(len(all_efs))]
        x_label = "Dim Value"
        y_label = "EF Vol"
        title = f"Change of EF Vol by changing all dims values"
        h_lines_y_values = [original_mesh_ef_vol]
        utils.plot_x_y_data(plot_dir, filename, y_data_list, x_data_list, labels, x_label, y_label, title, h_lines_y_values)

        parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)
        logger.info(f"Done saving videos in parallel")
        return
        max_distances.sort()
        logger.info(f"Done saving interpolations on all dims, max_distance: {max_distances[-1]}")
        logger.info(f"Sorted Distances (floor): {np.floor(max_distances)}")
    
    def denoise_mesh(self, dataset, set_name, mesh_id, noise_mu=0.0, noise_sigma=0.01):
        logger.info(f"Adding noise to mesh {mesh_id} of {set_name} dataset and getting model output")
        ref_poly = self.data_handler.reference_poly
        batch_size = utils.get_mesh_dataset_batch_size(dataset)
        
        all_original_mesh_feats = []
        all_noisy_mesh_feats = []
        all_decoded_feats = []
        all_decoded_noisy_feats = []

        all_mesh_times = []
        all_mesh_row_lengths = []

        for feats, times, _, _ in dataset.skip(int(mesh_id // batch_size)): # encode the batch where the mesh is
            # extract the mesh feats and times
            mesh_id_in_batch = int(mesh_id % batch_size)
            row_lengths = times.row_lengths()
            row_limits = tf.cumsum(row_lengths)
            if mesh_id_in_batch == 0: # start of mesh feats and times
                i = 0
            else:
                i = row_limits[mesh_id_in_batch-1]
            j = row_limits[mesh_id_in_batch] # end of mesh feats and times
            mesh_feats = feats.values[i:j]
            mesh_times = times.values[i:j]
            
            # add noise to feats
            noise_shape = mesh_feats.shape
            noise = np.reshape(np.random.normal(noise_mu, noise_sigma, np.prod(noise_shape)), noise_shape)
            noisy_mesh_feats = mesh_feats + noise
            
            # encode mesh video
            mesh_row_lengths = [mesh_feats.shape[0]] # we are encoding only 1 mesh video
            params = self.encode(mesh_feats, mesh_times, mesh_row_lengths, training=False)
            noisy_params = self.encode(noisy_mesh_feats, mesh_times, mesh_row_lengths, training=False)

            decoded_feats = self.decode(params, mesh_times, mesh_row_lengths, training=False)
            decoded_noisy_feats = self.decode(noisy_params, mesh_times, mesh_row_lengths, training=False)

            # save values
            all_original_mesh_feats.append(mesh_feats.numpy())
            all_noisy_mesh_feats.append(noisy_mesh_feats.numpy())
            all_decoded_feats.append(decoded_feats.numpy())
            all_decoded_noisy_feats.append(decoded_noisy_feats.numpy())

            all_mesh_times.append(mesh_times.numpy())
            all_mesh_row_lengths.append(mesh_row_lengths)
            break
        
        all_original_mesh_feats = np.concatenate(all_original_mesh_feats, axis=0)
        all_noisy_mesh_feats = np.concatenate(all_noisy_mesh_feats, axis=0)
        all_mesh_times = np.concatenate(all_mesh_times, axis=0)
        all_mesh_row_lengths = np.concatenate(all_mesh_row_lengths, axis=0)
        all_decoded_feats = np.concatenate(all_decoded_feats, axis=0)
        all_decoded_noisy_feats = np.concatenate(all_decoded_noisy_feats, axis=0)

        all_mesh_row_limits = np.cumsum(all_mesh_row_lengths)

        feats_to_save = {
                         "OriginalMesh": all_original_mesh_feats,
                         "OriginalNoisyMesh": all_noisy_mesh_feats,
                         "ReconstructedMesh": all_decoded_feats,
                         "ReconstructedNoisyMesh": all_decoded_noisy_feats
                        }
        vtp_types = {
                     "OriginalMesh": "original vtps",
                     "OriginalNoisyMesh": "original noisy vtps",
                     "ReconstructedMesh": "reconstructed vtps",
                     "ReconstructedNoisyMesh": "reconstructed noisy vtps"
                    }

        for prefix in feats_to_save:
            all_feats = feats_to_save[prefix]
            vtp_type = vtp_types[prefix]

            t1 = time.time()
            i = 0
            for counter, j in enumerate(all_mesh_row_limits):
                feats = all_feats[i:j]
                output_dir = self.log_dir / 'denoising' / set_name / f"mesh_{str(counter).zfill(3)}"
                logger.info(f"Saving {vtp_type} vtps under {output_dir}")
                
                for idx, mesh in enumerate(feats): # for each mesh in the mesh seq
                    filename = f"{prefix}_{str(idx).zfill(3)}"
                    overwrite_vtkpoly(ref_poly, points=mesh, save=True, output_dir=output_dir, name=filename)
                
                i = j
            
            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done saving {vtp_type} in {h}:{m}:{s}\n")

# -------------------------------------------------------------------------------------------------------------------