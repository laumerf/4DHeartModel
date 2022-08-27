from itertools import chain
import logging, time, gc
import numpy as np
import tensorflow as tf
import yaml
from itertools import groupby
import multiprocessing as mp

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses

from source.shape_model_utils import overwrite_vtkpoly
from source.shape_model_utils import construct_triangles
import source.utils as utils
from source.constants import ROOT_LOGGER_STR
from source.laplacian_utils import get_trained_GMM

import bz2
import _pickle as cPickle
import tensorflow_probability as tfp

tfd = tfp.distributions

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

gif_processes = []


# ---------------------------- Not used ------------------------------------
class CustomLayer(Layer):

    def __init__(self, name='trajectory', **kwargs):
        super(CustomLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        output = None

        return output
# --------------------------------------------------------------------------

# Encoder
class Encoder(Layer):

    def __init__(self, L, D, params, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        # save params in self
        self.params = params
        # save list of conv layers and dense layers of the encoder in self
        self.conv_params = params['conv_layers']
        self.dense_params = params['dense_layer']

        self.latent_dim = self.dense_params['units']

        # To simplify, below explanations are when only 1 data point (i,e 1 shape) is input to the network, the generalization
        # to batches of data follows automatically in tensorflow
        # D = list of "d_sampling = down_sampling" matrices, i,e let A be output of conv layer i (no batches here), 
        # then B = D[i] * A is the downsampled version of A. Here D is precomputed and can stay fixed or further tweaked
        # by making it a learnable matrix when self.params['trainable_down'] is True.
        # We turn each d_sampling matrix in D into a tf.Variable
        # save the new list D of the tf.Variables in self
        self.D = [tf.Variable(down.toarray(),
                              name=f"conv_d_sampling_mat_{i}",
                              trainable=self.params['trainable_down'],
                              dtype=tf.float32)
                  for i, down in enumerate(D)]

        # list of bias matrices, added after downsampling, so with prev example:
        # let O_c[i] be output of conv layer i, O_c[i] is "m x n" where "n = nb_output_channels_conv_i" and "m = D[i].shape[1]"
        # the output of the downsampling layer O_d[i] = D[i] * O_c[i] is l x n with "l = D[i].shape[0]"
        # thus each bias matrix D_b[i] below (initialized with zeros) is l x n and added to O_d[i]
        # again, the bias matrices can be made learnable by setting self.params['trainable_down'] to True
        self.D_b = [tf.Variable(
            tf.zeros([d.shape[0], p['channels']]),
            trainable=params['trainable_down'],
            name=f"conv_d_sampling_bias_{i}",
            dtype=tf.float32)
            for i, (d, p) in enumerate(zip(D, self.conv_params))]

        # L = input matrices to each conv layer, aka enc_in. Convert these matrices to tensors
        # lap = laplacian matrix
        self.L = [tf.convert_to_tensor(lap.toarray(),
                                       dtype=tf.float32,
                                       name=f"laplacian_{i}")
                  for i, lap in enumerate(L)]

        # Graph-Convolution, save list of successive graph conv layers in list
        self.conv_layers = []
        for layer_params in self.conv_params:
            layer = utils.get_conv_layer(layer_params.get('name'))
            self.conv_layers.append(layer(**layer_params))
        self.fc = Dense(**self.dense_params)

        # When self.params['learn_down'] is True, we apply a learnable Fully Connected layer and use the output matrix of this
        # layer as the downsampling matrix, instead of using D[i].
        
        # So, we create a new list of matrices called "w_d_1" and a new list of bias matrices called "b_d" (here
        # w_d_1[i] and b_d[i] are the weights and biases of the FC layer i)
        # w_d_1, b_d, self.D_b and self.D_b all have the same length since each pair (w_d_1[i], b_d[i]) is used to compute
        # the new downsampling matrix new_D[i] that will replace D[i]
        # So new_D[i] has the same shape of D[i]
        # To show how new_D[i] is computed, let's take the previous example: given output O_c[i] (m x n) of conv layer i, 
        # we pass it through the FC layer to get new_D[i], so new_D[i] = softmax[ (O_c[i] * w_d_1[i])^T + b_d[i]) ]
        
        # new_D[i] and D[i] are "l x m" so b_d[i] also "l x m" (i,e same shape, see init of b_d[i] below)
        # thus since O_c[i] is m x n, and new_D[i] is "l x m" then w_d_1[i] must be "n x l" (see init of w_d_1[i] below)

        # Given this new_D[i], we compute the rest is the same as before, so we compute O_d[i] = new_D[i] * O_c[i]
        # then add D_b[i]
        if self.params['learn_down']:
            w_init = tf.random_normal_initializer() # create the weights initializer w_init

            # use w_init to initialize weights of the "learnable" downsampling matrices self.w_d_1
            self.w_d_1 = [tf.Variable(w_init(
                shape=(p['channels'], d.shape[0])),
                trainable=True,
                name=f"down_w1_{i}")
                for i, (d, p) in enumerate(zip(self.D, self.conv_params))]

            # use the w_init to initialize the "learnable" bias matrices
            self.b_d = [tf.Variable(w_init(
                shape=d.shape),
                trainable=True,
                name=f"down_b_{i}")
                for i, d in enumerate(self.D)]

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

        adj, features = inputs
        down = []
        for i in range(len(self.D)): # apply conv_d_sampling layers successively

            assert features.shape[1] == self.L[i].shape[0], \
                f"incompatible shapes between feature matrix {features.shape}" \
                f"and graph structure matrix {self.L[i].shape}"

            features = self.conv_layers[i]((features, self.L[i])) # apply conv_d_sampling layer on the features, also use 
                                                                  # matrix self.L[i] needed as input to the Graph Conv layer i
            # lap = tf.repeat(tf.expand_dims(self.L[i], axis=0),
            #                 tf.shape(features)[0], axis=0)  # match batch dim
            # d = tf.matmul(lap, features)

            # Down-sample
            if self.params['learn_down']:
                # d = (X*W_1)_T + b
                # FC layer with softmax activation:
                d = tf.transpose(tf.matmul(features, self.w_d_1[i]),
                                 perm=[0, 2, 1])
                d = tf.keras.activations.softmax(d + self.b_d[i]) # new downsampling matrix
                down.append(d) # append output d of downsampling + softmax activation
            else:
                d = tf.repeat(tf.expand_dims(self.D[i], axis=0),
                              tf.shape(features)[0], axis=0)  # match batch dim, i,e repeat self.D[i] "batch_size" times 
                                                              # along axis 0
                                                              # we use D[i] as downsampling matrix

            features = tf.matmul(d, features) + self.D_b[i] # apply downsampling matrix to features then add bias matrix

        # Final FC layer of the encoder
        features = tf.reshape(
            features,
            shape=[-1, self.conv_params[-1]['channels'] * self.D[-1].shape[0]])
        features = self.fc(features)

        # latent features
        return features, down

# Decoder
class Decoder(Layer):

    def __init__(self, L, U, params, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.params = params
        self.conv_params = params['conv_layers']
        self.dense_params = params['dense_layer']

        self.U = [tf.Variable(u.toarray(),
                              name=f'conv_u_sampling_mat_{i}',
                              trainable=self.params['trainable_up'],
                              dtype=tf.float32)
                  for i, u in enumerate(U)][::-1]

        b_dim = [self.conv_params[i - 1]['channels'] if i > 0 else
                 p['channels']
                 for i, p in enumerate(self.conv_params)]

        self.U_b = [tf.Variable(
            tf.zeros([u.shape[0], dim]),
            trainable=params['trainable_up'],
            name=f"conv_u_sampling_bias_{i}",
            dtype=tf.float32)
            for i, (u, dim) in enumerate(zip(self.U, b_dim))]

        self.L = list(map(
            lambda l:
            tf.convert_to_tensor(l.toarray(), dtype=tf.float32), L))

        self.latent_dim = self.conv_params[0]['channels']

        # first channel up is used twice
        self.dense_params['units'] = self.latent_dim * self.U[0].shape[-1]
        self.fc = Dense(**self.dense_params)

        # Graph-Convolution
        self.conv_layers = []
        for layer_params in self.conv_params:
            layer = utils.get_conv_layer(layer_params.get('name'))
            self.conv_layers.append(layer(**layer_params))

        if self.params['learn_up']:
            w_init = tf.random_normal_initializer()

            self.w_u_1 = [tf.Variable(w_init(
                shape=(dim, u.shape[0])),
                trainable=True,
                name=f"up_w1_{i}")
                for i, (u, dim) in enumerate(zip(self.U, b_dim))]

            self.b_u = [tf.Variable(w_init(
                shape=u.shape),
                trainable=True,
                name=f"up_b_{i}")
                for i, u in enumerate(self.U)]

    def call(self, inputs, training=False):

        latent_features = inputs
        features = self.fc(latent_features)
        features = tf.reshape(
            features,
            shape=(-1, self.U[0].shape[-1], self.latent_dim))

        up = []
        for i, conv_layer in enumerate(self.conv_layers):
            # u = (X*W_1)_T + b
            if self.params['learn_up']:
                u = tf.transpose(tf.matmul(features, self.w_u_1[i]),
                                 perm=[0, 2, 1])
                u = tf.keras.activations.softmax(u + self.b_u[i])
                up.append(u)
            else:
                u = tf.repeat(tf.expand_dims(self.U[i], axis=0),
                              tf.shape(features)[0], axis=0)  # match batch dim

            features = tf.matmul(u, features) + self.U_b[i]

            # apply graph Convolution
            assert features.shape[1] == self.L[i].shape[0], \
                f"incompatible shapes between feature matrix {features.shape}" \
                f"and graph structure matrix {self.L[i].shape}"
            features = conv_layer((features, self.L[i]))

        return features, up

# AutoEncoder
class MeshAutoencoderModel(Model):
    def __init__(self, data_handler, log_dir, training_params, model_params,
                 **kwargs):

        # maybe add **kwargs if properly defined
        super(MeshAutoencoderModel, self).__init__(name='autoencoder')

        self.data_handler = data_handler # save data handler CONRADData instance in self
        # save paths in self
        self.log_dir = log_dir # current experiment log dir (datetime dir)
        self.train_log_dir = self.log_dir / 'train'
        self.val_log_dir = self.log_dir / 'validation'
        self.rec_dir = self.log_dir / 'reconstruction'
        self.rec_png_dir = self.log_dir / 'reconstruction_png'
        self.rec_gif_dir = self.log_dir / 'reconstruction_gif'
        self.inter_dir = self.log_dir / 'interpolation'
        self.scale_path = self.log_dir / 'scales.yml'
        self.trained_model_path = self.log_dir / "trained_models" / "CoMA"

        
        self.input_dim_mesh = data_handler.input_dim_mesh # None for CONRADData
        # save training parameters from config file in self
        self.learning_rate = training_params['learning_rate']
        self.learning_rate_decay = training_params['decay_rate']
        self.max_epochs = training_params['num_epochs']
        self.num_steps_per_epochs = training_params['num_steps_per_epochs']
        self.training_mode = training_params['training_mode']
        self.up_down_reg = training_params['up_down_reg']
        self.alternate_epochs = training_params['alternate_epochs']
        self.up_down_fac = training_params['up_down_fac']
        self.up_down_decay = training_params['up_down_decay']
        self.which_loss = training_params['loss_str']
        self.patience = training_params['patience']
        self.test_frequency = training_params['test_frequency']
        self.n_val_samples = training_params['n_val_samples']
        self.n_train_samples = training_params['n_train_samples']

        self.epoch = 0 # current epoch
        # train metrics
        train_metrics = ['loss', 'reconstruction_error', 'regularisation',
                         'regularisation/x', 'regularisation/y', 'regularisation/z',
                         "up", "down"]
        val_metrics = ['reconstruction_error']

        # Prior
        self.use_laplacian_regularizer = False # by default not using Laplacian prior regularization
        # if training params in config file specify usage of Laplacian prior regularization, use it
        if 'compute_laplacian_prior' in training_params and \
                training_params['compute_laplacian_prior']:
            self.use_laplacian_regularizer = True
            self._laplacian_compute = True
            self._laplacian_k = training_params['laplacian_k']
            self._laplacian_gen_std_factor = training_params['laplacian_gen_std_factor']
            self._laplacian_num_gen = training_params['laplacian_num_gen']
        # otherwise if Laplacian path specified in training params of config file, use Laplacian prior regularization
        elif 'laplacian_path' in training_params and training_params['laplacian_path']:
            self.use_laplacian_regularizer = True
            self._laplacian_path = training_params['laplacian_path']
            self._laplacian_compute = False

        # Prepare encoder/decoder
        enc = [l['name'] for l in model_params['encoder']['conv_layers']] # list of consecutive conv layers of encoder in config file
                                                                          # each element of this list is a dict representing the params
                                                                          # of the conv layer (layer name, nb channels, activation func,
                                                                          # use bias ...)
        dec = [l['name'] for l in model_params['decoder']['conv_layers']] # likewise, list of consecutive conv layers of decoder in 
                                                                          # config file, each elem of the list being a dict

        _, _, D, U, enc_in, dec_in = data_handler.build_transform_matrices(
            enc, dec)
        self.downsampling_matrices = D
        self.upsampling_matrices = U

        self.encoder = Encoder(enc_in, D, params=model_params['encoder']) # init encoder 
        self.decoder = Decoder(dec_in, U, params=model_params['decoder']) # init decoder

        self.latent_space_dim = model_params['encoder']['dense_layer']['units'] # latent space dim

        # setup summary writers and metrics
        self._val_metrics = dict()
        self._train_metrics = dict()

        self._train_summary_writer = \
            tf.summary.create_file_writer(str(self.train_log_dir))
        self._val_summary_writer = \
            tf.summary.create_file_writer(str(self.val_log_dir))

        for metric in train_metrics:
            self._train_metrics[metric] = tf.keras.metrics.Mean(metric)

        for metric in val_metrics:
            self._val_metrics[metric] = tf.keras.metrics.Mean(metric)

        logger.info("Model initialized.")

    # ------------------------------------------------- Model Running ---------------------------------------------------
    def call(self, inputs, training=True, **kwargs):
        """
            Run Model End-to-End and return reconstructed shapes
        """
        adj, features = inputs
        latent_rep, down = self.encoder((adj, features), training=training)
        reconstructions, up = self.decoder(latent_rep, training=training)
        return reconstructions

    # -------------------------------------------------------------------------------------------------------------------

    # ---------------------------------------- Laplacian regularization functions ---------------------------------------
    def _init_laplacian_regularizer(self):
        if self._laplacian_compute:
            self._compute_laplacian_prior()
        else:
            self._load_laplacian_prior()

    def _compute_laplacian_prior(self):
        print('Computing Laplacian prior')
        # set params
        self._laplacian_dims = 3

        self.L, self.means, self.covs = get_trained_GMM(training_variance=self._laplacian_gen_std_factor,
                                                        n_samples=self._laplacian_num_gen,
                                                        n_components=self._laplacian_k,
                                                        data_handler=self.data_handler)

    def _load_laplacian_prior(self):
        print('Loading Laplacian prior')
        path = 'heart_mesh/' + self._laplacian_path

        with bz2.BZ2File(path, mode='rb') as file:
            L, gmms = cPickle.load(file)

        self.L = tf.constant(L, dtype=tf.dtypes.float32)

        # can adjust to different number of dimensions and components
        self._laplacian_dims = len(gmms)
        self._laplacian_k = gmms[0].means_.shape[0]

        means = []
        covs = []
        for dim in tf.range(self._laplacian_dims):
            dim_means = []
            dim_covs = []
            for k in tf.range(self._laplacian_k):
                dim_means.append(gmms[dim].means_[k])
                dim_covs.append(gmms[dim].covariances_[k])
            means.append(dim_means)
            covs.append(dim_covs)

        self.means = tf.constant(np.array(means))
        self.covs = tf.constant(np.array(covs))

    @tf.function
    def _regularisation(self, params, pred_mesh_features):
        """
        computes the regularization for the Laplacian prior. Requires it to be
        loaded (see _init_laplacian) otherwise returns 0
        :param params: not required
        :param pred_mesh_features: the mesh features (vertex locations)
        :return: loss per shape (in dimension of bath size)
        """

        # computes the probability given means, covariances (diagonal) and deltas
        def _get_prob(means, covs, deltas):
            return tfd.MultivariateNormalDiag(loc=means,
                                              scale_diag=tf.math.sqrt(covs)).log_prob(deltas)

        if self.use_laplacian_regularizer:
            assert self.L is not None and \
                   self.means is not None and \
                   self.covs is not None and \
                   self._laplacian_dims is not None and \
                   self._laplacian_k is not None

            # compute deltas as L @ v (where L is the Laplacian matrix, loaded earlier)
            deltas = tf.cast(tf.matmul(self.L, pred_mesh_features), tf.float64)

            # placeholder
            xloss = tf.constant(0.0, shape=[deltas.shape[0]], dtype=tf.dtypes.float64)
            yloss = tf.constant(0.0, shape=[deltas.shape[0]], dtype=tf.dtypes.float64)
            zloss = tf.constant(0.0, shape=[deltas.shape[0]], dtype=tf.dtypes.float64)

            # setup the conditions and dimension for the tf.loop
            k_condition = lambda dim, k, sum: tf.less(k, self._laplacian_k)

            # the inner loop operation over the k components
            def k_body(dim, k, sum):
                return dim, tf.add(k, 1), tf.add(sum, _get_prob(self.means[dim, k, :],
                                                                self.covs[dim, k, :],
                                                                deltas[:, :, dim]))

            # computes the loss using a tf.while_loop (condition, body, parameters)
            k = tf.constant(0)
            x = -tf.while_loop(k_condition, k_body, (0, k, xloss))[-1]
            k = tf.constant(0)
            y = -tf.while_loop(k_condition, k_body, (1, k, yloss))[-1]
            k = tf.constant(0)
            z = -tf.while_loop(k_condition, k_body, (2, k, zloss))[-1]

            stacked = tf.stack([x, y, z])
            # scales the loss according to the number of dimensions and components
            scaled = stacked * (1 / ((deltas.shape[1] ** 2) * self._laplacian_dims * self._laplacian_k))
            loss = tf.math.reduce_sum(scaled, axis=0)

            return tf.cast(loss, tf.float32), scaled
        else:
            return tf.constant(0, dtype=tf.float32), tf.constant(0, shape=[3, 1], dtype=tf.float32)

    # -------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------- Errors and Losses -------------------------------------------------
    def _reconstruction_error(self, true_mesh_features,
                              predicted_mesh_features):

        loss = 0
        if self.which_loss == 'l1' or self.which_loss == "l1l2" or self.which_loss == "l2l1":
            loss += tf.reduce_mean(
                losses.mean_absolute_error(
                    true_mesh_features, predicted_mesh_features))
        elif self.which_loss == 'l2' or self.which_loss == "l1l2" or self.which_loss == "l2l1":
            loss += tf.reduce_mean(
                losses.mean_squared_error(
                    true_mesh_features, predicted_mesh_features))
        else:
            raise NotImplementedError()

        return loss

    def _down_error(self, down):
        # init reconstruction loss and regularization loss to 0
        loss = tf.constant(0.0, shape=[self.data_handler.batch_size])
        reg_loss = tf.constant(0.0, shape=[self.data_handler.batch_size])
        # init l1 regularizer
        reg_l = tf.keras.regularizers.l1(l=self.up_down_reg)
        # only compute down_error/loss if params['learn_down'] is True
        if self.encoder.params['learn_down']:
            for i, d in enumerate(down): # for each (batch of) learned downsampling matrix new_D[i] at layer i
                true_d = tf.repeat(tf.expand_dims(self.encoder.D[i], axis=0),
                                   tf.shape(d)[0], axis=0) # get the true downsampling matrix D[i] (target to be learned)
                loss += tf.reduce_mean(
                    losses.mean_absolute_error(true_d, d), axis=1) # compute the mean absolute error to get an error for each
                                                                   # learned matrix in the batch, then reduce the errors of the
                                                                   # batch by computing the mean error
                                                                   # add result to the total reconstruction loss "loss"
                reg_loss += reg_l(d) # add regularization loss for parameters learned new_D[i] 
                                     # of the total regularization loss "reg_loss"
            loss = tf.truediv(loss, tf.constant(len(self.encoder.D),
                                                dtype=tf.float32)) # divide total reconstruction loss by 
                                                                   # nb of downsampling layers (i,e len(D))
            reg_loss = tf.truediv(reg_loss, tf.constant(len(self.encoder.D),
                                                        dtype=tf.float32)) # divide total regularization loss
                                                                           # by nb of downsampling layers (i,e len(D))

        return loss + reg_loss # return reconstruction loss + regularization loss

    def _up_error(self, up):
        # Like _down_error() but using upsampling matrices U instead of downsampling matrices D
        loss = tf.constant(0.0, shape=[self.data_handler.batch_size])
        reg_loss = tf.constant(0.0, shape=[self.data_handler.batch_size])
        reg_l = tf.keras.regularizers.l1(l=self.up_down_reg)
        if self.decoder.params['learn_up']:
            for i, u in enumerate(up):
                true_u = tf.repeat(tf.expand_dims(self.decoder.U[i], axis=0),
                                   tf.shape(u)[0], axis=0)

                loss += tf.reduce_mean(
                    losses.mean_absolute_error(true_u, u), axis=1)
                reg_loss += reg_l(u)
            loss = tf.truediv(loss, tf.constant(len(self.decoder.U),
                                                dtype=tf.float32))
            reg_loss = tf.truediv(reg_loss, tf.constant(len(self.decoder.U),
                                                        dtype=tf.float32))
        return loss + reg_loss

    def _loss(self, true_mesh_features, pred_mesh_features,
              up, down, params=None):

        # flatten sequences and calculate reconstruction error
        reconstruction_error = self._reconstruction_error(true_mesh_features,
                                                          pred_mesh_features)
        down_error = self._down_error(down)
        up_error = self._up_error(up)

        # Laplacian regularisation
        regularisation, dim_losses = self._regularisation(params, pred_mesh_features) # Laplacian regularization not used, 
                                                                                      # so regularization loss = 0
        # total loss
        # ud_f = self.up_down_fac * self.up_down_decay**self.epoch
        # if self.training_mode == 'joint':
        #     loss = reconstruction_error + regularisation + ud_f*up_down_error
        # else:
        loss = reconstruction_error + 0.1 * regularisation
        return loss, reconstruction_error, regularisation, dim_losses, down_error, up_error

    def _log_metrics(self, losses):
        """ 
            Saves loss, reconstruction error, dim_losses, down error and up error metrics
        """
        loss, rec_error, regularisation, dim_losses, down_error, up_error = losses
        self._train_metrics['loss'](loss)
        self._train_metrics['reconstruction_error'](rec_error)
        self._train_metrics['regularisation'](regularisation)
        for i, dim in enumerate(["x", "y", "z"]):
            self._train_metrics['regularisation/' + dim](dim_losses[i, :])
        self._train_metrics['down'](down_error)
        self._train_metrics['up'](up_error)

    # -------------------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------- Training -----------------------------------------------------
    @tf.function
    def _train_step(self, adj, mesh_features, optimizer):
        with tf.GradientTape() as tape:
            latents, down = self.encoder((adj, mesh_features), training=True)
            reconstructed_mesh, up = self.decoder(latents, training=True)

            # used for parameter regularization (TODO (fabian): implement)
            params = None

            ers = self._loss(mesh_features, reconstructed_mesh,
                             up, down, params)
            loss, rec_error, regularisation, dim_losses, down_error, up_error = ers

        # update model weights
        variables = self.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        self._log_metrics((loss, rec_error, regularisation, dim_losses,
                           down_error, up_error))

    def fit(self, **kwargs):
        global gif_processes

        train_dataset = self.data_handler.get_train_dataset(
            self.n_train_samples) # get train dataset of size self.n_train_samples as a tf dataset
        val_dataset = self.data_handler.get_val_dataset(self.n_val_samples) # get validation dataset of size self.n_val_samples
                                                                            # as a tf dataset

        if self.use_laplacian_regularizer: # init laplacian regularizer if used
            self._init_laplacian_regularizer()

        opt_weights = self.get_weights()
        opt_early_stopping_metric = np.inf
        optimizer0 = Adam(lr=self.learning_rate) # use Adam as optimizer

        count = 0
        step = 0
        self.epoch = 0
        # log number of weights only at first epoch
        n_trainable = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_weights])
        logger.info(f"Number of trainable weights: {n_trainable}")

        logger.info("Start training...")
        t1_epoch = time.time() # start epoch timer
        for adj, features in train_dataset: # for each (batch of) data shape from the dataset 
                                            # (adj = adjacency matrix, features = 3D points/vertices coordinates)
            # t1_step = time.time() # start step timer
            if self.training_mode == 'alternate':
                raise NotImplementedError  # Maybe later
            else: # perform a train step with 1 batch
                self._train_step(adj, features, optimizer0)
            # t2_step = time.time() # end step timer
            # h, m, s = utils.get_HH_MM_SS_from_sec(t2_step - t1_step)
            # logger.info("Train step {} done in {}:{}:{}".format(step, h, m, s))
            
            step += 1 # increment nb of steps

            # epoch finished
            if step == self.num_steps_per_epochs:
                t2_epoch = time.time() # end epoch timer
                h, m, s = utils.get_HH_MM_SS_from_sec(t2_epoch - t1_epoch)

                # reset step
                step = 0

                self.write_summaries(self._train_summary_writer,
                                     self._train_metrics, self.epoch, "Train")
                logger.info("Epoch done in {}:{}:{}\n".format(h, m, s))
                
                # Validation at the end of every epoch.
                logger.info("Computing validation error...")
                t1_val = time.time() # end epoch timer

                for adj_v, features_v in val_dataset:
                    self._evaluate(adj_v, features_v)
                
                t2_val = time.time() # end epoch timer
                h, m, s = utils.get_HH_MM_SS_from_sec(t2_val - t1_val)

                self.write_summaries(self._val_summary_writer,
                                     self._val_metrics, self.epoch,
                                     "Validation")
                logger.info("Validation done in {}:{}:{}".format(h, m, s))
                                
                # get reconstruction error on validation data
                val_rec_error = self._val_metrics['reconstruction_error'].result()
                # reset early stopping counter on improvement of validation error
                if val_rec_error < opt_early_stopping_metric:
                    logger.info("validation loss improved, saving model.")
                    opt_early_stopping_metric = val_rec_error
                    opt_weights = self.get_weights()

                    # save optimal model
                    self.save_me(name='best')

                    # reset counter
                    count = 0
                else:
                    count += 1
                    logger.info(f"Validation loss did not improve from "
                                f"{opt_early_stopping_metric}. "
                                f"Counter: {count}")
                    
                    # update learning rate
                    # after waiting half of the patience, decay learning rate ( by multiplying by the decay_rate) 
                    # to see if any improvement on validation loss can be seen
                    if count == self.patience - int(self.patience / 2):
                        lr = optimizer0.learning_rate * self.learning_rate_decay
                        logger.info(f"Reduced learning rate to {lr}")
                        optimizer0.learning_rate = lr

                    # patience reached, perform early stopping
                    if self.patience == count:
                        logger.info("Early stopping.")

                logger.info("\n")
                # save reconstructions on test data
                # on first epoch and whenever the validation loss improves (equivalent to count == 0)
                if count == 0 or self.epoch == 0:
                    logger.info("Reconstructing Test data...")
                    t1_test = time.time() # end epoch timer

                    self.reconstruct_test_data()

                    t2_test = time.time() # end epoch timer
                    h, m, s = utils.get_HH_MM_SS_from_sec(t2_test - t1_test)
                    logger.info("Test done in {}:{}:{}\n".format(h, m, s))

                # reset metrics at beginning of next epoch
                self.reset_metrics()
                
                # increment epoch nb
                self.epoch += 1

                # start epoch timer
                t1_epoch = time.time() 

                # check after every epoch for gif processes which terminated and close them
                current_gif_processes = []
                for process in gif_processes:
                    if not process.is_alive(): # if gif generating process terminated, close process to release ressources
                        process.close()
                    else:
                        current_gif_processes.append(process)
                gif_processes = current_gif_processes
                
            # stop training
            if count == self.patience or self.epoch == self.max_epochs:
                logger.info("Training stopped!")
                self.reconstruct_test_data()
                break

        # reset to optimal weights
        self.set_weights(opt_weights)
        
        # wait for running gif_processes
        for process in gif_processes:
            process.join()
            process.close()
        gif_processes = []

        logger.info(f"Finished model training at epoch {self.epoch}")

    # -------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------- Model Loading and Saving ---------------------------------------------

    def save_me(self, name=None):
        if name is None:
            name = "E_" + str(self.epoch)

        self.save_weights(str(self.trained_model_path) + f"_{name}")
        scales = self.data_handler.scales
        with self.scale_path.open(mode="w") as scale_file:
            yaml.dump(scales, scale_file)
        logger.info(f"Model saved to file {self.trained_model_path}")

    def load_me(self, epoch_n, name=None):

        if name is None:
            model_path = str(self.trained_model_path) + f"E_{epoch_n}"
        else:
            model_path = str(self.trained_model_path) + f"_{name}"

        self.load_weights(model_path).expect_partial()
        with self.scale_path.open(mode="r") as scale_file:
            scales = yaml.safe_load(scale_file)
        if scales != self.data_handler.scales:
            logger.error(f"Scales with which this model is trained are "
                         f"different from those of the data processing. "
                         f"Make sure you know what's happening. For now, "
                         f"the model's scales will be set to the data "
                         f"handler.")
            self.data_handler.set_scales(scales)
        logger.info(f"Model loaded from {model_path}")

    # -------------------------------------------------------------------------------------------------------------------

    # --------------------------------------------- Metrics Write and Reset ---------------------------------------------
    @staticmethod
    def write_summaries(summary_writer, metrics, epoch, log_str):
        with summary_writer.as_default():
            for metric, value in metrics.items():
                tf.summary.scalar(metric, value.result(), step=epoch)

        # print train metrics
        strings = ['%s: %.5e' % (k, v.result()) for k, v in metrics.items()]
        logger.info(f"Epoch {epoch} | "
                    f"{log_str}: {' - '.join(strings)} ")

    def reset_metrics(self):
        metrics = chain(self._train_metrics.values(),
                        self._val_metrics.values())
        for metric in metrics:
            metric.reset_states()

    # -------------------------------------------------------------------------------------------------------------------

    # --------------------------------------------- Test data reconstruction --------------------------------------------
    def reconstruct_test_data(self):
        global gif_processes
        logger.info("Making reconstructions on the test dataset...")
        test_dataset = self.data_handler.get_test_dataset(batch_size=1)
        all_vtps = [] # complete list of vtp files
        for counter, (adj, features) in enumerate(test_dataset):
            save_original = False if self.epoch != 0 else True
            _, vtps = self.reconstruct(adj, features, f_name=str(counter), save_original=save_original)
            all_vtps.extend(vtps) # vtp files to complete list
        
        # group vtp files by prefix (string before underscore _), result list of pairs (prefix_p, vtps_with_prefix_p)
        groups = [(j, list(i)) for j, i in groupby(all_vtps, lambda a: a.partition('_')[0])]
        # convert each vtp to a png view of the 3D shape using the predefined camera view in "paraview_state.pvsm"
        # vtps in the same group get pngs in the same folder
        # pngs in the same folder gets converted to a gif
        # each group gets processed by a new parallel process
        for prefix, filenames in groups:
            process = mp.Process(target=vtps_to_gif, args=(filenames, self.rec_dir, self.rec_png_dir, prefix, self.rec_gif_dir))
            gif_processes.append(process)
            process.start()
        
        logger.info(f"Finished reconstructions, generating gifs in parallel")

    def reconstruct(self, adj, features, guess_triangle=False,
                    save_original=True, f_name=''):
        latents, _ = self.encoder((adj, features), training=False)
        reconstructed_mesh_features, _ = self.decoder(latents, training=False)

        recs = []
        vtp_filenames = []
        # iterate over batchsize
        for i in range(latents.shape[0]):
            ref_poly = self.data_handler.reference_poly
            new_points = reconstructed_mesh_features[i, ...].numpy()
            new_polys = None
            if guess_triangle:
                new_polys = construct_triangles(adj[i, ...].numpy())
            
            epoch_str = str(self.epoch).zfill(3)
            filename = "ReconstructedMeshE{}_{}".format(epoch_str, f_name.zfill(3))
            reconstructed_poly = overwrite_vtkpoly(
                ref_poly, points=new_points, polys=new_polys,
                save=True, output_dir=self.rec_dir,
                name=filename)
            
            vtp_filenames.append(filename) # add filename to list of generated files in self.rec_dir

            if save_original:
                filename = "OriginalMeshE{}_{}".format(str(self.epoch).zfill(3), f_name.zfill(3))
                original_poly = overwrite_vtkpoly(
                    ref_poly, points=features[i, ...].numpy(),
                    save=True, output_dir=self.rec_dir,
                    name=filename)
                vtp_filenames.append(filename) # add filename to list of generated files in self.rec_dir

            recs.append(reconstructed_poly)
        
        return recs, vtp_filenames

    def latent_space_visualization(self):
        logger.info(f"Latent space visualisation for all test set...")
        test_dataset = self.data_handler.get_test_dataset(batch_size=1)
        ref_poly = self.data_handler.reference_poly

        factors = [-2, 2]
        dev = 0.33
        for i, (adj, features) in enumerate(test_dataset.take(1)): # we only take 1 random shape from the test dataset
                                                                   # and perform the following computations on it

            overwrite_vtkpoly(
                ref_poly, points=features.numpy()[0, ...],
                save=True, output_dir=self.inter_dir,
                name="original_mesh") # save the test shape

            latents, _ = self.encoder((adj, features), training=False) # encode test shapes and get latent vector 
                                                                       # representation of it of dim L
            for l_idx in range(latents.shape[-1]): # for each dimension l_idx of latent vector (i,e 0 to L-1)
                for f in factors: # for each factor f
                    temp = latents.numpy().copy() # create a copy of the latent to work on
                    temp[..., l_idx] = temp[..., l_idx] * (1 + f * dev) # add variance/deviation to all values in the vector
                                                                        # up to l_idx 

                    reconstructed_mesh_features, _ = self.decoder(
                        tf.convert_to_tensor(temp), training=False) # decode the modified latent vector

                    new_points = reconstructed_mesh_features.numpy()[0, ...] # convert reconstructed mesh to numpy array
                    overwrite_vtkpoly(
                        ref_poly, points=new_points,
                        save=True, output_dir=self.inter_dir,
                        name=f"i{i:03d}_l_idx_{l_idx:03d}"
                             f"_f_{f:03d}") # save reconstructed mesh

    # -------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------- Model Evaluation ------------------------------------------------
    def evaluate(self, **kwargs):
        pass

    @tf.function
    def _evaluate(self, adj, features):
        latents, _ = self.encoder((adj, features), training=False)
        reconstructed, _ = self.decoder(latents, training=False)
        error = self._reconstruction_error(features, reconstructed)
        # logging
        self._val_metrics['reconstruction_error'](error)
        return error

    # -------------------------------------------------------------------------------------------------------------------