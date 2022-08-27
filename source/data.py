from abc import ABC, abstractmethod
from os import listdir
from pathlib import Path
import logging
import glob
import string
import random, time
from typing import List
import source.utils as utils
import vtk
import matplotlib.pyplot as plt

import yaml
from zipfile import ZipFile
import multiprocessing as mp

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import CubicSpline
from spektral.utils.convolution import normalized_adjacency
from spektral.utils.convolution import gcn_filter
from spektral.utils.convolution import normalized_laplacian
from spektral.utils.convolution import rescale_laplacian
from spektral.utils.convolution import chebyshev_filter
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from source.mesh_sampling import generate_transform_matrices
from source.mesh_sampling import _get_poly_vertices
from source.shape_model_utils import decimate_poly
from source.shape_model_utils import generate_shape_from_modes
from source.shape_model_utils import load_polydata
from source.shape_model_utils import vtkpoly_to_adj, vtkpoly_to_feats
from source.shape_model_utils import poly_to_nxgraph
from source.shape_model_utils import nxgraph_to_vtkpoly
from source.shape_model_utils import get_subgraph
from source.shape_model_utils import merge_polys
from source.shape_model_utils import vtkpoly_linearly_interpolated
from source.shape_model_utils import overwrite_vtkpoly
from source.shape_model_utils import write_vtk_xml
from source.shape_model_utils import compute_volumes_feats
from source.shape_model_utils import generate_seg_maps
from source.shape_model_utils import get_disk_method_ef
from source.shape_model_utils import save_mesh_vid
from source.shape_model_utils import compute_ef_mesh_vid
from source.shape_model_utils import save_mesh_vid_slice_with_data_as_np

from source.misc.rename_reconstructed import add_zeros
from source.constants import ROOT_LOGGER_STR
from source.constants import UK_FOLDER
from source.constants import CISTIB_FOLDER
from source.constants import CISTIB_ALL_FOLDER
from source.constants import CISTIB_EPIENDO
from source.constants import CONRAD_FOLDER
from source.constants import CONRAD_VTK_FOLDER
from source.constants import CONRAD_MEAN_FILE
from source.constants import CONRAD_PARAM_FILE
from source.constants import CONRAD_COMPONENTS
from source.constants import CONRAD_NUM_PHASE
from source.constants import CONV_INPUTS
from source.constants import CONRAD_DATA_PARAMS
from source.constants import CONRAD_COMPONENTS_TO_IDS
from source.constants import CONRAD_VIEWS

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


# ----------------------------------------- Parent Class of the classes below -----------------------------------------------
class MeshDataset(ABC):
    def __init__(self, data_dir, results_dir, data_name, n_modes,
                 modes, std_shape_test,
                 mesh_reduction, save_files, tf_record,
                 batch_size, n_prefetch, shuffle_buffer,
                 std_shape_generation, ds_factors,fix_time_step=None, 
                 gen_seqs=False, time_per_cycle_mean=None, time_per_cycle_std=None,
                 nb_cycles_mean=None, nb_cycles_std=None, shift_max=None,
                 shapes_per_cycle_mean=None, shapes_per_cycle_std=None, pulse_min=None, pulse_max=None,
                 low_efs=False):
        
        if modes is None: # if list of modes to use not specified in the config file, use all 16 modes [0, ..., 15]
            modes = np.array(range(n_modes))
        
        # used when expeeimenting with having one std value per dynamic mode
        # assert len(std_shape_generation) == len(modes)
        # assert len(std_shape_test) == len(modes)

        # save params in self
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.generation_folder = results_dir / data_name / "generated" # folder to generate data when needed
        self.save_files = save_files
        self.tf_records = tf_record
        self.data_name = data_name
        self.n_modes = len(modes)
        self.n_rand_coeff = n_modes
        self.modes = np.sort(modes)
        self.fix_time_step = fix_time_step # either a fixed timestep or sequence of timesteps (when generating a sequence of shapes)
        self.gen_seqs = gen_seqs
        # data generation params
        self.time_per_cycle_mean = time_per_cycle_mean
        self.time_per_cycle_std = time_per_cycle_std
        self.nb_cycles_mean = nb_cycles_mean
        self.nb_cycles_std = nb_cycles_std
        self.shift_max = shift_max
        self.shapes_per_cycle_mean = shapes_per_cycle_mean
        self.shapes_per_cycle_std = shapes_per_cycle_std
        self.pulse_min = pulse_min
        self.pulse_max = pulse_max
        
        self.ds_factors = ds_factors # the downsampling (DS) factor

        self.mesh_reduction = mesh_reduction
        self.batch_size = batch_size
        self.n_prefetch = n_prefetch
        self.shuffle_buffer = shuffle_buffer
        # data variability
        self.data_std = std_shape_generation # variability (standard dev) for the generated shapes for training
        self.test_data_std = std_shape_test # same but for validation and test data
        self.low_efs = low_efs

        # This will be set after `build_transform_matrices` is called
        self.built = False
        self.A, self.M, self.D, self.U = None, None, None, None
        self.enc_in, self.dec_in = None, None
        self.reference_poly = None
        self.input_dim_mesh = None
        self.dataset_output_types = None
        self.dataset_output_shapes = None

        # These will be set once the entire dataset is written to file in
        # case of save_true
        self.scales = None
        self.dataset_min = np.inf
        self.dataset_max = -np.inf
        self.dataset_scale = None
        self.data_dir = None # vtk data dir
        self.datasets_dir = None # train, val and test data dir

        # Later gets written to logs
        self.params = {"data_name": self.data_name,
                       "n_modes": self.n_modes,
                       "modes": self.modes.tolist(),
                       "mesh_reduction": self.mesh_reduction,
                       "std_train": self.data_std,
                       "std_test": self.test_data_std,
                       "low_efs": self.low_efs}

    # save new scales
    def set_scales(self, scales):
        self.scales = scales

        x_min, x_max = scales['x_min'], scales['x_max']
        z_min, z_max = scales['z_min'], scales['z_max']
        y_min, y_max = scales['y_min'], scales['y_max']

        self.dataset_min = np.min([x_min, y_min, z_min]).astype(np.float32)
        self.dataset_max = np.max([x_max, y_max, z_max]).astype(np.float32)
        self.dataset_scale = self.dataset_max - self.dataset_min
        logger.debug(f"Set scales: {scales}")
    
    def pick_scales(self, scales1, scales2):
        if scales1 is None:
            return scales2
        if scales2 is None:
            return scales1
        
        new_scales = {}
        for key in scales1:
            if key[-3:] == "min":
                new_scales[key] = float(np.min([scales1[key], scales2[key]]))
            elif key[-3:] == "max":
                new_scales[key] = float(np.max([scales1[key], scales2[key]]))
        return new_scales
    
    def load_scales(self, scales_p):
        if not scales_p.exists():
            return False
        
        with scales_p.open(mode="r") as scales_file:
            scales = yaml.safe_load(scales_file)
            self.set_scales(scales)
        return True
    
    # Abstract helper methods to get the mean shape used to generate a sample data, 
    # using **kwargs used so that args can be anything when defining this method for 
    # each data handler (e,g CONRADData)  
    @abstractmethod
    def _get_mean_polydata(self, **kwargs):
        """This must be set otherwise `build_transform_matrices` has no mean
        polydata to generate transform matrices"""
        pass

    # Abstract helper method to generate a sample data shape (note usage of **kwargs here too)
    @abstractmethod
    def _generate_sample(self, time, std_num, **kwargs):
        """Generate a shape from mean and PCA data of the dataset at the
        given `time` """
        pass

    # build and store matrices used in graph conv layers, more details below
    # matrices built using the reference poly (mean poly (optionally reduced))
    def build_transform_matrices(self, enc_list, dec_list, enc_chebyshev_K_list = None, dec_chebyshev_K_list = None):
        """Build necessary inputs for graph convolutions
        At each graph convolution layer, specific inputs such as Chebyshev
        polynomials, modified Laplacian, binary adjacency matrix or
        normalised adjacency matrix are required alongside the node features.

        `enc_list` and `dec_list` args are lists of the same length containing
        string names of each convolution layer for the encoder and decoder
        respectively. For example:
        `enc_list = ['ChebConv', 'ChebConv']` and
        'dec_list = [''GraphConv', 'GraphConv']'

         for a complete list of possible convolution layers please see
         `constants.CONV_INPUTS`
        
        Given the name of the conv layers of encoder and decoder as a string 
        in the enc_list and dec_list args, this function computes and returns
        the list of necessary input matrices for each of these conv layers.

        In addition to the list of inputs for each convolution layer of the
        encoder and decoder (aka `enc_in` and `dec_in` below in code), it also 
        returns outputs of the `generate_transform_matrices`.

        The function also sets some necessary information about the dataset
        to be used during training.

        :param enc_list: list of strings, each specifying the name of a
        convolution layer in the encoder
        :param dec_list:list of strings, each specifying the name of a
        convolution layer in the decoder
        :return:
               M: a list of polys where M[i+1] is the downsampled version of M[i] (i = 0, ...)
               the downsampling factor from M[i] to M[i+1] is self.ds_factor[i] and M[0] is the reference poly
               (mean poly possibly with a mesh reduction, see below)
               e,g: self.ds_factor = [2 2 2] and M[0] = mean_poly then
               M[1] = M[0] downsampled by a factor of 2
               M[2] = M[1] downsampled by a factor of 2
               M[3] = M[2] downsampled by a factor of 2
               specified in `factors`.
               A: Adjacency matrix for each of the polys/shapes/graphs in M, same order
               D: Downsampling transforms between each of the polys,
               so D[i] is the transform to downsample from M[i] to M[i+1]
               U: Upsampling transforms between each of the polys
               so U[i] is the transform to upsample from M[i+1] to M[i]
               enc_in: List of input matrices for each convolution layer of the encoder
               dec_in: List of input matrices for each convolution layer of the decoder
        """
        # if self.built:  # Already set
        #     return self.M, self.A, self.D, self.U, self.enc_in, self.dec_in

        # needed assertion, since each downsampling block in the encoder has 1 conv and 1 downsample
        # so nb of downsampling layers (i,e downsampling factors) = nb of conv layers in the encoder
        # each downsample block in the encoder has a corresponding upsample block in the decoder
        # so nb of downsample layers = nb of upsample layers
        # Also each upsampling block in the decoder has 1 upsample and 1 conv,
        # so nb of upsampling layers = nb of conv layers in the decoder
        # so nb of dowsample layers = nb conv layers in encoder = nb of upsample layers = nb of conv layers in decoder
        assert len(self.ds_factors) == len(enc_list) == len(dec_list)

        # reference poly: either mean poly at time 0.4 if reduction is None, or mean poly at time 0.4 with reduction applied 
        tmp_poly = decimate_poly(
            polydata=self._get_mean_polydata(time=0.4, is_train=True),
            reduction=self.mesh_reduction)
        
        reference_poly = vtk.vtkPolyData()
        reference_poly.DeepCopy(tmp_poly)
        
        # remove unnecessary data arrays
        point_data = reference_poly.GetPointData()
        for i in range(17):
            point_data.RemoveArray(f"eig_mode_{i}")
            point_data.RemoveArray(f"mode_{i}")

        transform_matrices_folder = self.generation_folder.parent / "transform matrices"
        transform_matrices_folder.mkdir(parents=True, exist_ok=True)

        matrices_loaded = False
        for folder in transform_matrices_folder.iterdir():
            if folder.is_dir():
                matrices_path = folder / "matrices.npz"
                loaded_data = np.load(matrices_path, allow_pickle=True)
                loaded_ds_factors = list(loaded_data['ds_factors'])
                if self.ds_factors == loaded_ds_factors:
                    logger.info(f"Found transform matrices in folder {folder}...")
                    logger.info("Skip generating and load matrices and meshes...")
                    V = loaded_data['V']
                    A = loaded_data['A']
                    D = loaded_data['D']
                    U = loaded_data['U']
                    
                    matrices_loaded = True
                    logger.info("Loaded matrices")

        if not matrices_loaded: # could not find file containing transform matrices to load from, generate them and save
            logger.info("Generating and saving Transform Matrices and meshes...")
            # TODO (fabian): check if A, D, U are scale invariant.
            #  reference_poly already scaled?
            M, A, D, U = generate_transform_matrices(reference_poly, self.ds_factors)

            # make entries zero/one
            A = [*map(lambda x: x.astype('bool'), A)]

            # convert to float32
            V = [*map(lambda x: _get_poly_vertices(x).astype('float32'), M)] # get vertices from polys to save
            A = [*map(lambda x: x.astype('float32'), A)]
            D = [*map(lambda x: x.astype('float32'), D)]
            U = [*map(lambda x: x.astype('float32'), U)]
            

            folder_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(6)) # random folder name
            matrices_folder = transform_matrices_folder / folder_name
            matrices_folder.mkdir(parents=True, exist_ok=True)
            matrices_file = matrices_folder / "matrices.npz"
            np.savez_compressed(matrices_file, V=V, A=A, D=D, U=U, ds_factors=self.ds_factors)
            ds_polys_folder = matrices_folder / "ds_polys"
            ds_polys_folder.mkdir(parents=True, exist_ok=True)
            for i, poly in enumerate(M): # for each mesh in the sequence of downsampled meshes
                filename = f"M_{i}"
                write_vtk_xml(poly, save=True, output_dir=ds_polys_folder, name=filename)

            logger.info(f"Transform matrices and meshes saved in {matrices_folder}")

        # Define an inner method get_inputs()
        # returns a list of matrices called "matrices" where matrices[i] is the input matrix
        # to conv layer in layer_list[i].

        # For the encoder, adj = A, i,e A[i] is the adjacency matrix of poly/shape M[i]
        # so A[0] is the biggest shape (not downsampled) and A[-1] is the smallest one (most downsampled)
        # for the decoder, let rev_A = reverse(A), then adj = rev_A[1:], i,e we skip the 
        # smallest shape so adj[0] is the 2nd smallest shape and shapes in adj are ordered in
        # increasing order of upsampling (i,e adj[i+1] bigger than adj[i])
        
        # For each of the encoder and decoder, adj[i] is used to compute 
        # matrices[i] = input to the i'th conv layer layer_list[i]
        def get_inputs(layer_list, adj, chebyshev_K_list):
            """get correct input matrix for the different GNN layers"""
            matrices = []
            mat = None
            for i, layer_name in enumerate(layer_list):
                if layer_name in CONV_INPUTS['modified_laplacian']:
                    mat = gcn_filter(adj[i])
                elif layer_name in CONV_INPUTS['normalized_rescaled_laplacian']:
                    mat = rescale_laplacian(normalized_laplacian(adj[i]))
                elif layer_name in CONV_INPUTS['normalized_adjacency']:
                    mat = normalized_adjacency(adj[i])
                elif layer_name in CONV_INPUTS['chebyshev_polynomials']:
                    mat = chebyshev_filter(adj[i], chebyshev_K_list[i] - 1)
                elif layer_name in CONV_INPUTS['adjacency']:
                    mat = adj[i]
                else:
                    raise NotImplementedError(f"Layer {layer_name} is not supported.")
                matrices.append(mat)
            return matrices

        enc_in = get_inputs(enc_list, A, enc_chebyshev_K_list) # pass list of adjacency matrices of shapes in decreasing order of downsampling
        dec_in = get_inputs(dec_list, A[::-1][1:], dec_chebyshev_K_list) # pass list of adjacency matrices of shapes in increasing order of upsampling
                                                                         # (skip adjacency matrix of smallest/most downsampled mesh)
        
        self.enc_in, self.dec_in = enc_in, dec_in
        self.input_dim_mesh = A[0].shape[0]
        self.reference_poly = reference_poly
        if not self.gen_seqs:
            self.dataset_output_types = (tf.float32) # data type of the (feats)
        else:
            # data types of the (feats, times, all_volumes, params), one per elem
            self.dataset_output_types = (tf.float32, tf.float32, tf.float32, tf.float32) 
        self.built = True


        return V, A, D, U, enc_in, dec_in
    
    # ---------------------------------- TF Datasets for Train, Val and Test (Parallel) ------------------------------------

    def generate_and_save(self, sample_params, sample_id, this_dataset_dir, return_dict, low_efs, is_train=True):
        """
            Generates and saves the sample_id'th data sample to this_dataset_dir using the variation
            coefficients std_n which is in sample_params
        """
        assert self.save_files
        # reference poly to compute volumes
        ref_poly = self.reference_poly
        
        # generate a sequence of shapes in a sequence of times t_seq
        nb_cycles, time_per_cycle, cycle_shift, shapes_per_cycle, std_n = sample_params
        
        # min and max nb of cycles per minute
        pulse_min, pulse_max = self.pulse_min, self.pulse_max
        # min and max time per cycle (in sec)
        time_per_cycle_min, time_per_cycle_max = 60 / pulse_max, 60 / pulse_min
        # constraint the time_per_cycle to the range [time_per_cycle_min, time_per_cycle_max]
        time_per_cycle = np.clip(time_per_cycle, time_per_cycle_min, time_per_cycle_max)

        # ------------------------------------ Method 1 of data generation --------------------------------
        # compute time values from cycle values
        total_time = time_per_cycle * nb_cycles
        time_shift = cycle_shift * time_per_cycle

        total_generated_shapes = int(np.floor(shapes_per_cycle*nb_cycles)) # use np.floor to generate possibly 1 less shape
                                                                        # for this nb of cycles (i,e less computation in the NN)
        times = np.linspace(0, total_time, total_generated_shapes)

        # the generator considers a range of length 1 as 1 cycle (e,g [0, 1] or [0.5, 1.5]),
        # we want to generate "nb_cycles" starting at "cycle_shift", so we input a range of length "nb_cycles" 
        # that starts at "cycle_shift" i,e [cycle_shift, nb_cycles+cycle_shift]
        cycles_seq = np.linspace(cycle_shift, nb_cycles+cycle_shift, total_generated_shapes)
        # -------------------------------------------------------------------------------------------------

        feats = [] # feature matrices of the mesh video with above parameters

        skip_low_ef = np.random.choice([True, False])
        if low_efs and not skip_low_ef:
            assert "leftVentricle" in self.components
            # generate the 10 phases of 1 cycle, pick 2 frames as EDF and ESF then interpolate between them
            phases = 10
            seq_1_cycle = np.linspace(0, 0.9, phases)
            cycle_feats = []
            for cycle_point in seq_1_cycle:
                feat = self._prepare_polydata(time=cycle_point, std_num=std_n, is_train=is_train)
                cycle_feats.append(feat)
            
            # calculate lv volumes
            volumes_lv = compute_volumes_feats(cycle_feats, ref_poly, components_list=["leftVentricle"])["leftVentricle"]

            # pick 2 different frames for EDF and ESF
            ef_value = -1
            while ef_value < 5: # only pick 2 frames such that ef value is at least 5
                idx1 = np.random.randint(phases)
                idx2 = np.random.randint(phases)
                while idx1 == idx2:
                    idx2 = np.random.randint(phases)
                
                vols = [volumes_lv[idx1], volumes_lv[idx2]]
                min_vol = min(vols)
                max_vol = max(vols)
                ef_value = ((max_vol - min_vol) / max_vol) * 100
            
            # gather frames in the same order as the cycle, keeping only frames with volumes
            # in the range [min_vol, max_vol]
            new_cycle_feats = []
            new_volumes_lv = []
            for i in range(len(cycle_feats)):
                if volumes_lv[i] >= min_vol and volumes_lv[i] <= max_vol:
                    new_cycle_feats.append(cycle_feats[i])
                    new_volumes_lv.append(volumes_lv[i])
            # if last frame volume larger than first frame, then put last frame first
            if new_volumes_lv[-1] > new_volumes_lv[0]:
                edf = new_cycle_feats[-1]
                l = [edf]
                l.extend(new_cycle_feats[:-1])
                new_cycle_feats = l
            
            # frames representing 1 cycle from EDF to ESF and back to EDF
            cycle_feats = new_cycle_feats

            for cycle_point in cycles_seq:
                cycle_point = self.remap_time(cycle_point) # remap cycle_point to range [0, 1)
                # select 2 frames based on the cycle_point, then linearly interpolate between them
                y_1, y_2, diff = self._get_y1_y2(len(cycle_feats), cycle_point)
                feat_1 = cycle_feats[y_1]
                feat_2 = cycle_feats[y_2]
                feat = feat_1 + diff * (feat_2 - feat_1) # linear interpolate
                feats.append(feat)
        else:
            # generate the feature matrix at each cycle point
            for cycle_point in cycles_seq:
                feat = self._prepare_polydata(time=cycle_point, std_num=std_n, is_train=is_train)
                assert (self.save_files or (feat >= 0).all())

                feats.append(feat)
        
        feats = np.stack(feats)

        # compute the per-component-volumes for each feature matrix and each component in self.components
        # note: these heart are unscaled, so we compute the volume over unscaled heart shapes
        volumes = compute_volumes_feats(feats, ref_poly, components_list=self.components)
        
        # concat the "component volumes" lists in the same order they appear in self.components
        all_volumes = []
        for c in self.components:
            all_volumes.extend(volumes[c])
        
        ef_disks = 0
        ef_vol = 0
        if "leftVentricle" in volumes: # if computed volumes of left ventricle
            # compute EF using disks method
            volumes_lv = volumes["leftVentricle"]
            ed_idx, es_idx = np.argmax(volumes_lv), np.argmin(volumes_lv)
            ed_feats, es_feats = feats[ed_idx], feats[es_idx]
            output_dir = None
            ef_disks, _, _ = get_disk_method_ef(ed_feats, es_feats, ref_poly, ("4CH", "2CH"), slicing_ref_frame="EDF", output_dir=output_dir)
            # compute EF using volumes
            max_vol, min_vol = max(volumes_lv), min(volumes_lv)
            ef_vol = (max_vol - min_vol) / max_vol
            ef_vol = ef_vol * 100.0

        params = {
            "nb_cycles": nb_cycles,
            "time_per_cycle": time_per_cycle,
            "shapes_per_cycle": shapes_per_cycle,
            "cycle_shift": cycle_shift,
            "time_shift": time_shift,
            "frequency": 1/time_per_cycle,
            "EF_Vol": ef_vol,
            "EF_Biplane": ef_disks
        }

        params_list = []
        for p in CONRAD_DATA_PARAMS: # append params in the same order they are in the CONRAD_DATA_PARAMS list
            params_list.append(params[p])
        
        filename = f"{this_dataset_dir.name}_{sample_id:06d}"
        file = this_dataset_dir / filename
        np.savez_compressed(f'{file}.npz', feat_matrix=feats, times=times, all_volumes=all_volumes, params=params_list)

        # save mesh video as .avi file
        mesh_vid = feats
        vid_out_dir = this_dataset_dir.parent / "videos" / this_dataset_dir.name
        vid_duration = total_time
        save_mesh_vid(mesh_vid, self, vid_duration, vid_out_dir, filename, rescale=False)

        sample_scales = {}
        sample_scales["x_min"] = np.min(feats[:, :, 0])
        sample_scales["x_max"] = np.max(feats[:, :, 0])
        sample_scales["y_min"] = np.min(feats[:, :, 1])
        sample_scales["y_max"] = np.max(feats[:, :, 1])
        sample_scales["z_min"] = np.min(feats[:, :, 2])
        sample_scales["z_max"] = np.max(feats[:, :, 2])

        return_data = {}
        return_data["scales"] = sample_scales
        return_data["params"] = params
        return_data["FileName"] = filename
        return_dict[sample_id] = return_data
    
    def get_dataset_from_disk(self, set_name, n_samples, batch_size=None, repeat=False, return_filepaths=False):
        if n_samples is None:
            raise ValueError(f"n_samples can not be infinite (None) while saving data to disk.")
        
        assert self.built
        
        folder, _ = self._save_files_to_disk_timed_parallel(set_name, n_samples)
        dataset, filepaths = self._get_dataset(folder, n_samples, batch_size=batch_size)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(self.n_prefetch)
        
        if not return_filepaths:
            return dataset
        else:
            return dataset, filepaths
    
    def get_dataset_from_disk_spaced_efs(self, set_name, n_samples, batch_size=None, repeat=False):
        if n_samples is None:
            raise ValueError(f"n_samples can not be infinite (None) while saving data to disk.")
        
        assert self.built
        
        folder, dataset_params = self._save_files_to_disk_timed_parallel(set_name, n_samples)
        filenames = dataset_params["FileName"]
        efs = dataset_params["EF_Vol"]
        indices = utils.spaced_efs_indices(efs, n_samples)
        selected_filenames = [filenames[i] for i in indices]
        dataset, _ = self._get_dataset(folder, n_samples, selected_filenames=selected_filenames, batch_size=batch_size)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(self.n_prefetch)
        return dataset

    def _save_files_to_disk_timed_parallel(self, set_name, n_samples):
        logger.info(f"{set_name} dataset generation in parallel...")
        t1 = time.time()
        folder, dataset_params = self._save_files_to_disk_parallel(set_name, n_samples)
        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
        logger.info(f"{set_name} {n_samples} samples generated in {h}:{m}:{s}")
        return folder, dataset_params
    
    def _save_files_to_disk_parallel(self, set_name, n_samples):
        assert self.save_files and self.built

        if self.low_efs:
            logger.info("Low EF Heart Mesh Videos to be included in dataset...")
        else:
            logger.info("Low EF Heart Mesh Videos not included in dataset...")

        # we generate n_parallel files in parallel. If one file is not properly saved
        # then the whole batch of n_parallel data files is dismissed
        n_parallel = 10

        # check if data already exists
        self.generation_folder.mkdir(parents=True, exist_ok=True)
        generated_dirs = [x for x in self.generation_folder.iterdir() if x.is_dir()] # list of dirs in the generation_folder (i,e data folders)
        for generated_dir in generated_dirs: # check each previously generated dir if it matches the current run's config
            config_path = self.generation_folder / generated_dir / "configs.yml"
            if not config_path.exists(): # make sure the config file exists
                continue

            with config_path.open(mode='r') as yamlfile:# load config file
                config_dict = yaml.safe_load(yamlfile)
            
            if config_dict != self.params: # if config params doesn't match current run's params, skip
                continue
            
            # config matched params, then use this folder for storing/loading all the datasets
            self.datasets_dir = self.generation_folder / generated_dir
            # load scales if exists
            scales_path = self.generation_folder / generated_dir / "scale.yml"
            self.load_scales(scales_path)

            # check how many files are already generated and saved
            this_dataset_dir = self.generation_folder / generated_dir / set_name
            generated_files = list(this_dataset_dir.glob(f"*.npz")) # get previously generated .npz files
            n_generated = len(generated_files) # number of previously generated .npz files
            
            if n_samples <= n_generated: # if found more generated files then requested, return this_dataset_dir
                logger.info(f"{n_generated} files already exist on {this_dataset_dir}")
                logger.info(f"{n_samples} requested. Skip generating. Will use old data.")
                if not scales_path.exists():
                    logger.warning(f"No scales file found.")
                params_dir = this_dataset_dir.parent / "params"
                params_filename = f"{this_dataset_dir.name}.csv"
                dataset_params = pd.read_csv(params_dir / params_filename)
                return this_dataset_dir, dataset_params # return path dataset's folder path and dataset params
            else:
                n_to_dismiss = n_generated % n_parallel # dismiss the last group of generated files
                                                        # this is to avoid the case where not all 
                                                        # n_parallel jobs of the last group successfully
                                                        # completed
                count = n_generated - n_to_dismiss # number of files kept
                if count > 0: # if more than one file kept
                    logger.info(f"Will continue to add data to existing folder {this_dataset_dir} to reach {n_samples} samples.")
                    if not scales_path.exists():
                        logger.warning(f"No scales file found.")
                break # found data_dir, then stop searching
        
        if self.datasets_dir is None: # did not find any data dir with matching config file
            # create a new data_dir with random name
            dir_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(6)) # name of the new folder to be created
            self.datasets_dir = self.generation_folder / dir_name
            config_path = self.generation_folder / dir_name / "configs.yml"
            scales_path = self.generation_folder / dir_name / "scale.yml"
            this_dataset_dir = self.generation_folder / dir_name / set_name # folder to save data, set_name = train, val or test
            count = 0
        this_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing {n_samples} random samples to {this_dataset_dir} in parallel.\nThis may take a while")

        # Write parameters to the folder
        with config_path.open(mode='w') as yaml_file:
            yaml.dump(self.params, yaml_file)

        if set_name == "train":
            std_min, std_max = self.data_std[0], self.data_std[1]
        else:
            std_min, std_max = self.test_data_std[0], self.test_data_std[1]
        
        # parallel jobs to generate samples
        manager = mp.Manager()
        return_dict = manager.dict()
        parallel_jobs = []

        samples_ids = []
        samples_params = {}
        samples_params["FileName"] = []
        for i in range(count, n_samples):
            # Get the random parameters for the sample to be generated
            nb_cycles = np.random.normal(self.nb_cycles_mean, self.nb_cycles_std) # nb of heart cycles to be generated
            if(nb_cycles < 1): # generate at least 1 cycle
                nb_cycles = 1
            # draw a cycle duration randomly
            time_per_cycle = np.random.normal(self.time_per_cycle_mean, self.time_per_cycle_std)
            time_per_cycle = np.random.normal(self.time_per_cycle_mean, self.time_per_cycle_std)
            if time_per_cycle <= 0:
                time_per_cycle = self.time_per_cycle_mean

            # at which point of cycle the video starts (e,g 0.5 start the video after 0.5 cycles passed
            # i,e in the middle of a cycle)
            cycle_shift = np.random.uniform(0, self.shift_max)
            # nb of shapes generated for 1 cycle
            shapes_per_cycle = int(np.rint(np.random.normal(self.shapes_per_cycle_mean, self.shapes_per_cycle_std)))
            if shapes_per_cycle <= 0: # avoid edge case
                shapes_per_cycle = self.shapes_per_cycle_mean
            if shapes_per_cycle < 7:
                shapes_per_cycle = 7

            # generate n_rand_coeff weights, one per mode
            std_n = np.random.uniform(std_min, std_max, self.n_rand_coeff)
            sample_params = (nb_cycles, time_per_cycle, cycle_shift, shapes_per_cycle, std_n)

            # self.generate_and_save(sample_params, i, this_dataset_dir, return_dict, set_name == "train")
            # exit(0)
            args=(sample_params, i, this_dataset_dir, return_dict, self.low_efs, set_name == "train")
            p = mp.Process(target=self.generate_and_save, args=args)
            parallel_jobs.append(p)
            p.start()
            samples_ids.append(i)

            # if not yet reached n_parallel jobs, continue adding jobs
            # unless the last added parallel job was for the final sample
            if len(parallel_jobs) < n_parallel and not i+1 == n_samples:
                continue
            
            # wait for the parallel jobs to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)
            logger.info(f"{n_parallel} samples done...")
            logger.info(samples_ids)

            # update scales from the newly generated samples if the dataset being generated is the train dataset
            if set_name == "train":
                new_scales = None
                new_scales = self.pick_scales(new_scales, self.scales)
                for j in samples_ids:

                    if j not in list(return_dict.keys()):
                        logger.warning(f"{j} not in return dict.")
                        continue

                    sample_scales_j = return_dict[j]["scales"]
                    new_scales = self.pick_scales(new_scales, sample_scales_j)
                
                # if self.scales is not None:
                #     for key in new_scales:
                #         if new_scales[key] != self.scales[key]:
                #             logger.debug(f"{key} changed in new scales from {self.scales[key]} to {new_scales[key]}")

                self.set_scales(new_scales)
                with scales_path.open(mode="w") as scale_file:
                    yaml.dump(self.scales, scale_file)
            
            for j in samples_ids:

                if j not in list(return_dict.keys()):
                    logger.warning(f"{j} not in return dict.")
                    continue

                params = return_dict[j]["params"]
                filename = return_dict[j]["FileName"]
                samples_params["FileName"].append(filename)
                for p in params:
                    if p in samples_params:
                        samples_params[p].append(params[p])
                    else:
                        samples_params[p] = [params[p]]
            
            # reset samples ids
            samples_ids = []

        logger.info(f"Finished writing dataset files to disk. Writing dataset params to disk...")
        params_dir = this_dataset_dir.parent / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        params_filename = f"{this_dataset_dir.name}.csv"
        dataset_params = pd.DataFrame(samples_params)
        dataset_params.to_csv(params_dir / params_filename, index=False)
        return this_dataset_dir, dataset_params # return dataset dir
    
    def get_dataset_slices_from_disk(self, mesh_ae, set_name, n_samples, batch_size):
        slices_output_dir = self.save_slices_to_disk_parallel(mesh_ae, set_name, n_samples)

        file_list = glob.glob(str(slices_output_dir) + "/*.npz")
        file_list.sort()
        # data types for (frames, times, params, ed_lax)
        output_types = (tf.float32, tf.float32, tf.float32, tf.float32) 
        dataset = tf.data.Dataset.from_tensor_slices(file_list)
        dataset = dataset.map(lambda f: tf.py_function(self._load_slice_data, [f], output_types),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Messy batching, as RaggedTensors not fully supported by Tensorflow's Dataset
        dataset = dataset.map(lambda a, b, c, d: (tf.expand_dims(a, 0), tf.expand_dims(b, 0), 
                                                    tf.expand_dims(c, 0), tf.expand_dims(d, 0)),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda a, b, c, d: (tf.RaggedTensor.from_tensor(a), tf.RaggedTensor.from_tensor(b),
                                                    tf.RaggedTensor.from_tensor(c), tf.RaggedTensor.from_tensor(d)),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda a, b, c, d: (tf.squeeze(a, axis=1), tf.squeeze(b, axis=1),
                                                    tf.squeeze(c, axis=1), tf.squeeze(d, axis=1)),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(self.n_prefetch)

        return dataset, file_list
    
    @staticmethod
    def _load_slice_data(filepath):
        data = np.load(filepath.numpy())
        times = data['times']
        frames = data['frames']
        params = data['params']
        ed_lax = data['ed_lax']
        
        frames = (frames/255).astype('float32') # normalize
        return frames, times, params, ed_lax

    def save_slices_to_disk_parallel(self, mesh_ae, set_name, n_samples):
        logger.info(f"Slicing mesh {set_name} dataset and saving videos...")
        t1 = time.time()
        mesh_batch_size = 8
        dataset, filepaths = self.get_dataset_from_disk(set_name, n_samples, batch_size=mesh_batch_size, return_filepaths=True)

        # get the list of files not yet processed
        filepaths = [Path(f) for f in filepaths]
        slices_output_dir = filepaths[0].parent / "slices_4CH"
        processed_files = [Path(f).stem for f in glob.glob(str(slices_output_dir) + "/*.npz")] # list of filenames already in slices dir
        remaining_filepaths = [f for f in filepaths if f.stem not in processed_files]
        filepaths = remaining_filepaths

        if len(filepaths) == 0: # no file to process
            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info("All slice videos already exist, using those slice videos...")
            logger.info(f"Done in {h}:{m}:{s}")
            return slices_output_dir
        
        data_folder = filepaths[0].parent
        filenames = [f.stem for f in filepaths]
        dataset, _ = self._get_dataset(data_folder, selected_filenames=filenames, batch_size=mesh_batch_size)
        dataset = dataset.prefetch(self.n_prefetch)

        # parallel jobs to generate slices
        manager = mp.Manager()
        return_dict = manager.dict()
        parallel_jobs = []

        # laxes = utils.get_all_valid_laxes_from_mask()

        for b, (mesh_vids, times, _, params) in enumerate(dataset):
            latents = mesh_ae.encode(mesh_vids.values, times.values, times.row_lengths(), training=False)
            
            all_params = params.to_tensor(default_value=0.0).numpy()
            # up-sample the times to get a higher fps
            times_tensor = times.to_tensor(default_value=-1.0).numpy()
            all_durations = np.amax(times_tensor, axis=1)
            all_fps = np.rint(np.clip(np.random.normal(55, 2, size=(len(all_durations),)), 50, 60)).astype(np.int32)
            all_frames = np.rint(all_fps * all_durations).astype(np.int32)
            times_new = []
            row_lengths_new = []
            for duration, frames in zip(all_durations, all_frames):
                ts = np.linspace(0, duration, frames)
                times_new.extend(ts)
                row_lengths_new.append(len(ts))
            times_new = tf.RaggedTensor.from_row_lengths(times_new, row_lengths=row_lengths_new)

            # reconstruct and convert to numpy
            reconstructions = mesh_ae.decode(latents, times_new.values, times_new.row_lengths(), training=False).numpy()
            row_lengths = times_new.row_lengths().numpy()
            row_limits = np.cumsum(row_lengths)
            times_new = times_new.values.numpy()

            i = 0
            samples_ids = []
            for k, j in enumerate(row_limits):
                mesh_vid_rec = reconstructions[i:j]
                mesh_vid_times = times_new[i:j]
                mesh_vid_params = all_params[k]
                # mesh_lax_new = (np.array([56, 28]), np.array([56, 80]))
                # mesh_lax_new = laxes[np.random.randint(len(laxes))]
                mesh_lax_new = None

                global_index = b * mesh_batch_size + k

                filename = filepaths[global_index].stem
                samples_ids.append(filename)

                # save_mesh_vid_slice_with_data_as_np(mesh_vid_rec, self, mesh_vid_times, mesh_vid_params, mesh_lax_new, slices_output_dir, filename, True)
                # exit(0)
                args=(mesh_vid_rec, self, mesh_vid_times, mesh_vid_params, mesh_lax_new, slices_output_dir, filename, True)
                p = mp.Process(target=save_mesh_vid_slice_with_data_as_np, args=args)
                parallel_jobs.append(p)
                p.start()

                i = j
            
            # wait for the parallel jobs to complete
            n_parallel = len(parallel_jobs)
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)
            logger.debug(f"{n_parallel} samples done...")
            logger.info(samples_ids)
        
        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
        logger.info(f"Done saving mesh slice videos in {h}:{m}:{s}...")
        return slices_output_dir
    
    def get_dataset_mesh_slice_pairs_encoded_from_disk(self, mesh_model, echo_model, set_name, n_samples, force_recompute=False):
        mesh_ae, mesh_model_exp = mesh_model["mesh_ae"], mesh_model["mesh_model_exp"]
        echo_ae, echo_model_exp, echo_model_nb = echo_model["echo_ae"], echo_model["echo_model_exp"], echo_model["echo_model_nb"]
        encoded_data_p, scale_p = self.get_encoded_mesh_slice_pair_data_folder(mesh_model_exp, echo_model_exp, echo_model_nb, set_name, n_samples)
        if encoded_data_p is None or force_recompute:
            slices_dir = self.save_slices_to_disk_parallel(mesh_ae, set_name, n_samples)
            meshes_dir = slices_dir.parent

            slice_filepaths = glob.glob(str(slices_dir) + "/*.npz")
            slice_filepaths.sort()
            mesh_filepaths = glob.glob(str(meshes_dir) + "/*.npz")
            mesh_filepaths.sort()
            # check corresponding filepaths are in the positions in the list
            assert len(slice_filepaths) == len(mesh_filepaths)
            slice_filenames = [Path(f).stem for f in slice_filepaths]
            mesh_filenames = [Path(f).stem for f in mesh_filepaths]
            for i in range(len(mesh_filenames)):
                assert slice_filenames[i] == mesh_filenames[i]

            logger.info(f"Loading {set_name} mesh/slice pairs dataset...")
            
            filepath_pairs = list(zip(slice_filepaths, mesh_filepaths))
            batch_size = 16
            # data types for (slice_frames, slice_times, mesh_feats, mesh_times)
            output_types = (tf.float32, tf.float32, tf.float32, tf.float32) 
            dataset = tf.data.Dataset.from_tensor_slices(filepath_pairs)
            dataset = dataset.map(lambda f: tf.py_function(self._load_mesh_slice_pairs, [f], output_types),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(lambda a, b, c, d: self._tf_normalize_mesh_slice_pairs((a, b, c, d), self.dataset_min, self.dataset_scale),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Messy batching, as RaggedTensors not fully supported by Tensorflow's Dataset
            dataset = dataset.map(lambda a, b, c, d: (tf.expand_dims(a, 0), tf.expand_dims(b, 0), 
                                                        tf.expand_dims(c, 0), tf.expand_dims(d, 0)),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(lambda a, b, c, d: (tf.RaggedTensor.from_tensor(a), tf.RaggedTensor.from_tensor(b),
                                                        tf.RaggedTensor.from_tensor(c), tf.RaggedTensor.from_tensor(d)),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(lambda a, b, c, d: (tf.squeeze(a, axis=1), tf.squeeze(b, axis=1),
                                                        tf.squeeze(c, axis=1), tf.squeeze(d, axis=1)),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.prefetch(self.n_prefetch)

            logger.info("Loaded mesh/slice pairs dataset")

            t1 = time.time()
            dataset_slice_latents = []
            dataset_mesh_latents = []
            for b, (slice_frames, slice_times, mesh_feats, mesh_times) in enumerate(dataset):
                slice_latents = echo_ae.encode(slice_times, slice_frames, return_freq_phase=True, training=False).numpy()
                mesh_latents = mesh_ae.encode(mesh_feats.values, mesh_times.values, mesh_times.row_lengths(), training=False).numpy()

                dataset_slice_latents.append(slice_latents)
                dataset_mesh_latents.append(mesh_latents)

                if b % 10 == 0:
                    t2 = time.time()
                    h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
                    logger.info(f"Batch {b} after {h}:{m}:{s}")
            
            dataset_slice_latents = np.concatenate(dataset_slice_latents, axis=0)
            dataset_mesh_latents = np.concatenate(dataset_mesh_latents, axis=0)
            
            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done encoding {set_name} slices and meshes in {h}:{m}:{s}")

            encoded_data_p = self.datasets_dir / "encoded_mesh_slice_pairs" / f"{mesh_model_exp}_{echo_model_exp}_{echo_model_nb}"
            logger.info(f"Saving encoded data under {encoded_data_p}")
            encoded_data_p.mkdir(parents=True, exist_ok=True)
            encoded_data_file_p = encoded_data_p / f"{set_name}.npz"
            np.savez_compressed(encoded_data_file_p, slice_latents=dataset_slice_latents, mesh_latents=dataset_mesh_latents)
            logger.info(f"Done saving.")
        else:
            logger.info(f"Loading pre-encoded data from {encoded_data_p}")
            t1 = time.time()
            data = np.load(encoded_data_p)
            dataset_slice_latents = data["slice_latents"]
            dataset_mesh_latents = data["mesh_latents"]
            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done loading {set_name} data in {h}:{m}:{s}")

            # set the scales
            if scale_p.exists(): # if scales file exists, load it
                with scale_p.open(mode="r") as scale_file:
                    scales = yaml.safe_load(scale_file)
                    
                    self.set_scales(scales)
                    logger.debug(f"Loaded scales: {scales}")
            
            dataset_slice_latents, dataset_mesh_latents = dataset_slice_latents[:n_samples], dataset_mesh_latents[:n_samples]
        
        logger.info(f"Slice latents shape: {dataset_slice_latents.shape}")
        logger.info(f"Mesh latents shape: {dataset_mesh_latents.shape}")

        return dataset_slice_latents, dataset_mesh_latents



    @staticmethod
    def _load_mesh_slice_pairs(filepath_pair):
        slice_filepath, mesh_filepath = filepath_pair

        slice_data = np.load(slice_filepath.numpy())
        slice_frames, slice_times = slice_data['frames'], slice_data['times']

        mesh_data = np.load(mesh_filepath.numpy())
        mesh_feats, mesh_times = mesh_data['feat_matrix'], mesh_data['times']

        slice_frames = (slice_frames/255).astype('float32') # normalize
        return slice_frames, slice_times, mesh_feats, mesh_times
    
    @staticmethod
    def _tf_normalize_mesh_slice_pairs(input, dataset_min, dataset_scale):

        slice_frames, slice_times, mesh_feats, mesh_times = input
        
        mesh_feats = tf.subtract(mesh_feats, dataset_min)
        mesh_feats = tf.truediv(mesh_feats, dataset_scale)

        return slice_frames, slice_times, mesh_feats, mesh_times


    # --------------------------------------------- TF Datasets for Train, Val and Test ------------------------------------
    def get_train_dataset(self, n_samples=None, batch_size=None, repeat=False):
        """Return a tf dataset with `n_samples`.

        If `self.save_files == True`, data will first get saved to disk then
        get feed to the network. This initially will take considerable
        amount of time, but the training will be faster.

        If `self.save_files == True`, a proper scaling of data is performed.
        Otherwise each sample is separately shifted to be positive in all
        three axis.

        Note that if `self.save_files == False`, spline interpolation probably
        will take an eternity to be done.

        :param n_samples: Number of samples to generate. If None,
        the infinite number. Don't do this with `self.save_files == True`!
        :return: tf dataset
        """
        assert self.built # make sure the model has been built

        std_min, std_max = self.data_std[0], self.data_std[1]

        if self.save_files:
            if n_samples is None:
                raise ValueError(f"n_samples can not be infinite (None) while "
                                 f"saving data to disk.")

            gen = self._dataset_generator(std_min, std_max, n_samples, is_train=True) # generator to generate shapes (not generated yet)
            folder = self._save_files_to_disk_timed("train", gen, n_samples) # use the generator to generate and save data shapes in a folder
            dataset, _ = self._get_dataset(folder, n_samples, batch_size=batch_size) # get TF Dataset from data in save folder of shapes
            if repeat:
                dataset = dataset.repeat()
            dataset = dataset.prefetch(self.n_prefetch)
            return dataset
        else:
            return tf.data.Dataset.from_generator(
                self._dataset_generator,
                self.dataset_output_types,
                self.dataset_output_shapes,
                args=(std_min, std_max, n_samples, True)
                # TODO: check repeat or n_samples = inf for not saved
            ).shuffle(self.shuffle_buffer).batch(self.batch_size).repeat(
            ).prefetch(self.n_prefetch)

    def get_val_dataset(self, n_samples, batch_size=None):
        """Return a tf dataset with `n_samples`.
        Same as `self.get_train_dataset`, only will save data to `val`
        folder."""

        assert self.built

        std_min, std_max = self.test_data_std[0], self.test_data_std[1]
        if self.save_files:
            gen = self._dataset_generator(std_min, std_max, n_samples, is_train=False)
            folder = self._save_files_to_disk_timed("val", gen, n_samples)
            dataset, _ = self._get_dataset(folder, n_samples, batch_size=batch_size, data='val')
            return dataset
        else:
            return tf.data.Dataset.from_generator(
                self._dataset_generator,
                self.dataset_output_types,
                self.dataset_output_shapes,
                args=(std_min, std_max, n_samples, False)
            ).shuffle(self.shuffle_buffer).batch(self.batch_size).prefetch(
                self.n_prefetch)
    
    def get_test_dataset(self, n_samples, batch_size=None):
        assert self.built

        std_min, std_max = self.test_data_std[0], self.test_data_std[1]

        if self.save_files:
            gen = self._dataset_generator(std_min, std_max, n_samples, is_train=False) # dataset generator
            folder = self._save_files_to_disk_timed("test", gen, n_samples) # save n_samples to disk
            dataset, _ = self._get_dataset(folder, n_samples, batch_size=batch_size, data='test') # read samples from disk as tf Dataset
            return dataset
        else:
            return tf.data.Dataset.from_generator(
                self._dataset_generator,
                self.dataset_output_types,
                self.dataset_output_shapes,
                args=(std_min, std_max, n_samples, False)
            ).shuffle(self.shuffle_buffer).batch(self.batch_size).prefetch(
                self.n_prefetch)

    def get_test_dataset_old(self, batch_size=None):
        """Return a tf dataset.
        Sample configurations are hardcoded for test purposes."""
        assert self.built

        # TODO: make configurable
        times = np.linspace(0, 1, 50, endpoint=False)
        std = self.std_shape_test
        n_items = len(times)

        if self.save_files:
            gen = self._test_data_generator(std, times)
            folder = self._save_files_to_disk_timed("test", gen, n_items)
            dataset, _ = self._get_dataset(folder, n_items, batch_size=batch_size, data='test')
            return dataset
        else:
            return tf.data.Dataset.from_generator(
                self._test_data_generator,
                self.dataset_output_types,
                self.dataset_output_shapes,
                args=(std, times)
            ).take(n_items).batch(1).prefetch(n_items)
    
    # generator function that yields a shape (i,e adjacency matrix and feature matrix)
    # recall: generator function is put on hold (paused) when the yield keyword is met
    # the argument of the "yield" keyword are returned
    # the function continues from the last encountered yield when called using the next()
    # e,g
    # myGen = genFunc
    # next(myGen)
    # next(myGen)
    # ... 
    def _dataset_generator(self, std_min, std_max, n_samples, is_train=True):
        """Generates `n_samples` random polygons where for each sample the
        variation `std_num` is randomly drawn between `std_min` and
        `std_max`"""

        # generate n_samples many meshes
        ii32 = np.iinfo(np.int32)
        n_samples = int(ii32.max() if n_samples is None else n_samples) # n_samples = INTEGER.MAX (i,e infinite) OR finite n_samples

        # reference poly to compute volumes
        ref_poly = decimate_poly(
            polydata=self._get_mean_polydata(time=0.4, is_train=True),
            reduction=self.mesh_reduction)
        
        for i in range(n_samples): # loop that keeps generating a new shape (adjacency matrix and feature matrix)
            std_n = np.random.uniform(std_min, std_max, self.n_rand_coeff) # generate n_rand_coeff weights, one per mode
            
            if not self.gen_seqs: # generate 1 shape at a certain time t
                # we either generate a shape at a fixed point in time (for all data) or at a random point in time
                time = (self.fix_time_step if self.fix_time_step is not None else np.random.uniform(0, 1))
                point_feature_matrix = self._prepare_polydata(time=time, std_num=std_n, is_train=is_train)

                assert (self.save_files or (point_feature_matrix >= 0).all())
                yield point_feature_matrix

            else: # generate a sequence of shapes in a sequence of times t_seq
                # nb of heart cycles to be generated
                nb_cycles = np.random.normal(self.nb_cycles_mean, self.nb_cycles_std)
                if(nb_cycles < 1): # generate at least 1 cycle
                    nb_cycles = 1
                
                # min and max nb of cycles per minute
                pulse_min, pulse_max = self.pulse_min, self.pulse_max
                # min and max time per cycle (in sec)
                time_per_cycle_min, time_per_cycle_max = 60 / pulse_max, 60 / pulse_min
                # draw a cycle duration randomly
                time_per_cycle = np.random.normal(self.time_per_cycle_mean, self.time_per_cycle_std)
                # constraint the time_per_cycle to the range [time_per_cycle_min, time_per_cycle_max]
                time_per_cycle = np.clip(time_per_cycle, time_per_cycle_min, time_per_cycle_max)

                # at which point of cycle the video starts (e,g 0.5 start the video after 0.5 cycles passed
                # i,e in the middle of a cycle)
                cycle_shift = np.random.uniform(0, self.shift_max)
                # nb of shapes generated for 1 cycle
                shapes_per_cycle = int(np.random.normal(self.shapes_per_cycle_mean, self.shapes_per_cycle_std))

                # ------------------------------------ Method 1 of data generation --------------------------------
                # compute time values from cycle values
                total_time = time_per_cycle * nb_cycles
                time_shift = cycle_shift * time_per_cycle

                total_generated_shapes = int(np.floor(shapes_per_cycle*nb_cycles)) # use np.floor to generate possibly 1 less shape
                                                                              # for this nb of cycles (i,e less computation in the NN)
                # timestep = total_time / total_generated_shapes
                times = np.linspace(0, total_time, total_generated_shapes)

                # the generator considers a range of length 1 as 1 cycle (e,g [0, 1] or [0.5, 1.5]),
                # we want to generate "nb_cycles" starting at "cycle_shift", so we input a range of length "nb_cycles" 
                # that starts at "cycle_shift" i,e [cycle_shift, nb_cycles+cycle_shift]
                cycles_seq = np.linspace(cycle_shift, nb_cycles+cycle_shift, total_generated_shapes)
                # -------------------------------------------------------------------------------------------------

                point_feature_matrices = []
                for cycle_point in cycles_seq:
                    point_feature_matrix = self._prepare_polydata(time=cycle_point, std_num=std_n, is_train=is_train)
                    assert (self.save_files or (point_feature_matrix >= 0).all())

                    point_feature_matrices.append(point_feature_matrix)
                
                point_feature_matrices = np.stack(point_feature_matrices)

                # compute the per-component-volumes for each feature matrix and each component in self.components
                # note: these heart are unscaled, so we compute the volume over unscaled heart shapes
                volumes = compute_volumes_feats(point_feature_matrices, ref_poly, components_list=self.components)
                
                # concat the "component volumes" lists in the same order they appear in self.components
                all_volumes = []
                for c in self.components:
                    all_volumes.extend(volumes[c])

                ef = 0
                if "leftVentricle" in volumes: # if computed volumes of left ventricle
                    volumes_lv = volumes["leftVentricle"]
                    max_vol, min_vol = max(volumes_lv), min(volumes_lv)
                    ef = (max_vol - min_vol) / max_vol
                    ef = ef * 100.0


                params = {
                    "nb_cycles": nb_cycles,
                    "time_per_cycle": time_per_cycle,
                    "shapes_per_cycle": shapes_per_cycle,
                    "cycle_shift": cycle_shift,
                    "time_shift": time_shift,
                    "frequency": 1/time_per_cycle,
                    "EF_Vol": ef
                }

                params_list = []
                for p in CONRAD_DATA_PARAMS: # append params in the same order they are in the CONRAD_DATA_PARAMS list
                    params_list.append(params[p])

                yield point_feature_matrices, times, all_volumes, params_list
    
    # (experiment) generate dataset using one std value per dynamic mode to see if can get more diverse EFs while still 
    # having meshes looking like 3D hearts (i,e not too distorted)
    def gen_data_plot_efs(self, n_samples, experiment_path, counter, videos_to_save=0, is_train=True):
        logger.info(f"Generating {n_samples} samples and plotting their efs histogram...")
        t1 = time.time()
        efs = []

        # reference poly to compute volumes
        ref_poly = self.reference_poly
        
        efs = []
        stds = self.test_data_std
        if is_train:
            stds = self.data_std
        logger.info(stds)
        saved_feats = []
        saved_times = []
        for i in range(n_samples): # loop that keeps generating a new shape (adjacency matrix and feature matrix)
            std_n = []
            for std in stds:
                std = np.abs(std)
                std_i = np.random.uniform(-std, std, 1) # generate n_rand_coeff weights, one per mode
                std_n.append(std_i)
            std_n = np.concatenate(std_n)

            # nb of heart cycles to be generated
            nb_cycles = np.random.normal(self.nb_cycles_mean, self.nb_cycles_std)
            if(nb_cycles < 1): # generate at least 1 cycle
                nb_cycles = 1
            
            # min and max nb of cycles per minute
            pulse_min, pulse_max = self.pulse_min, self.pulse_max
            # min and max time per cycle (in sec)
            time_per_cycle_min, time_per_cycle_max = 60 / pulse_max, 60 / pulse_min
            # draw a cycle duration randomly
            time_per_cycle = np.random.normal(self.time_per_cycle_mean, self.time_per_cycle_std)
            # constraint the time_per_cycle to the range [time_per_cycle_min, time_per_cycle_max]
            time_per_cycle = np.clip(time_per_cycle, time_per_cycle_min, time_per_cycle_max)

            # at which point of cycle the video starts (e,g 0.5 start the video after 0.5 cycles passed
            # i,e in the middle of a cycle)
            cycle_shift = np.random.uniform(0, self.shift_max)
            # nb of shapes generated for 1 cycle
            shapes_per_cycle = int(np.random.normal(self.shapes_per_cycle_mean, self.shapes_per_cycle_std))

            # ------------------------------------ Method 1 of data generation --------------------------------
            # compute time values from cycle values
            total_time = time_per_cycle * nb_cycles
            time_shift = cycle_shift * time_per_cycle

            total_generated_shapes = int(np.floor(shapes_per_cycle*nb_cycles)) # use np.floor to generate possibly 1 less shape
                                                                          # for this nb of cycles (i,e less computation in the NN)
            timestep = total_time / total_generated_shapes
            times = [i*timestep for i in range(total_generated_shapes)]

            # the generator considers a range of length 1 as 1 cycle (e,g [0, 1] or [0.5, 1.5]),
            # we want to generate "nb_cycles" starting at "cycle_shift", so we input a range of length "nb_cycles" 
            # that starts at "cycle_shift" i,e [cycle_shift, nb_cycles+cycle_shift]
            cycles_seq = np.linspace(cycle_shift, nb_cycles+cycle_shift, total_generated_shapes)
            # -------------------------------------------------------------------------------------------------

            point_feature_matrices = []
            for cycle_point in cycles_seq:
                point_feature_matrix = self._prepare_polydata(time=cycle_point, std_num=std_n, is_train=is_train)
                assert (self.save_files or (point_feature_matrix >= 0).all())

                point_feature_matrices.append(point_feature_matrix)
            
            point_feature_matrices = np.stack(point_feature_matrices)

            if videos_to_save > 0:
                saved_feats.append(point_feature_matrices)
                saved_times.append(times)
                videos_to_save -= 1

            # compute the per-component-volumes for each feature matrix and each component in self.components
            # note: these heart are unscaled, so we compute the volume over unscaled heart shapes
            volumes = compute_volumes_feats(point_feature_matrices, ref_poly, components_list=self.components)
            
            # concat the "component volumes" lists in the same order they appear in self.components
            all_volumes = []
            for c in self.components:
                all_volumes.extend(volumes[c])

            ef = 0
            if "leftVentricle" in volumes: # if computed volumes of left ventricle
                volumes_lv = volumes["leftVentricle"]
                max_vol, min_vol = max(volumes_lv), min(volumes_lv)
                ef = (max_vol - min_vol) / max_vol
                ef = ef * 100.0
            
            efs.append(ef)
            
            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done 1 sample in {h}:{m}:{s}")
            t1 = time.time()
        

        logger.info(efs)
        # ejection fraction histogram
        ef_hist_dir = experiment_path / "ejection_fraction_hist"
        ef_hist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving histogram under {ef_hist_dir}")

        plt.clf()
        filename = f"{counter}.png"
        plt.hist(efs, bins=list(range(5, 101, 5)), histtype='bar', ec='black')
        plt.title(f'Histogram of EFs on {n_samples} samples\nmin: {min(efs)}, max: {max(efs)}')
        plt.savefig(ef_hist_dir / filename, bbox_inches='tight')
        logger.info("Plotted efs histogram")

        vtps_dir = experiment_path / f"{counter}"
        
        for i, (feats, times) in enumerate(zip(saved_feats, saved_times)):
            logger.info(f"Saving video {i} as vtps")
            vtps_dir.mkdir(parents=True, exist_ok=True)
            for j, feat in enumerate(feats):
                filename = f"Mesh{str(i).zfill(3)}_{str(j).zfill(3)}"
                overwrite_vtkpoly(ref_poly, points=feat, save=True, output_dir=vtps_dir, name=filename)

    # not used in CONRADDATA_DHB (only used for test purposes)
    def _test_data_generator(self, std, times):
        """Generate random polygons with given `stds` and `times`. If
        self.use_all_modes == False, for each mode a polygon is separately
        generated."""

        for time in times:
            point_feature_matrix = self._prepare_polydata(time=time,
                                                                      std_num=std,
                                                                      is_train=False)

            assert (self.save_files
                    or (point_feature_matrix >= 0).all())
            yield point_feature_matrix
    # ----------------------------------------------------------------------------------------------------------------------

    # -------------------------------- Generate and save data (or skip if already generated data) --------------------------
    def _save_files_to_disk_timed(self, type_folder, generator, n_samples):
        logger.info("{} dataset generation...".format(type_folder))
        t1 = time.time()
        folder = self._save_files_to_disk(type_folder, generator, n_samples) # use the generator to generate and save data shapes in a folder
        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
        logger.info("{} {} samples generated in {}:{}:{}".format(type_folder, n_samples, h, m, s))
        return folder


    def _save_files_to_disk(self, type_folder, generator, n_samples):
        """Generate `n_samples` polygons using `generator` and save them to
        `self.generation_folder / config_folder / type_folder`.

        If a folder with the similar config file already in
        self.generation_folder exists, then simply load those data.

        if self.tf_records == True, then save samples in tf_records.
        Otherwise as compressed numpy arrays.

        If tfrecords, then there will be `n_samples // 1000` files in the
        folder. Otherwise there will be `n_samples` numpy arrays in the folder.

        :param type_folder: A str that could be `train`, 'val' or 'test'
        :param generator: either `self._dataset_generator` or
        `self._test_data_generator`
        :param n_samples: int indicating the name of samples that the
        `generator` will produce. It is used to find out if a data folder
        with that many samples already exists.
        :return: Path of the generated data.
        """
        assert self.save_files

        # If data does not exist already, create it in a random folder
        config_folder = ''.join(random.choice(string.ascii_lowercase) for _ in range(6)) # name of the new folder to be created
                                                                                         # if data not exists
        # Paths
        self.datasets_dir = self.generation_folder / config_folder
        config_path = self.generation_folder / config_folder / "configs.yml"
        data_path = self.generation_folder / config_folder / type_folder # folder to save data, type_folder = train, val or test
        scale_path = self.generation_folder / config_folder / "scale.yml"
        ext = ".tfrecords" if self.tf_records else ".npz"

        count = 0
        print_step = 10 # when extending a dataset of x samples to y samples x < y
                        # then "c = (x // print_step) * print_step" samples are retained
                        # and y - c samples are further generated (or regenerated)
                        # BE CAREFUL: if print_step > x, then c = 0 samples are retained from previously
                        # generated dataset, i,e no sample retained and all of "y samples" are generated from scratch
                        # even though e,g y = x + 1 (i,e technically only need to generate 1 more sample
                        # but we generate all y samples) 
        n_files = (int(np.ceil(n_samples / print_step)) if self.tf_records
                   else n_samples) # = n_samples (when not using tf.records to save data)
        # Set min max of dataset
        x_min, x_max = np.inf, -np.inf
        y_min, y_max = np.inf, -np.inf
        z_min, z_max = np.inf, -np.inf

        self.generation_folder.mkdir(parents=True, exist_ok=True)
        configs = [x for x in self.generation_folder.iterdir() if x.is_dir()] # list of dir names in the generation_folder (i,e data folders)
        for cfg in configs: # for each of these dirs/data folders
            # build paths
            data_p = self.generation_folder / cfg / type_folder # path to folder with previously generated/saved data
            config_p = self.generation_folder / cfg / "configs.yml"
            scale_p = self.generation_folder / cfg / "scale.yml"

            past_files = list(data_p.glob(f"*{ext}"))
            old_c = len(past_files) # number of previously generated data in "data_p" folder

            if config_p.exists(): # load data config file if exists in data folder
                with config_p.open(mode='r') as yamlfile:
                    config_dict = yaml.safe_load(yamlfile)

            if config_p.exists() and config_dict == self.params: # if data config file has same params as current run of model
                self.datasets_dir = self.generation_folder / cfg # save directory where generated data for this experiment is found
                if scale_p.exists(): # if scales file exists, load it and set in "self"
                    with scale_p.open(mode="r") as scale_file:
                        scales = yaml.safe_load(scale_file)

                        x_min, x_max = scales['x_min'], scales['x_max']
                        z_min, z_max = scales['z_min'], scales['z_max']
                        y_min, y_max = scales['y_min'], scales['y_max']
                        self.set_scales(scales)
                        logger.debug(f"Loaded scales: {scales}")

                if n_files <= old_c: # if nb of previously generated data larger than the one required in the current experiment
                                     # then all data files required have already been generated. Use them.
                    logger.info(
                        f"{old_c} files already exist on {data_p}"
                        f"\n{n_files} requested. Skip generating. Will use old data.")
                    if not scale_p.exists():
                        logger.warning(f"No scales file found. ")
                    return data_p # return path to data folder
                else:
                    config_path = config_p
                    data_path = data_p
                    scale_path = scale_p
                    count = (old_c // print_step) * print_step # let rest = old_c % print_step
                                                               # then count = old_c - rest
                                                               # we keep "count" old data shapes in the data folder
                                                               # regenerate "rest" data shapes that were previously generated
                                                               # and we newly generate "n_samples - (count + rest)" 
                                                               # = "n_samples - old_c" data shapes
                    if count > 0: # = 0 only when no "print_step" reached in the previous data generation experiment/run
                                  # so generation process done from beginning (possibly regenerating up to "print_step" already
                                  # generated data shapes)
                                  # if > 0 then we keep "count" data shapes and continue the data generation 
                        logger.info(
                            f"Will continue to add data to existing folder {data_p} to reach {n_samples} samples.")
                        if not scale_p.exists():
                            logger.warning(f"No scales file found. ")

        data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing {n_samples} random samples to {data_path}.\nThis may take a while")

        # Write all samples to the folder

        tf_writer = None
        if self.tf_records: # create tf_writer when saving as tf.record
            tf_file = data_path / f'{type_folder}_{count}{ext}'
            tf_writer = tf.io.TFRecordWriter(str(tf_file), 'GZIP')

        for i, input in enumerate(generator): # for each data produced by the generator

            if not self.gen_seqs:
                feat = input
            else:
                feat, times, all_volumes, shape_params = input
            
            if self.tf_records: # if saving as tf.record
                feat_val = tf.train.Feature(float_list=tf.train.FloatList(
                    value=feat.flatten().tolist()))
                adj_val = tf.train.Feature(int64_list=tf.train.Int64List(
                    value=adj.flatten().tolist()))
                features = tf.train.Features(
                    feature={"feat_matrix": feat_val,
                             "adj_matrix": adj_val})
                example = tf.train.Example(features=features)
                serialized = example.SerializeToString()
                tf_writer.write(serialized)

            else: # if saving as numpy array .npz
                file = data_path / f'{type_folder}_{i + count:06d}'
                if not self.gen_seqs:
                    np.savez_compressed(f'{file}{ext}', feat_matrix=feat)
                else:
                    np.savez_compressed(f'{file}{ext}', feat_matrix=feat, times=times, all_volumes=all_volumes, params=shape_params)

            if type_folder == 'train': # update scales using x, y and z coordinates of vertices of the new generated shape
                if not self.gen_seqs:
                    x_min = float(np.min([x_min, np.min(feat[:, 0])]))
                    x_max = float(np.max([x_max, np.max(feat[:, 0])]))

                    y_min = float(np.min([y_min, np.min(feat[:, 1])]))
                    y_max = float(np.max([y_max, np.max(feat[:, 1])]))

                    z_min = float(np.min([z_min, np.min(feat[:, 2])]))
                    z_max = float(np.max([z_max, np.max(feat[:, 2])]))
                else:
                    x_min = float(np.min([x_min, np.min(feat[:, :, 0])]))
                    x_max = float(np.max([x_max, np.max(feat[:, :, 0])]))

                    y_min = float(np.min([y_min, np.min(feat[:, :, 1])]))
                    y_max = float(np.max([y_max, np.max(feat[:, :, 1])]))

                    z_min = float(np.min([z_min, np.min(feat[:, :, 2])]))
                    z_max = float(np.max([z_max, np.max(feat[:, :, 2])]))

            if i % print_step == 0 and i != 0: # every print_step, save data config file again and update scales file
                logger.debug(f"{i} samples done...")

                # Write parameters to the folder
                with config_path.open(mode='w') as yaml_file:
                    yaml.dump(self.params, yaml_file)

                if type_folder == 'train':
                    with scale_path.open(mode="w") as scale_file:
                        scales = {"x_min": x_min, "x_max": x_max,
                                  "y_min": y_min, "y_max": y_max,
                                  "z_min": z_min, "z_max": z_max}
                        self.set_scales(scales)
                        yaml.dump(scales, scale_file)

                if self.tf_records: # if saving as tf.record, close current writer and open a new one
                    tf_writer.close() 
                    f_c = (i // print_step) + count
                    tf_file = data_path / f'{type_folder}_{f_c}{ext}'
                    tf_writer = tf.io.TFRecordWriter(str(tf_file), 'GZIP')

            if not self.tf_records and i + count >= n_samples:
                break

        if self.tf_records:
            tf_writer.close()

        # Write data config and scales again one last time
        with config_path.open(mode='w') as yaml_file:
            yaml.dump(self.params, yaml_file)

        if type_folder == 'train': # write scales for train data
            with scale_path.open(mode="w") as scale_file:
                scales = {"x_min": x_min, "x_max": x_max,
                          "y_min": y_min, "y_max": y_max,
                          "z_min": z_min, "z_max": z_max}
                yaml.dump(scales, scale_file)
                self.set_scales(scales)

        logger.info(f"Finished writing files to disk.")
        return data_path # return data path

    def get_encoded_ef_data_folder(self, mesh_model_exp, type_folder, n_samples):
        self.generation_folder.mkdir(parents=True, exist_ok=True)
        configs = [x for x in self.generation_folder.iterdir() if x.is_dir()] # list of dir names in the generation_folder (i,e data folders)
        
        for cfg in configs: # for each of these dirs/data folders
            # build paths
            config_p = self.generation_folder / cfg / "configs.yml"

            if config_p.exists(): # load data config file if exists in data folder
                with config_p.open(mode='r') as yamlfile:
                    config_dict = yaml.safe_load(yamlfile)
                    if config_dict == self.params: # if data config file has same params as current run of model
                        encoded_data_p = self.generation_folder / cfg / "encoded" / mesh_model_exp / f"{type_folder}.npz"
                        scale_p = self.generation_folder / cfg / "scale.yml"
                        if encoded_data_p.exists():
                            return encoded_data_p, scale_p
                        else:
                            logger.info(f"Encoded data {encoded_data_p} does not exist. Encode again.")
                    else:
                        logger.info('Data configs do not match. \n'
                                    f'Encoded data config: {config_dict} \n'
                                    f'Current data config: {self.params} \n'
                                    'Extract encoded data again.')
            else:
                logger.info(f"Data config {config_p} for loading data does not exist")
        
        return None, None
    
    def get_encoded_mesh_slice_pair_data_folder(self, mesh_model_exp, echo_model_exp, echo_model_nb, set_name, n_samples):
        self.generation_folder.mkdir(parents=True, exist_ok=True)
        configs = [x for x in self.generation_folder.iterdir() if x.is_dir()] # list of dir names in the generation_folder (i,e data folders)
        
        for cfg in configs: # for each of these dirs/data folders
            # build paths
            config_p = self.generation_folder / cfg / "configs.yml"

            if config_p.exists(): # load data config file if exists in data folder
                with config_p.open(mode='r') as yamlfile:
                    config_dict = yaml.safe_load(yamlfile)
                    if config_dict == self.params: # if data config file has same params as current run of model
                        encoded_data_p = self.generation_folder / cfg / "encoded_mesh_slice_pairs" / f"{mesh_model_exp}_{echo_model_exp}_{echo_model_nb}" / f"{set_name}.npz"
                        scale_p = self.generation_folder / cfg / "scale.yml"
                        if encoded_data_p.exists():
                            data = np.load(encoded_data_p)
                            latents = data["slice_latents"]
                            if latents.shape[0] >= n_samples:
                                return encoded_data_p, scale_p
        
        return None, None
    
    def get_encoded_ef_data(self, mesh_model_exp, mesh_ae, type_folder, n_samples, return_true_efs=False, batch_size_unencoded=8, force_recompute=False):

        encoded_data_p, scale_p = self.get_encoded_ef_data_folder(mesh_model_exp, type_folder, n_samples)

        if encoded_data_p is None or force_recompute:
            dataset_unencoded = self.get_dataset_from_disk(type_folder, n_samples, batch_size=batch_size_unencoded)

            # encod dataset
            logger.info(f"Pre-encoding {type_folder} data...")
            t1 = time.time()
            latents = []
            true_efs = []
            pred_efs = []
            true_efs_index = CONRAD_DATA_PARAMS.index("EF_Biplane")

            # parallel jobs to compute video efs, 1 process per video ---> faster processing
            manager = mp.Manager()
            return_dict = manager.dict()
            parallel_jobs = []
            max_parallel_jobs = 16

            # compute ef biplane of reconstructed mesh
            mesh_data_handler = mesh_ae.data_handler
            ef_disks_params = {}
            ef_disks_params["view_pair"] = ("4CH", "2CH")
            ef_disks_params["slicing_ref_frame"] = "EDF"

            counter = 0

            for b, (mesh_vids, times, _, true_params) in enumerate(dataset_unencoded):
                # encode meshes (and convert to numpy)
                latent_params, _, reconstructions = mesh_ae(mesh_vids.values, times.values, times.row_lengths(), training=False) # run model end-to-end
                
                # convert to numpy
                reconstructions = reconstructions.numpy()
                latent_params = latent_params.numpy()
                true_params = true_params.to_tensor(default_value=0.0).numpy()

                # get row lengths and row limits
                row_lengths = times.row_lengths().numpy()
                row_limits = np.cumsum(row_lengths)

                logger.info(f"Batch {b}")
                
                if not return_true_efs:
                    i = 0
                    for k, j in enumerate(row_limits): # k'th video in the batch
                        mesh_vid_rec = reconstructions[i:j] # the video's reconstruction

                        # compute pred ef_vol and ef_biplane
                        args=(mesh_vid_rec, mesh_data_handler, ef_disks_params, True, return_dict, counter)
                        p = mp.Process(target=compute_ef_mesh_vid, args=args)
                        parallel_jobs.append(p)
                        p.start()
                    
                        i = j
                        counter += 1
                
                # save values
                latents.append(latent_params)
                true_efs.append(true_params[:, true_efs_index])

                # wait for some parallel jobs to complete
                parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, max_parallel_jobs)
            
            # wait for all parallel jobs to complete
            parallel_jobs = utils.wait_parallel_job_completion(parallel_jobs, 1)

            latents = np.concatenate(latents, axis=0)
            true_efs = np.concatenate(true_efs, axis=0)

            # gather data
            if not return_true_efs:
                to_delete_idx = []
                for i in range(counter):
                    if i not in list(return_dict.keys()):
                        logger.warning(f"{i} not in return dict.")
                        to_delete_idx.append(i)
                        continue
                    _, ef_biplane = return_dict[i]
                    pred_efs.append(ef_biplane)

                latents = np.asarray([l for i, l in enumerate(latents) if i not in to_delete_idx])
                true_efs = np.asarray([ef for i, ef in enumerate(true_efs) if i not in to_delete_idx])

            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done pre-encoding {type_folder} data in {h}:{m}:{s}")
            encoded_data_p = self.datasets_dir / "encoded" / mesh_model_exp
            logger.info(f"Saving encoded data under {encoded_data_p}")
            encoded_data_p.mkdir(parents=True, exist_ok=True)
            encoded_data_file_p = encoded_data_p / f"{type_folder}.npz"
            np.savez_compressed(encoded_data_file_p, latents=latents, true_efs=true_efs, pred_efs=pred_efs)
            logger.info(f"Done saving.")
            
            # select efs to use and reshape efs
            if return_true_efs:
                efs = true_efs
            else:
                efs = pred_efs
            efs = np.reshape(efs, (-1, 1))
        else:
            logger.info(f"Loading pre-encoded data from {encoded_data_p}")
            t1 = time.time()
            data = np.load(encoded_data_p)
            latents = data["latents"]
            # select efs
            if return_true_efs:
                efs = data["true_efs"]
            else:
                efs = data["pred_efs"]
            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
            logger.info(f"Done loading {type_folder} data in {h}:{m}:{s}")
            # reshape efs
            efs = np.reshape(efs, (-1, 1))

            # set the scales
            if scale_p.exists(): # if scales file exists, load it
                with scale_p.open(mode="r") as scale_file:
                    scales = yaml.safe_load(scale_file)
                    
                    self.set_scales(scales)
                    logger.debug(f"Loaded scales: {scales}")
            
            latents, efs = latents[:n_samples], efs[:n_samples]
        
        logger.info(f"Mesh latents shape: {latents.shape}")
        logger.info(f"Mesh efs shape: {efs.shape}")

        return latents, efs
    
    def get_2d_3d_ef_data_folder(self, type_folder, n_samples):
        self.generation_folder.mkdir(parents=True, exist_ok=True)
        configs = [x for x in self.generation_folder.iterdir() if x.is_dir()] # list of dir names in the generation_folder (i,e data folders)
        
        for cfg in configs: # for each of these dirs/data folders
            # build paths
            config_p = self.generation_folder / cfg / "configs.yml"

            if config_p.exists(): # load data config file if exists in data folder
                with config_p.open(mode='r') as yamlfile:
                    config_dict = yaml.safe_load(yamlfile)
                    if config_dict == self.params: # if data config file has same params as current run of model
                        saved_data_p = self.generation_folder / cfg / "2d_3d_ef_data" / f"{type_folder}.npz"
                        scale_p = self.generation_folder / cfg / "scale.yml"
                        if saved_data_p.exists():
                            data = np.load(saved_data_p)
                            efs_3d = data["efs_3d"]
                            if efs_3d.shape[0] >= n_samples:
                                return saved_data_p, scale_p
        
        return None, None


    # ----------------------------------------------------------------------------------------------------------------------
    
    # -------------------------------------- Load already generated data as a TF dataset -----------------------------------
    def _get_dataset(self, folder, n_samples=-1, selected_filenames=None, batch_size=None, data='train'):
        """Return a tf dataset from data that is saved to disk."""

        if self.tf_records:
            # Read tf records
            filenames = list(map(str, list(folder.glob("*.tfrecords"))))
            dataset = tf.data.TFRecordDataset(filenames, 'GZIP')
            dataset = dataset.shuffle(
                self.shuffle_buffer if data == 'train' else 1).batch(
                self.batch_size if batch_size is None else batch_size).map(
                lambda x: self._parse_function(x, self.input_dim_mesh),
                num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
                lambda x, y: self._tf_post_process(x, y,
                                                   self.dataset_min,
                                                   self.dataset_scale),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            if selected_filenames is not None: # only select the selected filenames
                file_list = [str(folder / f"{filename}.npz") for filename in selected_filenames]
            else: # select all files
                # Read numpy arrays
                file_list = glob.glob(str(folder) + "/*.npz")
            
                # For backward compatibility. Test data should keep the order
                file_names = [Path(f).name for f in file_list]
                renamed_names = [add_zeros(f, 6) for f in file_names]
                file_list = [f for _, f in sorted(zip(renamed_names, file_list))]

                if n_samples > 0:
                    file_list = file_list[:n_samples] # limit the number of data files read to avoid reading whole data in folder
                                                    # if we only want part of it (i,e less data used than already generated)
            
            dataset = tf.data.Dataset.from_tensor_slices(file_list)
            dataset = dataset.map(lambda f: tf.py_function(self._load_numpy, [f, self.gen_seqs], self.dataset_output_types),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # dataset = dataset.shuffle(self.shuffle_buffer if data == 'train' else 1)

            if not self.gen_seqs:
                dataset = dataset.map(
                    lambda x: self._tf_post_process((x), self.gen_seqs, self.dataset_min, self.dataset_scale),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                    )
                dataset = dataset.batch(self.batch_size if batch_size is None else batch_size)
            else:
                # process before batching
                dataset = dataset.map(lambda a, b, c, d: self._tf_post_process((a, b, c, d), self.gen_seqs, self.dataset_min, self.dataset_scale),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
                # Messy batching, as RaggedTensors not fully supported by Tensorflow's Dataset
                dataset = dataset.map(lambda a, b, c, d: (tf.expand_dims(a, 0), tf.expand_dims(b, 0), 
                                                          tf.expand_dims(c, 0), tf.expand_dims(d, 0)),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dataset = dataset.map(lambda a, b, c, d: (tf.RaggedTensor.from_tensor(a), tf.RaggedTensor.from_tensor(b),
                                                          tf.RaggedTensor.from_tensor(c), tf.RaggedTensor.from_tensor(d)),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dataset = dataset.batch(self.batch_size if batch_size is None else batch_size)
                dataset = dataset.map(lambda a, b, c, d: (tf.squeeze(a, axis=1), tf.squeeze(b, axis=1),
                                                          tf.squeeze(c, axis=1), tf.squeeze(d, axis=1)),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset, file_list
    # ----------------------------------------------------------------------------------------------------------------------
          
    # ------------------------------------ Loading already generated data (not used) ---------------------------------------
    # not used since not using tf records
    @staticmethod
    def _parse_function(example_proto, input_dim_mesh):
        """
        Read the tf_record from disk. This is used after
        `self._save_files_to_disk` saves data samples to disk in case of
        self.tf_records == True
        """

        dim = input_dim_mesh
        feats = {"feat_matrix": tf.io.FixedLenSequenceFeature(
            ([dim, 3]), tf.float32, allow_missing=True),
            "adj_matrix": tf.io.FixedLenSequenceFeature(
                ([dim, dim]), tf.int64, allow_missing=True)}
        parsed_features = tf.io.parse_example(example_proto, feats)
        feat_matrix = parsed_features['feat_matrix']
        adj_matrix = parsed_features['adj_matrix']
        return tf.squeeze(adj_matrix), tf.squeeze(feat_matrix)
    
    # not used
    @staticmethod
    def _generate_from_file(path):
        """
        Generate data from disk. Reads numpy arrays. This is used after
        `self._save_files_to_disk` saves data samples to disk in case of
        self.tf_records == False
        """

        path = path.decode("utf-8")
        files = [f for f in listdir(path) if f.endswith(".npz")]
        for file in files:
            data = np.load(path + "/" + file)
            yield data['adj_matrix'], data['feat_matrix']
        
    # ----------------------------------------------------------------------------------------------------------------------

    # ------------------------------------ Processing Functions (for creating TF Dataset) ----------------------------------
    def _prepare_polydata(self, time=None, std_num=None, is_train=True):
        """Generate a simple polygon and then decimate it. Make coordinates
        positive if necessary"""

        time = self.remap_time(time)
        

        # generate polydata sample
        polydata = self._generate_sample(time=time, std_num=std_num, is_train=is_train)
        # perform reduction on it (possibly)
        decimated = decimate_poly(polydata, reduction=self.mesh_reduction)
        # convert polydata to feature matrix
        point_feature_matrix = vtkpoly_to_feats(decimated)

        if not self.save_files: # skipped since we are saving files
            # make coordinates positive. If `save_file` == True, then a
            # proper scaling will be performed in `self._tf_post_process()`
            # after files are being read from disk.

            x_coords = point_feature_matrix[:, 0]
            y_coords = point_feature_matrix[:, 1]
            z_coords = point_feature_matrix[:, 2]

            point_feature_matrix[:, 0] = (x_coords + abs(np.min(x_coords)))
            point_feature_matrix[:, 1] = (y_coords + abs(np.min(y_coords)))
            point_feature_matrix[:, 2] = (z_coords + abs(np.min(z_coords)))

        return point_feature_matrix

    @staticmethod
    def _tf_post_process(inp, gen_seqs, dataset_min, dataset_scale):
        """Scale each sample with scaling data already computed"""

        if not gen_seqs:
            point_feature_matrix = inp
        else:
            point_feature_matrix, times, all_volumes, params = inp
        
        nom = tf.subtract(point_feature_matrix, dataset_min)
        normalised_features = tf.truediv(nom, dataset_scale)
        if not gen_seqs:
            return normalised_features
        else:
            return normalised_features, times, all_volumes, params
    
    @staticmethod
    def _load_numpy(path, gen_seqs):
        data = np.load(path.numpy())
        if not gen_seqs:
            return data['feat_matrix']
        else:
            return data['feat_matrix'], data['times'], data['all_volumes'], data['params']

# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------ CONRAD DHB dataset -------------------------------------------------
class CONRADData_DHB(MeshDataset):
    def __init__(self, data_dir, results_dir, components, phases,
                 dynamic_modes,
                 train_interpolation,
                 test_interpolation,
                 mesh_reduction, batch_size, n_prefetch, save_files,
                 std_shape_generation, shuffle_buffer, ds_factors,
                 time_per_cycle_mean, time_per_cycle_std,
                 nb_cycles_mean, nb_cycles_std, shift_max, shapes_per_cycle_mean, shapes_per_cycle_std,
                 pulse_min, pulse_max, low_efs, **kwargs):

        super().__init__(n_modes=16, # using all 16 eigenvectors/modes
                         data_dir=data_dir, # data_dir = repo/heart_mesh/shape_models
                         results_dir=results_dir, # results_dir = repo/experiments
                         # params from config file, explained in config file
                         save_files=save_files,
                         mesh_reduction=mesh_reduction,
                         batch_size=batch_size,
                         n_prefetch=n_prefetch,
                         ds_factors=ds_factors,
                         shuffle_buffer=shuffle_buffer,
                         std_shape_generation=std_shape_generation,
                         gen_seqs=True, # for the DHB model, we generate sequences of the same heart i,e heart beats
                         time_per_cycle_mean=time_per_cycle_mean, 
                         time_per_cycle_std=time_per_cycle_std,
                         nb_cycles_mean=nb_cycles_mean,
                         nb_cycles_std=nb_cycles_std,
                         shift_max=shift_max,
                         shapes_per_cycle_mean=shapes_per_cycle_mean,
                         shapes_per_cycle_std=shapes_per_cycle_std,
                         pulse_min=pulse_min,
                         pulse_max=pulse_max,
                         low_efs=low_efs,
                         **kwargs)
        
        self.data_dir = data_dir / CONRAD_FOLDER # self.data_dir = repo/heart_mesh/shape_models/CardiacModel
        self.vtk_path = self.data_dir / CONRAD_VTK_FOLDER # self.data_dir = repo/heart_mesh/shape_models/CardiacModel/vtkPolys
        if not self.vtk_path.exists(): # vtkPolys folder exists (otherwise create it by unzipping vtkPolys zip file in self.data_dir)
            logger.debug(f"Unzipping Conrad data to {self.vtk_path}")
            with ZipFile(self.vtk_path.with_suffix('.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

        # "phases" list provided in yml config file, CONRAD_NUM_PHASE = 10 (from constants.py) 
        # make sure each of the provided phases < CONRAD_NUM_PHASES
        # since there are only 10 phases (i,e 10 timesteps) of a cardiac cycle (heartbeat) which are used by CONRAD
        # the heart is split into its different parts (left ventricle, right ventricle, etc...) called heart components
        # Recall CONRAD:
        # For each heart component c and for each timestep t (i,e each phase p), CONRAD extracts 16 principal components
        # aka modes (of variation), which capture well the variability of that 3D heart component (e,g left ventricle)
        # in the dataset (20 subjects: 9 male, 11 female)
        # These 16 principal components can then be linearly combined to produce a new 3D heart component (e,g new left ventricle)
        # Also, a mean shape (mean of all 20 subjects) is provided
        # In the vtk polys files: the first vertices listed represent the mean shape vertices, then the triangles of this mean
        # shape are listed then the 16 principal components
        # using the mean shape and the 16 principal components, we can linearly combine the PCs (using 16 weights)
        # each combination of weights give a new shape
        # Note that CONRAD extracts 16 PCs along with their 16 corresponding eigenvalues, these are saved 
        # in csv file of the same name, for each phase p and each heart component c
        assert all([p < CONRAD_NUM_PHASE for p in phases])
        # "components" list provided in yml config file
        # CONRAD_COMPONENTS = {"aorta", "leftAtrium", "leftVentricle", "myocardium", "rightAtrium", "rightVentricle"}
        # make sure each of the provided components is in CONRAD_COMPONENTS list
        assert all([c in CONRAD_COMPONENTS for c in components])

        # save in self
        self.components = components
        self.phases = list(sorted(phases))
        self.train_interpolation = train_interpolation
        self.test_interpolation = test_interpolation

        # from constants.py, CONRAD_MEAN_FILE = "mean_phase_{}_{}.{}"
        # for each phase p, for each component "c" create a path in the form:
        # self.vtk_path/c/mean_phase_{p}_{c}.vtk
        # e,g for p = 0 and c = "leftVentricle", the path is:
        # repo/heart_mesh/shape_models/CardiacModel/vtkPolys/leftVentricle/mean_phase_0_leftVentricle.vtk
        mean_paths = [[self.vtk_path / c / CONRAD_MEAN_FILE.format(p, c, 'vtk')
                       for c in components]
                      for p in phases]
        # exactly same as mean_paths, except each path has .csv instead of .vtk at the end
        var_paths = [[self.vtk_path / c / CONRAD_MEAN_FILE.format(p, c, 'csv')
                      for c in components]
                     for p in phases]

        # Load paths i,e load vtk polys (mean shape of each phase p of each component c and the 16 PCs)
        # and csv data files (16 eigenvalues) from above 2 lists
        mean_polys = [[load_polydata(c) for c in comp_list]
                      for comp_list in mean_paths]
        p_eigens = [[np.genfromtxt(str(p)) for p in comp_list]
                    for comp_list in var_paths]

        self.mean_polys = mean_polys # save list of "mean poly + 16 PCs"
        self.p_eigens = p_eigens # save list of eigenvalues, here p_eigens[p][c] is the list of 16 eigenvalues
                                 # corresponding to component c at phase p
                                 # The corresponding poly (i,e mean shape + 16 PCs) are saved in mean_polys[p][c]
                                 # so mean_polys[p][c] is the poly (i,e mean shape + 16 PCs) of heart component c
                                 # at phase p
                                 # Note: each PC is a Matrix of same dimensions as the mean shape
                                 # i,e dim(PC) = nb_vertices x 3 (3 for 3D)

        # For each point/vertex, add the eigenvalue of the mode as point data (also add component nb as point data array). This
        # will help keeping them after the merging of polys
        for j, comp_list in enumerate(mean_polys): # for each phase j
            for i, poly_data in enumerate(comp_list): # for each component i
                eigs = p_eigens[j][i] # get 16 eigenvalues of component i at phase j
                n_points = poly_data.GetNumberOfPoints() # number of vertices in the poly
                for mode, v in enumerate(eigs): # for each eigenvalue
                    attr = numpy_to_vtk(
                        np.repeat(v, n_points).astype(np.float)) # create array with eigenvalue repeated n_points = nb_vertices times
                    attr.SetName(f"eig_mode_{mode + 1}") # name the array
                    poly_data.GetPointData().AddArray(attr) # save the array in the corresponding poly, the array is saved as
                                                            # point data array in the PointData object
                                                            # at the end of this for-loop, each poly is composed of
                                                            # mean shape + 16 PCs + 16 arrays of eigenvalue (repeated n_points times)

                component_nb = CONRAD_COMPONENTS_TO_IDS[components[i]] # get the index nb of the i'th component
                attr = numpy_to_vtk(
                    np.repeat(component_nb, n_points).astype(np.float)) # create array with component index repeated n_points times
                attr.SetName(f"components") # name the array
                poly_data.GetPointData().AddArray(attr) # save array to poly object
                poly_data.GetPointData().SetActiveScalars(f"components")

        # ---------------------- CONRAD dynamic model params (not using dynamic model) -------------------------
        model_path = self.vtk_path / CONRAD_PARAM_FILE.format('vtk')
        eigen_path = self.vtk_path / CONRAD_PARAM_FILE.format('csv')
        model = load_polydata(model_path) # used to get self.rohs (dynamic modes of shape (160, 1)) and self.beta_bar

        if dynamic_modes is None: # when None, use all 16 modes
            self.dynamic_modes = np.array(range(16))
        else: # otherwise, use only listed modes
            self.dynamic_modes = np.sort(dynamic_modes)
        self.dynamic_eigenvalues = np.sqrt(np.genfromtxt(str(eigen_path)))
        self.beta_bar = vtk_to_numpy(model.GetPoints().GetData())[:, 0]  # 1st dim  (all dim are the same)

        # for the conrad model we vary the dynamic modes and keep the spatial modes fixed
        self.n_rand_coeff = len(self.dynamic_modes)

        # the dynamic modes
        self.rohs = []
        for mode in range(16):
            mode_str = f'mode_{mode + 1}'
            pc_i = vtk_to_numpy(model.GetPointData().GetArray(mode_str))
            self.rohs.append(pc_i)

        # For each phase, merge all heart components to a single polydata
        # self._mean_polydata is the merging of the all components "self.mean_polys" for each phase
        self._mean_polydata = [merge_polys(c) for c in mean_polys]

        # ------------------------------------------------------------------------------------------------------

        # add data set specific parameters
        # i,e save params in self
        self.params["components"] = components
        self.params["phases"] = phases
        self.params["modes"] = self.modes.tolist()
        self.params["dynamic_modes"] = self.dynamic_modes.tolist()
        self.params["train_interpolation"] = train_interpolation
        self.params["test_interpolation"] = test_interpolation
        self.params["time_per_cycle_mean"] = time_per_cycle_mean
        self.params["time_per_cycle_std"] = time_per_cycle_std
        self.params["nb_cycles_mean"] = nb_cycles_mean
        self.params["nb_cycles_std"] = nb_cycles_std
        self.params["shift_max"] = shift_max
        self.params["shapes_per_cycle_mean"] = shapes_per_cycle_mean
        self.params["shapes_per_cycle_std"] = shapes_per_cycle_std
        self.params["pulse_min"] = pulse_min
        self.params["pulse_max"] = pulse_max

    def _get_mean_polydata(self, **kwargs):
        time = kwargs['time']
        is_train = kwargs.get('is_train', True)
        interp = (self.train_interpolation if is_train
                  else self.test_interpolation)
        if interp == 'linear':
            return self._evaluate_linear_interpolation(self._mean_polydata,
                                                       time=time)
        elif interp == 'spline':
            return self._compute_and_evaluate_cubic_spline(self._mean_polydata,
                                                           time)

    def _generate_sample(self, time, std_num, **kwargs):
        """
            'std_num' are the random coefficients for the dynamic modes.
            'modes' are the spatial modes
        """
        
        # the mode indices
        modes = kwargs.get('modes', self.modes)
        dynamic_modes = kwargs.get('dynamic_modes', self.dynamic_modes)

        # these are the deltas, here "dynamic_coeffs" is None since "dynamic_coeffs" not passed to this function
        dynamic_coeffs = kwargs.get('dynamic_coeffs', None)

        if not isinstance(modes, np.ndarray):
            modes = np.array([m for m in modes])

        if not isinstance(dynamic_modes, np.ndarray): # convert list of the dynamic modes indices to numpy
            dynamic_modes = np.array([dm for dm in dynamic_modes])

        modes = np.sort(modes)
        dynamic_modes = np.sort(dynamic_modes)

        if not isinstance(std_num, np.ndarray): # not executed since std_num is numpy array (randomly uniformly sampled)
            std_num = np.ones(dynamic_modes.shape) * std_num

        # random coefficient for dynamic modes
        assert len(std_num) == len(dynamic_modes)

        # build dynamic weights
        dynamic_weights = np.zeros_like(self.beta_bar)

        if dynamic_coeffs is None: # if not passed dynamic_coeffs to this function, use "std_num[i] * self.dynamic_eigenvalues[mode_i]" as coeffs
            for i, d_mode_idx in enumerate(dynamic_modes): # for each dynamic mode index "mode_i" = "d_mode_idx"
                dynamic_weights += self.rohs[d_mode_idx] * std_num[i] * self.dynamic_eigenvalues[d_mode_idx]
        else: # else use the passed dynamic coeffs
            for i, d_mode_idx in enumerate(dynamic_modes):
                dynamic_weights += self.rohs[d_mode_idx] * dynamic_coeffs[d_mode_idx]

        dynamic_weights += self.beta_bar  # add mean
        dynamic_weights = [p for p in np.reshape(dynamic_weights, (10, 16))]

        is_train = kwargs.get('is_train', True)
        interp = (self.train_interpolation if is_train
                  else self.test_interpolation)
        if interp == 'linear':
            # If linear interpolation, compute only the two relevant phases
            y_1, y_2, diff = self._get_y1_y2(len(self._mean_polydata), time)

            time = diff
            if diff != 0:
                mean_polys = [self._mean_polydata[y_1],
                              self._mean_polydata[y_2]]
                dynamic_weights = [dynamic_weights[y_1], dynamic_weights[y_2]]
            else:
                mean_polys = [self._mean_polydata[y_1]]
                dynamic_weights = [dynamic_weights[y_1]]
        else:
            # Otherwise we may need all phases
            mean_polys = self._mean_polydata
        
        # we have p phases for the mean poly in our data, we take 2 phases: one before time t (phase p1) and one after time t (phase p2)
        # we use the mean poly at phase p1 to generate the new poly at phase p1 (using PCs and eigenvalues of mean poly at phase p1
        # (i,e the modes) as well as the weights of the new poly). Likewise for the new poly at phase p2.
        # We then interpolate (linear or spline) between the 2 generated polys to get the poly at time t (p1 <= t <= p2)
        
        generated_phases = [] # array of generated polys
        for i, mean_phase in enumerate(mean_polys): # for each mean poly at a certain phase, generate the corresponding new poly at the same phase
            pcs = []
            eigens = []
            for mode in modes: # get the PCs and eigenvalues
                pc_i, eigv = self.get_mode(mean_phase, mode)
                pcs.append(pc_i)
                eigens.append(eigv)
            
            # generate new poly at same phase as the current mean poly (using PCs, eigenvalues and "weights of new poly")
            weights = np.take(dynamic_weights[i], modes)
            poly = generate_shape_from_modes(mean_polydata=mean_phase,
                                             pcs=pcs,
                                             eigens=eigens,
                                             weights=weights)
            generated_phases.append(poly) # append to list of generated polys
        
        if interp == 'linear': # interpolate between polys
            return self._evaluate_linear_interpolation(generated_phases,
                                                       diff=time)
        elif interp == 'spline':
            return self._compute_and_evaluate_cubic_spline(generated_phases,
                                                           time)

    # used in method _generate_sample() above
    @staticmethod
    def get_mode(polydata, mode):
        mode = mode + 1
        n_points = polydata.GetNumberOfPoints()
        pc_i = polydata.GetPointData().GetArray(f'mode_{mode}')
        var_i = polydata.GetPointData().GetArray(f'eig_mode_{mode}')
        # all elements of `var_i` must be the same
        std = np.sqrt(float(vtk_to_numpy(var_i)[0]))
        pc_i = np.reshape(vtk_to_numpy(pc_i), (n_points, 3))
        return pc_i, std
    
    # perform cubic spline interpolation between 2 polydatas
    @staticmethod
    def _compute_and_evaluate_cubic_spline(polys, time):
        if len(polys) == 1:  # No interpolation if only one phase is given
            return polys[0]

        assert 0 <= time < 1
        apprx = 0.001

        xs = np.linspace(0.0, 1.0, num=len(polys))

        idx = np.where(np.abs(xs - time) < apprx)[0]
        if len(idx) > 0:
            return polys[idx[0]]

        points_list = np.array([vtk_to_numpy(p.GetPoints().GetData())
                                for p in polys])
        points = np.transpose(points_list, [1, 0, 2])
        css = [CubicSpline(xs, all_phase, axis=0)
               for all_phase in points]

        points = [cs(time) for cs in css]
        interpolated = overwrite_vtkpoly(polys[0], points=points,
                                         point_scalar="components")
        return interpolated

    # the indices of the polydata to use for interpolation given that we have n_phases (i,e n_phases polydata)
    # and we want to generate the shape at time "time"
    @staticmethod
    def _get_y1_y2(n_phases, time):
        assert 0 <= time < 1

        y_1 = int(np.floor(time * n_phases))
        y_2 = int((y_1 + 1) % n_phases)
        diff = time * n_phases - y_1
        return y_1, y_2, diff
    
    @staticmethod
    def remap_time(time):
        # if time not in [0, 1), shift it to be in [0, 1) 
        if time < 0:
            temp = time - int(time)
            if time != int(time):
                temp += 1
            time = temp
        elif time >= 1:
            time = time - int(time)
        return time
    
    # perform linear interpolation between 2 polydatas
    def _evaluate_linear_interpolation(self, polys, time=None, diff=None):
        """Return a vtk.vtkPolyData which the result of interpolation from
        polydata in `polys`.

        Based on `time` which is between 0 and 1, first it finds two
        polydata `y1` and `y2` before and after the time point,
        then interpolates those two polydata.

        if `diff` is given instead of `time`, it supposes that both `y1` and
        `y2` are already provided.
        """
        assert (time is not None) or (diff is not None)

        n_polys = len(polys) # nb phase, one poly per phase (recall 10 phases)
        if time is not None:
            assert 0 <= time <= 1
            y1, y2, diff = self._get_y1_y2(n_polys, time)
        else:
            assert n_polys <= 2
            y1, y2 = 0, 1

        if diff == 0:
            return polys[y1]

        polydata_1 = polys[y1]
        polydata_2 = polys[y2]
        interpolated = vtkpoly_linearly_interpolated(polydata_1,
                                                     polydata_2,
                                                     diff)
        #  Note: Here we get the polys of the first polydata
        interpolated.SetPolys(polydata_1.GetPolys())
        interpolated.GetPointData().SetScalars(
            polydata_1.GetPointData().GetScalars())
        return interpolated

    # not used at the moment
    def get_modes(self) -> List[np.array]:
        """
        Modes of each component of the mesh
        :return: [(phase, mode, n_points, d) for each component]
        """

        def group_component(component):
            """
            Group all phases and modes into a single array for a given component
            :param component: id of the component to group
            :return: np.array of shape (phases, modes, n_points, d) with d=3
            """
            nr_modes = len(self.modes)
            ans = []
            for phase in self.phases:
                ans.append([vtk_to_numpy(self.mean_polys[phase][component].GetPointData().GetArray(f'mode_{m + 1}'))
                            for m in range(nr_modes)])
            return np.stack(ans)

        modes = [group_component(component)
                 for component in range(len(self.components))]
        return modes

    # not used at the moment
    def get_variances(self) -> List[np.array]:
        """
        Variances of each component of the mesh
        :return: [(phase, mode) for each component]
        """
        variances = [np.stack([[self.p_eigens[phase][component][m]
                                for m in range(16)] for phase in self.phases])
                     for component in range(len(self.components))]
        return variances
# ---------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------- CONRAD dataset ----------------------------------------------------
class CONRADData(MeshDataset):
    def __init__(self, data_dir, results_dir, components, phases,
                 dynamic_modes,
                 train_interpolation,
                 test_interpolation,
                 mesh_reduction, batch_size, n_prefetch, save_files,
                 std_shape_generation, shuffle_buffer, ds_factors, **kwargs):

        super().__init__(n_modes=16, # using all 16 eigenvectors/modes
                         data_dir=data_dir, # data_dir = repo/heart_mesh/shape_models
                         results_dir=results_dir, # results_dir = repo/experiments
                         # params from config file, explained in config file
                         save_files=save_files,
                         mesh_reduction=mesh_reduction,
                         batch_size=batch_size,
                         n_prefetch=n_prefetch,
                         ds_factors=ds_factors,
                         shuffle_buffer=shuffle_buffer,
                         std_shape_generation=std_shape_generation,
                         **kwargs)
        # "data_dir = repo/heart_mesh/shape_models" so "data_dir.name = shape_models"
        if data_dir.name == CONRAD_FOLDER: # from constants.py: CONRAD_FOLDER = CardiacModel
            pass
        elif data_dir.name == CONRAD_VTK_FOLDER: # from constants.py: CONRAD_VTK_FOLDER = "vtkPolys"
            self.data_dir = data_dir.parent
        else: # ------> Executed
            self.data_dir = data_dir / CONRAD_FOLDER # self.data_dir = repo/heart_mesh/shape_models/CardiacModel
        self.vtk_path = self.data_dir / CONRAD_VTK_FOLDER # self.data_dir = repo/heart_mesh/shape_models/CardiacModel/vtkPolys
        if not self.vtk_path.exists(): # vtkPolys folder exists (otherwise create it by unzipping vtkPolys zip file in self.data_dir)
            logger.debug(f"Unzipping Conrad data to {self.vtk_path}")
            with ZipFile(self.vtk_path.with_suffix('.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

        # "phases" list provided in yml config file, CONRAD_NUM_PHASE = 10 (from constants.py) 
        # make sure each of the provided phases < CONRAD_NUM_PHASES
        # since there are only 10 phases (i,e 10 timesteps) of a cardiac cycle (heartbeat) which are used by CONRAD
        # the heart is split into its different parts (left ventricle, right ventricle, etc...) called heart components
        # Recall CONRAD:
        # For each heart component c and for each timestep t (i,e each phase p), CONRAD extracts 16 principal components
        # aka modes (of variation), which capture well the variability of that 3D heart component (e,g left ventricle)
        # in the dataset (20 subjects: 9 male, 11 female)
        # These 16 principal components can then be linearly combined to produce a new 3D heart component (e,g new left ventricle)
        # Also, a mean shape (mean of all 20 subjects) is provided
        # In the vtk polys files: the first vertices listed represent the mean shape vertices, then the triangles of this mean
        # shape are listed then the 16 principal components
        # using the mean shape and the 16 principal components, we can linearly combine the PCs (using 16 weights)
        # each combination of weights give a new shape
        # Note that CONRAD extracts 16 PCs along with their 16 corresponding eigenvalues, these are saved 
        # in csv file of the same name, for each phase p and each heart component c
        assert all([p < CONRAD_NUM_PHASE for p in phases])
        # "components" list provided in yml config file
        # CONRAD_COMPONENTS = {"aorta", "leftAtrium", "leftVentricle", "myocardium", "rightAtrium", "rightVentricle"}
        # make sure each of the provided components is in CONRAD_COMPONENTS list
        assert all([c in CONRAD_COMPONENTS for c in components])

        self.components = components # save in object
        self.phases = list(sorted(phases)) # save in object sorted
        self.train_interpolation = train_interpolation # save in object
        self.test_interpolation = test_interpolation # save in object

        # from constants.py, CONRAD_MEAN_FILE = "mean_phase_{}_{}.{}"
        # for each phase p, for each component "c" create a path in the form:
        # self.vtk_path/c/mean_phase_{p}_{c}.vtk
        # e,g for p = 0 and c = "leftVentricle", the path is:
        # repo/heart_mesh/shape_models/CardiacModel/vtkPolys/leftVentricle/mean_phase_0_leftVentricle.vtk
        mean_paths = [[self.vtk_path / c / CONRAD_MEAN_FILE.format(p, c, 'vtk')
                       for c in components]
                      for p in phases]
        # exactly same as mean_paths, except each path has .csv instead of .vtk at the end
        var_paths = [[self.vtk_path / c / CONRAD_MEAN_FILE.format(p, c, 'csv')
                      for c in components]
                     for p in phases]

        # Load paths i,e load vtk polys (mean shape of each phase p of each component c and the 16 PCs)
        # and csv data files (16 eigenvalues) from above 2 lists
        mean_polys = [[load_polydata(c) for c in comp_list]
                      for comp_list in mean_paths]
        p_eigens = [[np.genfromtxt(str(p)) for p in comp_list]
                    for comp_list in var_paths]

        self.mean_polys = mean_polys # save list of "mean poly + 16 PCs"
        self.p_eigens = p_eigens # save list of eigenvalues, here p_eigens[p][c] is the list of 16 eigenvalues
                                 # corresponding to component c at phase p
                                 # The corresponding poly (i,e mean shape + 16 PCs) are saved in mean_polys[p][c]
                                 # so mean_polys[p][c] is the poly (i,e mean shape + 16 PCs) of heart component c
                                 # at phase p
                                 # Note: each PC is a Matrix of same dimensions as the mean shape
                                 # i,e dim(PC) = nb_vertices x 3 (3 for 3D)

        # For each point/vertex, add the eigenvalue of the mode as point data (also add component nb as point data). This
        # will help keeping them after the merging of polys
        for j, comp_list in enumerate(mean_polys): # for each phase j
            for i, poly_data in enumerate(comp_list): # for each component i
                eigs = p_eigens[j][i] # get 16 eigenvalues of component i at phase j
                n_points = poly_data.GetNumberOfPoints() # number of vertices in the poly
                for mode, v in enumerate(eigs): # for each eigenvalue
                    attr = numpy_to_vtk(
                        np.repeat(v, n_points).astype(np.float)) # create array with eigenvalue repeated n_points = nb_vertices times
                    attr.SetName(f"eig_mode_{mode + 1}") # name the array
                    poly_data.GetPointData().AddArray(attr) # save the array in the corresponding poly, the array is saved as
                                                            # point data array in the PointData object
                                                            # at the end of this for-loop, each poly is composed of
                                                            # mean shape + 16 PCs + 16 arrays of eigenvalue (repeated n_points times)
                
                component_nb = CONRAD_COMPONENTS_TO_IDS[components[i]] # get the index nb of the i'th component
                attr = numpy_to_vtk(
                    np.repeat(component_nb, n_points).astype(np.float)) # create array with component number "i" repeated n_points times
                attr.SetName(f"components") # name the array
                poly_data.GetPointData().AddArray(attr) # save array to poly object
                poly_data.GetPointData().SetActiveScalars(f"components")

        # ---------------------- CONRAD dynamic model params -------------------------
        model_path = self.vtk_path / CONRAD_PARAM_FILE.format('vtk')
        eigen_path = self.vtk_path / CONRAD_PARAM_FILE.format('csv')
        model = load_polydata(model_path)

        if dynamic_modes is None: # when None, use all 16 modes
            self.dynamic_modes = np.array(range(16))
        else: # otherwise, use only listed modes
            self.dynamic_modes = np.sort(dynamic_modes)
        self.dynamic_eigenvalues = np.sqrt(np.genfromtxt(str(eigen_path)))
        self.beta_bar = vtk_to_numpy(model.GetPoints().GetData())[:, 0]  # 1st dim  (all dim are the same), dimension 160

        # for the conrad model we vary the dynamic modes and keep the spatial modes fixed
        self.n_rand_coeff = len(self.dynamic_modes)

        # the dynamic modes, each one had dimension 160, one per component per phase
        self.rohs = []
        for mode in range(16):
            mode_str = f'mode_{mode + 1}'
            pc_i = vtk_to_numpy(model.GetPointData().GetArray(mode_str))
            self.rohs.append(pc_i)

        # For each phase, merge all heart components to a single polydata
        self._mean_polydata = [merge_polys(c) for c in mean_polys]

        # ------------------------------------------------------------------------------------------------------

        # add data set specific parameters
        # i,e save the params passed to the constructor
        self.params["components"] = components
        self.params["phases"] = phases
        self.params["modes"] = self.modes.tolist()
        self.params["dynamic_modes"] = self.dynamic_modes.tolist()
        self.params["train_interpolation"] = train_interpolation
        self.params["test_interpolation"] = test_interpolation

    def _get_mean_polydata(self, **kwargs):
        time = kwargs['time']
        is_train = kwargs.get('is_train', True)
        interp = (self.train_interpolation if is_train
                  else self.test_interpolation)
        if interp == 'linear':
            return self._evaluate_linear_interpolation(self._mean_polydata,
                                                       time=time)
        elif interp == 'spline':
            return self._compute_and_evaluate_cubic_spline(self._mean_polydata,
                                                           time)

    def _generate_sample(self, time, std_num, **kwargs):
        """
            'std_num' are the random coefficients for the dynamic modes.
            'modes' are the spatial modes
        """

        # the mode indices
        modes = kwargs.get('modes', self.modes)
        dynamic_modes = kwargs.get('dynamic_modes', self.dynamic_modes)

        # these are the deltas
        dynamic_coeffs = kwargs.get('dynamic_coeffs', None)

        if not isinstance(modes, np.ndarray):
            modes = np.array([m for m in modes])

        if not isinstance(dynamic_modes, np.ndarray):
            dynamic_modes = np.array([dm for dm in dynamic_modes])

        modes = np.sort(modes)
        dynamic_modes = np.sort(dynamic_modes)

        if not isinstance(std_num, np.ndarray):
            std_num = np.ones(dynamic_modes.shape) * std_num

        # random coefficient for dynamic modes
        assert len(std_num) == len(dynamic_modes)

        # build dynamic coefficients
        dynamic_weights = np.zeros_like(self.beta_bar)

        if dynamic_coeffs is None:
            for i, d_mode_idx in enumerate(dynamic_modes):
                dynamic_weights += self.rohs[d_mode_idx] * std_num[i] * self.dynamic_eigenvalues[d_mode_idx]
        else:
            for i, d_mode_idx in enumerate(dynamic_modes):
                dynamic_weights += self.rohs[d_mode_idx] * dynamic_coeffs[d_mode_idx]

        dynamic_weights += self.beta_bar  # add mean
        dynamic_weights = [p for p in np.reshape(dynamic_weights, (10, 16))]

        is_train = kwargs.get('is_train', True)
        interp = (self.train_interpolation if is_train
                  else self.test_interpolation)
        if interp == 'linear':
            # If linear interpolation, compute only the two relevant phases
            y_1, y_2, diff = self._get_y1_y2(len(self._mean_polydata), time)

            time = diff
            if diff != 0:
                mean_polys = [self._mean_polydata[y_1],
                              self._mean_polydata[y_2]]
                dynamic_weights = [dynamic_weights[y_1], dynamic_weights[y_2]]
            else:
                mean_polys = [self._mean_polydata[y_1]]
                dynamic_weights = [dynamic_weights[y_1]]
        else:
            # Otherwise we may need all phases
            mean_polys = self._mean_polydata

        generated_phases = [] # list of generated polys at the phases before and after the "time" argument (used to interpolate),
                              # polys generated using mean polys at the same phases + weights for each PC mode
        for i, mean_phase in enumerate(mean_polys): # 
            pcs = []
            eigens = []
            for mode in modes: # 16 modes (or less if specified in the config file), so modes is a list of indices [0, ..., 15]
                pc_i, eigv = self.get_mode(mean_phase, mode)
                pcs.append(pc_i)
                eigens.append(eigv)

            weights = np.take(dynamic_weights[i], modes) # take 1 dynamic weight per mode.
            poly = generate_shape_from_modes(mean_polydata=mean_phase,
                                             pcs=pcs,
                                             eigens=eigens,
                                             weights=weights)
            generated_phases.append(poly)

        if interp == 'linear':
            return self._evaluate_linear_interpolation(generated_phases,
                                                       diff=time)
        elif interp == 'spline':
            return self._compute_and_evaluate_cubic_spline(generated_phases,
                                                           time)

    @staticmethod
    def _compute_and_evaluate_cubic_spline(polys, time):
        if len(polys) == 1:  # No interpolation if only one phase is given
            return polys[0]

        assert 0 <= time < 1
        apprx = 0.001

        xs = np.linspace(0.0, 1.0, num=len(polys))

        idx = np.where(np.abs(xs - time) < apprx)[0]
        if len(idx) > 0:
            return polys[idx[0]]

        points_list = np.array([vtk_to_numpy(p.GetPoints().GetData())
                                for p in polys])
        points = np.transpose(points_list, [1, 0, 2])
        css = [CubicSpline(xs, all_phase, axis=0)
               for all_phase in points]

        points = [cs(time) for cs in css]
        interpolated = overwrite_vtkpoly(polys[0], points=points,
                                         point_scalar="components")
        return interpolated

    @staticmethod
    def _get_y1_y2(n_phases, time):
        assert 0 <= time < 1

        y_1 = int(np.floor(time * n_phases))
        y_2 = int((y_1 + 1) % n_phases)
        diff = time * n_phases - y_1
        return y_1, y_2, diff

    def _evaluate_linear_interpolation(self, polys, time=None, diff=None):
        """Return a vtk.vtkPolyData which the result of interpolation from
        polydata in `polys`.

        Based on `time` which is between 0 and 1, first it finds two
        polydata `y1` and `y2` before and after the time point,
        then interpolates those two polydata.

        if `diff` is given instead of `time`, it supposes that both `y1` and
        `y2` are already provided.
        """
        assert (time is not None) or (diff is not None)

        n_polys = len(polys) # nb phase, one poly per phase (recall 10 phases)
        if time is not None:
            assert 0 <= time <= 1
            y1, y2, diff = self._get_y1_y2(n_polys, time)
        else:
            assert n_polys <= 2
            y1, y2 = 0, 1

        if diff == 0:
            return polys[y1]

        polydata_1 = polys[y1]
        polydata_2 = polys[y2]
        interpolated = vtkpoly_linearly_interpolated(polydata_1,
                                                     polydata_2,
                                                     diff)
        #  Note: Here we get the polys of the first polydata
        # so generated poly has same adjacency matrix as the source data (mean shapes)
        interpolated.SetPolys(polydata_1.GetPolys())
        interpolated.GetPointData().SetScalars(
            polydata_1.GetPointData().GetScalars())
        return interpolated

    @staticmethod
    def get_mode(polydata, mode):
        mode = mode + 1
        n_points = polydata.GetNumberOfPoints()
        pc_i = polydata.GetPointData().GetArray(f'mode_{mode}')
        var_i = polydata.GetPointData().GetArray(f'eig_mode_{mode}')
        # all elements of `var_i` must be the same
        std = np.sqrt(float(vtk_to_numpy(var_i)[0]))
        pc_i = np.reshape(vtk_to_numpy(pc_i), (n_points, 3))
        return pc_i, std

    def get_modes(self) -> List[np.array]:
        """
        Modes of each component of the mesh
        :return: [(phase, mode, n_points, d) for each component]
        """

        def group_component(component):
            """
            Group all phases and modes into a single array for a given component
            :param component: id of the component to group
            :return: np.array of shape (phases, modes, n_points, d) with d=3
            """
            nr_modes = len(self.modes)
            ans = []
            for phase in self.phases:
                ans.append([vtk_to_numpy(self.mean_polys[phase][component].GetPointData().GetArray(f'mode_{m + 1}'))
                            for m in range(nr_modes)])
            return np.stack(ans)

        modes = [group_component(component)
                 for component in range(len(self.components))]
        return modes

    def get_variances(self) -> List[np.array]:
        """
        Variances of each component of the mesh
        :return: [(phase, mode) for each component]
        """
        variances = [np.stack([[self.p_eigens[phase][component][m]
                                for m in range(16)] for phase in self.phases])
                     for component in range(len(self.components))]
        return variances
# ---------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------- Other datasets ---------------------------------------------------
class UKHeartData(MeshDataset):
    def __init__(self, data_dir, results_dir,
                 data_shape, mesh_reduction, batch_size,
                 n_prefetch, ds_factors, save_files,
                 shuffle_buffer, std_shape_generation, **kwargs):

        super().__init__(n_modes=100,
                         data_dir=data_dir,
                         results_dir=results_dir,
                         save_files=save_files,
                         mesh_reduction=mesh_reduction,
                         batch_size=batch_size,
                         n_prefetch=n_prefetch,
                         ds_factors=ds_factors,
                         shuffle_buffer=shuffle_buffer,
                         std_shape_generation=std_shape_generation,
                         **kwargs)

        if data_dir.name != UK_FOLDER:
            self.data_dir = data_dir / UK_FOLDER
        self.mean_path = self.data_dir / f"{data_shape}_ED_mean.vtk"
        self.pc_path = self.data_dir / f"{data_shape}_ED_pc_100_modes.csv.gz"
        self.var_path = self.data_dir / f"{data_shape}_ED_var_100_modes.csv.gz"
        self.epiendo = kwargs.get('epiendo', None)

        # Read the mean data
        polydata = load_polydata(self.mean_path)
        if self.epiendo:
            logger.debug(f"Using subgraph {self.epiendo} for data generation")
            graph = poly_to_nxgraph(polydata)
            sub_nodes = [node for node in graph.nodes if
                         graph.nodes[node]['scalars'] == self.epiendo]
            sub_graph = get_subgraph(graph, sub_nodes)
            polydata = nxgraph_to_vtkpoly(sub_graph, point_scalar='scalars')

        self._mean_polydata = polydata
        self.n_points = self._get_mean_polydata().GetNumberOfPoints()

        # Read the principal component and variance
        self.pcs = np.genfromtxt(str(self.pc_path), delimiter=',')
        self.variances = np.genfromtxt(str(self.var_path), delimiter=',')

        self.params["epiendo"] = self.epiendo
        self.params["data_shape"] = data_shape

    def _get_mean_polydata(self, **kwargs):
        return self._mean_polydata

    def get_mode(self, mode):
        pc_i = self.pcs[:, mode]
        pc_i = np.reshape(pc_i, (self.n_points, 3))
        std = np.sqrt(self.variances[mode])
        return pc_i, std

    def _generate_sample(self, time, std_num, **kwargs):

        modes = kwargs.get('modes', self.modes)

        if not isinstance(modes, np.ndarray):
            modes = np.array([modes])

        pcs = []
        eigens = []
        for mode in modes:
            pc_i, eigv = self.get_mode(mode)
            pcs.append(pc_i)
            eigens.append(eigv)

        weights = np.ones((len(modes))) * std_num
        poly = generate_shape_from_modes(mean_polydata=self._get_mean_polydata(),
                                         pcs=pcs,
                                         eigens=eigens,
                                         weights=weights)
        return poly

class CISTIBData(MeshDataset):
    def __init__(self, data_dir, results_dir,
                 data_shape, mesh_reduction, batch_size,
                 n_prefetch, std_shape_generation, save_files,
                 shuffle_buffer, ds_factors, epiendo, **kwargs):

        self.data_shape = data_shape
        self.n_modes = 50 if self.data_shape == 'Full' else 34
        super().__init__(n_modes=self.n_modes,
                         data_dir=data_dir,
                         results_dir=results_dir,
                         save_files=save_files,
                         mesh_reduction=mesh_reduction,
                         batch_size=batch_size,
                         n_prefetch=n_prefetch,
                         ds_factors=ds_factors,
                         shuffle_buffer=shuffle_buffer,
                         std_shape_generation=std_shape_generation,
                         **kwargs)

        sub_folder = CISTIB_ALL_FOLDER.format(data_shape)
        if data_dir.name == sub_folder:
            pass
        elif data_dir.name == CISTIB_FOLDER:
            self.data_dir = data_dir / sub_folder
        else:
            self.data_dir = data_dir / CISTIB_FOLDER / sub_folder

        self.mean_path = self.data_dir / f"PCAmodel_{data_shape}" \
                                         f"_phaseAll_90pct_proc05it.vtp"
        self.var_path = self.data_dir / f"PCAmodel_{data_shape}" \
                                        f"_phaseAll_90pct_proc05it_Lambdas.txt"

        self.epiendo = epiendo

        # Read the mean data
        polydata = load_polydata(self.mean_path)
        if self.epiendo:
            logger.debug(f"Using subgraph {self.epiendo} for data generation")
            ep_val = CISTIB_EPIENDO[self.epiendo]

            graph = poly_to_nxgraph(polydata)
            sub_nodes = [node for node in graph.nodes if
                         graph.nodes[node]['epiendo'] == ep_val]
            sub_graph = get_subgraph(graph, sub_nodes)
            polydata = nxgraph_to_vtkpoly(sub_graph)

        self._mean_polydata = polydata
        self.n_points = self._get_mean_polydata().GetNumberOfPoints()
        self.variances = open(self.var_path, 'r').readlines()

        self.params["epiendo"] = self.epiendo
        self.params["n_modes"] = self.n_modes
        self.params["data_shape"] = data_shape

    def _get_mean_polydata(self, **kwargs):
        return self._mean_polydata

    def get_mode(self, mode):
        pc_i = self._get_mean_polydata().GetPointData().GetArray(
            f'mode_{mode + 1:02d}')
        std = np.sqrt(float(self.variances[mode]))
        pc_i = np.reshape(vtk_to_numpy(pc_i), (self.n_points, 3))
        return pc_i, std

    def _generate_sample(self, std_num, **kwargs):

        modes = kwargs.get('modes', self.modes)

        if not isinstance(modes, np.ndarray):
            modes = np.array([modes])

        pcs = []
        eigens = []
        for mode in modes:
            pc_i, eigv = self.get_mode(mode)
            pcs.append(pc_i)
            eigens.append(eigv)

        weights = std_num
        if std_num is None:
            weights = np.random.uniform(self.data_std[0], self.data_std[1], len(modes))

        assert len(weights) == len(modes)

        poly = generate_shape_from_modes(mean_polydata=self._get_mean_polydata(),
                                         pcs=pcs,
                                         eigens=eigens,
                                         weights=weights)
        return poly
# ---------------------------------------------------------------------------------------------------------------------------


