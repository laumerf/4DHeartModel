import csv
import time
import numpy as np
import os, sys, time
import pandas as pd
import scipy.io
import skvideo.io
import glob
import tensorflow as tf
from pathlib import Path

from source.models.echo_dhb import *
from source.constants import ECHO_DATA_PARAMS
import source.utils as utils


# data loader
def load_data_echonet(logger, data_dir):

    """
    EchoNet-Dynamic data.
    Returns a dictionary with a mapping from subject ids to .npz-file paths which contain a frames and a times array.
    """

    # create cache folder (if not exists) to store cached videos for faster loading of data over different experiments
    video_cache_folder = str(data_dir / 'cache/Videos')
    if not os.path.exists(video_cache_folder):
        os.makedirs(video_cache_folder)
    
    # file with data as a pandas series
    data_info = pd.read_csv(str(data_dir / 'FileList.csv'))
    # get filenames of video (without the extension .avi) and add them as a new column "globalID" to the pandas df "data_info"
    data_info['globalID'] = data_info['FileName'].apply(lambda s: s[:-4]).astype('string')
    
    # set the index for rows to be the column "globalID" instead of the row number
    data_info.set_index('globalID', inplace=True)

    files = dict()
    nb_videos = data_info.shape[0]
    logger.info("Reading {} videos...".format(nb_videos))
    i = 0
    # for each row + row index (i,e for each row indexed by the corresponding video filename)
    cached_file_list = glob.glob(str(video_cache_folder) + "/*.npz")
    for index, row in data_info.iterrows():
        i += 1

        if i == 1 or i % 1000 == 0 or i == nb_videos:
            logger.info("Video {}/{}".format(i, nb_videos))
        
        # path to the echo video
        filepath = str(data_dir / 'Videos' / index) + '.avi'
        # path to the cached video data (images of each video stored as numpy array)
        filepath_cached = video_cache_folder + '/' + index + '.npz'

        # if the video data doesn't have a cached version of it, then cache it
        if str(filepath_cached) not in cached_file_list:

            # load from dataset
            frames = skvideo.io.vread(filepath) # video as numpy array of shape (nb_frames x 112 x 112 x 3), where height = width = 112 amd channels = 3
                                                # but nb_frames is different from video to video
            frames = [frame[:, :, 0] for frame in frames] # we only keep the 1st channel to work with (Note: channels are different)            
            # times
            time_base = 1/data_info.loc[index]['FPS'] # time between every 2 frames
            times = [i*time_base for i in range(len(frames))] # sequence of timesteps   
            # params
            params = [row[param] for param in ECHO_DATA_PARAMS]

            # cache data
            np.savez(filepath_cached, frames=frames, times=times, params=params) # save the 3 arrays (frames, times, params) into 1 .npz archive file (uncompressed)
                                                                                 # for each argument key value pairs key=value  (e,g frames=frames), the data "value" is
                                                                                 # saved in a .npy file named "key.npy", all .npy files saved together in the same .npz file
        files[index] = filepath_cached
    
    logger.info("Done reading videos!\n")
    return data_info, files

# TF model loaders
def load_echonet_dynamic_model(weights_path, logger, log_dir=None):

    # initialise model
    model = EchocardioModel(logger=logger, latent_space_dim=128, batch_size=32, hidden_dim=128, log_dir=log_dir)
    # perform forward pass to init Keras weights and get trainable variables (not needed for loading model)
    model((tf.ragged.constant([[0.0]], dtype='float32'), tf.ragged.constant([[np.full((112, 112), 0.5)]], inner_shape=(112, 112), dtype='float32')))

    # load weights
    model.load_weights(weights_path).expect_partial()

    return model

def get_echo_edf_esf(data_dir, logger):

    # if echo EDF/ESF file already exists, read it and return
    echo_edf_esf_file = data_dir / 'Echo_EDF_ESF.csv'
    if echo_edf_esf_file.exists():
        logger.info("Found Echo EDF/ESF file, loading file...")
        echo_edf_esf = pd.read_csv(str(echo_edf_esf_file))
        return echo_edf_esf
    
    # otherwise compute the frame phase infos and save it
    volume_tracings = pd.read_csv(str(data_dir / 'VolumeTracings.csv')) # file with volume tracing data as a pandas df
    
    discarded_echos = ['0X10623D3AF96AC271', '0X10B04432B90E5AC2', '0X1BD7A625C9DA5292', '0X1D3D82FD91F61757', '0X1E433E7966FD7332', '0X20586B1BD35F38C9', '0X22A9B6ECBD591065', '0X22E82C3D081C819C', '0X286661146EB02EE4', '0X2A20552783F445AE', '0X2ABBA31A554D6E2', '0X2D3F2E5FFA807CFF', '0X2D405B452654D053', '0X31C7FC69D6C348EA', '0X33EAE0F44B7618C1', '0X374099556945A9EA', '0X39207AF594BF77D7', '0X39348579B2E55470', '0X3BAE126085E973A0', '0X3F3DC1A6F0B18FA', '0X4A11C148E80CABDE', '0X4F89846030713617', '0X52619FC2739EB1F1', '0X52832AF3B2EE7826', '0X533498EF6F72192', '0X55DD5AB1762EDCDA', '0X5746FD1045390A93', '0X64EE9FFA5DA69058', '0X659BBAF883211018', '0X67F0850A73E98169', '0X6CAFD90BF9854990', '0X6FA8D7C2278B5073', '0X7790F4DB18852455', '0X786686A9B8DE6547', '0X790C871B162806D2', '0X7CA9A912598CA322', '0X7F3ABE82B6992583', '0XBE06F978BB3226D']
    ed_frames = [] # end-diastole frame for each file
    es_frames = [] # end-systole frame for each file
    filenames = [] # names of echo files which were not skipped
    logger.info("Computing echo volumes to specify ED frame and ES frame")
    i = 0
    t_start = time.time()
    for filename, tracing_segs in volume_tracings.groupby("FileName"):
        # skip echo file if specifies more/less than 42 segments
        # or if in list of discarded echos
        if len(tracing_segs) != 42 or filename in discarded_echos:
            continue

        # Option 1: Compute volume using method of disks then decide which frame is EDF (and which one is ESF)
        frame_nbs = []
        frame_vols = []
        for frame_nb, frame_tracings in tracing_segs.groupby("Frame"):
            lengths = []
            for _, row in frame_tracings.iterrows():
                x1, x2, y1, y2 = row["X1"], row["X2"], row["Y1"], row["Y2"]
                length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                lengths.append(length)
            n = len(lengths) - 1
            height = lengths[0] / n # long axis length divided by number of disks to get height
            vol = 0
            for l in lengths[1:]:
                vol += np.pi * (l/2) * (l/2) * height
            frame_vols.append(vol)
            frame_nbs.append(frame_nb)
        
        if frame_vols[0] > frame_vols[1]:
            ed_frames.append(frame_nbs[0])
            es_frames.append(frame_nbs[1])
        else:
            ed_frames.append(frame_nbs[1])
            es_frames.append(frame_nbs[0])
        
        # # Option 2: min frame nb is EDF
        # ed_frames.append(min(frame_nbs))
        # es_frames.append(max(frame_nbs))

        filenames.append(filename)
        
        if i+1 == 1 or (i+1)%1000 == 0:
            t2 = time.time()
            h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
            logger.info(f"Done {i+1} echo videos in {h}:{m}:{s}")

        i+= 1

    echo_edf_esf = pd.DataFrame({'FileName':filenames, 'EDF':ed_frames, 'ESF': es_frames})
    echo_edf_esf.to_csv(str(echo_edf_esf_file), index=False)
    return echo_edf_esf

def get_dataset(file_list, batch_size=1, repeat=False, subsample=True, shuffle_buffer=1, n_prefetch=-1):

    dataset_output_types = (tf.float32, tf.float32, tf.float32) # for (times, frames, params)
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.map(lambda f: tf.py_function(_load_numpy_all, [f, subsample], dataset_output_types),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.shuffle(shuffle_buffer)

    # Messy batching, as RaggedTensors not fully supported by Tensorflow's Dataset
    dataset = dataset.map(lambda a, b, c: (tf.expand_dims(a, 0), tf.expand_dims(b, 0), tf.expand_dims(c, 0)), # add the "batch" dimension
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda a, b, c: (tf.RaggedTensor.from_tensor(a), tf.RaggedTensor.from_tensor(b), tf.RaggedTensor.from_tensor(c)),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda a, b, c: (tf.squeeze(a, axis=1), tf.squeeze(b, axis=1), tf.squeeze(c, axis=1)),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(n_prefetch) # -1 = tf.data.experimental.AUTOTUNE (can print it to see)

    return dataset

def get_encoded_dataset(filepaths, echo_ae, logger, output_dir=None, subsample=False):

    t_start = time.time()
    filenames = [Path(f).stem for f in filepaths]
    filename_data = {}
    if output_dir is not None and output_dir.exists():
        encoded_data_file_p = output_dir / "encoded_echos.npz"
        logger.info(f"Loading pre-encoded echos from {encoded_data_file_p}")
        data = np.load(encoded_data_file_p)
        loaded_latents = data["latents"]
        loaded_efs = np.reshape(data["efs"], (-1, 1))
        loaded_filenames = data["filenames"]
        # map each filename to its index
        loaded_filenames_indices = {}
        for i, filename in enumerate(loaded_filenames):
            loaded_filenames_indices[str(filename)] = i
        
        # save latent and ef of requested filenames from loaded data
        # also save filenames not found in the loaded data
        filenames_to_encode_indices = []
        for i, filename in enumerate(filenames):
            if filename in loaded_filenames_indices:
                idx = loaded_filenames_indices[filename]
                filename_latent = loaded_latents[idx, None]
                filename_ef = loaded_efs[idx, None]
                filename_data[filename] = (filename_latent, filename_ef)
            else:
                filenames_to_encode_indices.append(i)
        
        if len(filenames_to_encode_indices) == 0:
            logger.info(f"Found encoding of all requested echos, not encoding...")
            filepaths = [] # no echo files to encode
        else:
            # get the filepaths of the filenames remaining to encode
            filepaths = [filepaths[i] for i in filenames_to_encode_indices]
    
    if len(filepaths) > 0: # if there are any echo files to encode
        logger.info(f"Encoding {len(filepaths)} echo files...")

        batch_size = 32
        n = len(filepaths)
        n_batches = np.ceil(n/batch_size).astype(np.int32)
        dataset = get_dataset(filepaths, batch_size=batch_size, repeat=False, subsample=subsample, shuffle_buffer=1, n_prefetch=-1)
        true_efs_index = ECHO_DATA_PARAMS.index("EF")
        for b, (times_b, frames_b, echo_params) in enumerate(dataset):
            params = echo_ae.encode(times_b, frames_b, return_freq_phase=True, training=False).numpy()
            echo_params = echo_params.to_tensor(default_value=0.0).numpy()
            true_efs = echo_params[:, true_efs_index, None]
            for k in range(len(params)):
                global_index = b * batch_size + k
                filename = Path(filepaths[global_index]).stem
                filename_latent = params[k, None]
                filename_ef = true_efs[k, None]
                filename_data[filename] = (filename_latent, filename_ef)

            if b+1 == 1 or (b+1) % 50 == 0:
                t2 = time.time()
                h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
                logger.info(f"Batch {b+1}/{n_batches} done in {h}:{m}:{s}")

    latents = [filename_data[f][0] for f in filenames]
    efs = [filename_data[f][1] for f in filenames]
    latents = np.concatenate(latents, axis=0)
    efs = np.concatenate(efs, axis=0)

    t2 = time.time()
    h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t_start)
    logger.info(f"DHB encoded dataset in {h}:{m}:{s}")

    # if saving data
    if output_dir is not None:
        logger.info(f"Saving encoded data under {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        encoded_data_file_p = output_dir / "encoded_echos.npz"
        np.savez_compressed(encoded_data_file_p, latents=latents, efs=efs, filenames=filenames)
        logger.info(f"Done saving.")
    
    logger.info(f"Echo latents shape: {latents.shape}")
    logger.info(f"Echo efs shape: {efs.shape}")

    return latents, efs

def random_subsequence_start_end(N, min_length):
    """
    Random start and end indices for a random subsequence for a sequences of length N.
    """
    n_subsequences = ((N-min_length)**2 + N - min_length)/2
    start = np.random.choice(np.arange(N - min_length), p=np.arange(N - min_length, 0, step=-1)/n_subsequences)
    end = np.random.choice(np.arange(start+min_length, N))
    return start, end

def _load_numpy_all(filepath, subsample):
    data = np.load(filepath.numpy())
    times = data['times']
    frames = data['frames']
    params = data['params']

    minimum_length = np.ceil(2.0/times[1]).astype('int') # nb frames of the video that make up a 2.0 seconds duration
    if subsample and not times.shape[0] <= minimum_length: # if video duration not less than 2.0 seconds (i,e greater than 2.0 seconds)
        # cut video (i,e extract sub-sequence of frames between indices "start" and "end") to get a sub-video of duration
        # at least 2.0 seconds, i,e the video shows same heart, with same frequency but different shift
        # indices "start" and "end" are chosen randomly and "end - start" is at least "minimum_length" 
        start, end = random_subsequence_start_end(times.shape[0], minimum_length)
        times = times[start:end] - times[start] # extract subsequence of times, shift it to get first value = 0
        frames = frames[start:end] # extract corresponding subsequence of frames
    
    frames = (frames/255).astype('float32') # normalize
    return times, frames, params

