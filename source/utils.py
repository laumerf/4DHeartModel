import sys
import os, time
import logging
from pathlib import Path
import importlib
import io
import socket
import matplotlib.pyplot as plt
import tensorflow as tf
import GPUtil as GPU
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
import cv2
import multiprocessing as mp

from source.constants import ROOT_LOGGER_STR
from source.constants import LEOMED_DATA_DIR

LEOMED_RUN = "--leomed" in sys.argv

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

# colors in BGR mode
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
VIOLET_BLUE = (142, 120, 41)
ORANGE = (0, 165, 255)
PINK = (147, 20, 255)

bad_quality_echos = ['0X10623D3AF96AC271', '0X10B04432B90E5AC2', '0X1BD7A625C9DA5292', '0X1D3D82FD91F61757', '0X1E433E7966FD7332', '0X20586B1BD35F38C9', '0X22A9B6ECBD591065', '0X22E82C3D081C819C', '0X286661146EB02EE4', '0X2A20552783F445AE', '0X2ABBA31A554D6E2', '0X2D3F2E5FFA807CFF', '0X2D405B452654D053', '0X31C7FC69D6C348EA', '0X33EAE0F44B7618C1', '0X374099556945A9EA', '0X39207AF594BF77D7', '0X39348579B2E55470', '0X3BAE126085E973A0', '0X3F3DC1A6F0B18FA', '0X4A11C148E80CABDE', '0X4F89846030713617', '0X52619FC2739EB1F1', '0X52832AF3B2EE7826', '0X533498EF6F72192', '0X55DD5AB1762EDCDA', '0X5746FD1045390A93', '0X64EE9FFA5DA69058', '0X659BBAF883211018', '0X67F0850A73E98169', '0X6CAFD90BF9854990', '0X6FA8D7C2278B5073', '0X7790F4DB18852455', '0X786686A9B8DE6547', '0X790C871B162806D2', '0X7CA9A912598CA322', '0X7F3ABE82B6992583', '0XBE06F978BB3226D']

def get_mesh_feats_times_from_id(dataset, mesh_id):
    batch_size = get_mesh_dataset_batch_size(dataset)
    for feats, times, _, _ in dataset.skip(int(mesh_id // batch_size)): # encode the batch where mesh 1 is
        row_limits = tf.cumsum(times.row_lengths()).numpy()
        idx = int(mesh_id % batch_size)
        i = 0 if idx == 0 else row_limits[idx-1]
        j = row_limits[idx]
        mesh_feats = feats.values[i:j].numpy()
        mesh_times = times.values[i:j].numpy()
        return mesh_feats, mesh_times

def get_all_valid_laxes_from_mask():
    logger.info("Pre-computing laxes...")
    t1 = time.time()
    mask_points = cv2.imread("mask_points.png", cv2.IMREAD_GRAYSCALE)
    # grey pixels are for lax_top, white pixels for lax_bottom
    ys, xs = np.where(mask_points == 128)
    grey_pixels = np.array(list(zip(xs, ys)))
    ys, xs = np.where(mask_points == 255)
    white_pixels = np.array(list(zip(xs, ys)))

    logger.info(f"Nb top points: {len(grey_pixels)}")
    logger.info(f"Nb bottom points: {len(white_pixels)}")
    valid_laxes = []
    for lax_top in grey_pixels:
        for lax_bottom in white_pixels:
            length = np.linalg.norm(lax_top - lax_bottom)
            # TODO: maybe tune the length values
            if length >= 35 and length <= 65: # lax is valid only if length is greater than this value
                valid_laxes.append((lax_top, lax_bottom))
    
    t2= time.time()
    h, m, s = get_HH_MM_SS_from_sec(t2 - t1)
    logger.info(f"Done in {h}:{m}:{s}")
    return valid_laxes

def plot_x_y_data(plot_dir, filename, y_data_list, x_data_list, labels, x_label, y_label, title, h_lines_y_values=None):
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.clf()
    plt.title(title)

    for x_data, y_data, label in zip(x_data_list, y_data_list, labels):
        plt.plot(x_data, y_data, label=label)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    if h_lines_y_values is not None:
        for y_val in h_lines_y_values:
            plt.axhline(y=y_val, linestyle='-')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(plot_dir / filename, bbox_inches='tight')

def scatter_plot(plot_dir, filename, x_data, y_data, x_label, y_label, title, diff=False, plot_y_equals_x=False):
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.clf()
    plt.plot(x_data, y_data, 'bo')
    plt.title(title)
    min_x = min(x_data)
    max_x = max(x_data)
    min_y = min(y_data)
    max_y = max(y_data)
    if not diff:
        plt.xlabel(f'{x_label}\nmin val: {min_x:.5f}, max val: {max_x:.5f}')
        plt.ylabel(f'{y_label}\nmin val: {min_y:.5f}, max val: {max_y:.5f}')
    else:
        diff_x = max_x - min_x
        diff_y = max_y - min_y
        plt.xlabel(f'{x_label}\nmin val: {min_x:.5f}, max val: {max_x:.5f}, diff: {diff_x:.5f}')
        plt.ylabel(f'{y_label}\nmin val: {min_y:.5f}, max val: {max_y:.5f}, diff: {diff_y:.5f}')
    if plot_y_equals_x:
        l = [min(x_data), max(x_data)]
        plt.plot(l, l, color="red", linewidth=2)
    
    plt.savefig(plot_dir / filename, bbox_inches='tight')

def hist_plot(plot_dir, filename, data, bins, title):
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.clf()
    plt.hist(data, bins=bins, histtype='bar', ec='black')
    plt.title(title)
    plt.savefig(plot_dir / filename, bbox_inches='tight')

def get_volume_tracings_df(echo_data_dir):
    # Load DHB AE data
    # read volume tracings file
    volume_tracings = pd.read_csv(str(echo_data_dir / 'VolumeTracings.csv')) # file with volume tracing data as a pandas df
    return volume_tracings
    
def tf_dataset_from_tensor_slices(dataset, batch_size=None, shuffle_buffer=None, shuffle_seed=None, samples_taken=None, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer, seed=shuffle_seed)
    if samples_taken is not None:
        dataset = dataset.take(samples_taken)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset

def get_mesh_dataset_batch_size(dataset):
    # get batch_size from first batch
    for _, times, _, _ in dataset:
        # get row lengths
        row_lengths = times.row_lengths().numpy()
        return row_lengths.shape[0]

def compute_iou_and_dice(mesh_lv_frame, echo_lv_frame):
    # the 2 images must have the same shape
    assert mesh_lv_frame.shape == echo_lv_frame.shape

    # get mesh white pixels
    ys, xs = np.where(mesh_lv_frame == 255)
    mesh_pixels = list(zip(xs, ys))
    # get echo white pixels
    ys, xs = np.where(echo_lv_frame == 255)
    echo_pixels = list(zip(xs, ys))
    
    intersection = np.zeros(mesh_lv_frame.shape, dtype=np.uint8)
    for pixel in mesh_pixels:
        if pixel in echo_pixels:
            x, y = pixel
            intersection[y, x] = 255
    
    union = np.zeros(mesh_lv_frame.shape, dtype=np.uint8)
    for pixel in mesh_pixels:
        x, y = pixel
        union[y, x] = 255
    for pixel in echo_pixels:
        x, y = pixel
        union[y, x] = 255
    
    intersection_count = len(np.where(intersection == 255)[0])
    union_count = len(np.where(union == 255)[0])
    mesh_pixels_count = len(mesh_pixels)
    echo_pixels_count = len(echo_pixels)

    IoU = intersection_count / union_count
    Dice = (2 * intersection_count) / (mesh_pixels_count + echo_pixels_count)
    return IoU, Dice

def fill_contour(contour_bw):
    filled = np.zeros((contour_bw.shape[0], contour_bw.shape[1]), dtype=np.uint8)
    for i, row in enumerate(contour_bw):
        white_pixels = np.where(row == 255)[0]
        if len(white_pixels) > 0:
            min_x = white_pixels.min()
            max_x = white_pixels.max()
            filled[i, min_x:max_x+1] = 255
    
    return filled
   
def generate_echo_tracings(echo_filenames, echo_edfs, volume_tracings, output_size):
    # generate the echo tracings for EDF and ESF
    # skip echos with more than 42 segemnts traced (EDF+ESF) i,e where other chambers are
    # traced apart from the LV
    # also skip echos found with bad quality tracing or acquisition
    global bad_quality_echos
    skipped_echos = bad_quality_echos
    tracings_data = {}
    
    logger.info("Generating echo tracings...")
    t1 = time.time()
    for echo_filename, echo_edf in zip(echo_filenames, echo_edfs):
        if echo_filename not in skipped_echos:
            result = get_echo_tracing_and_long_axis(echo_filename, echo_edf, volume_tracings, output_size)
            if result is None: # skip this file
                skipped_echos.append(echo_filename)
                continue
        
        # echo file not skipped
        echo_lv_frames, echo_laxes, echo_frames_nbs = result
        tracings_data[echo_filename] = list(zip(echo_lv_frames, echo_laxes, echo_frames_nbs))
    
    t2 = time.time()
    h, m, s = get_HH_MM_SS_from_sec(t2 - t1)
    logger.info(f"Done generating echo tracings in {h}:{m}:{s}")
    logger.info(f"Skipped {len(skipped_echos)} echo videos")
    return tracings_data, skipped_echos

def get_echo_tracing_and_long_axis(echo_filename, echo_edf, volume_tracings, output_size):
    file_data = volume_tracings[volume_tracings["FileName"] == echo_filename]
    row_count = len(file_data.index)
    if row_count != 42:
        return None

    echo_frames_nbs = []
    echo_lv_frames = []
    echo_laxes = []
    for frame_nb, frame_data in file_data.groupby("Frame"):
        segs = []
        for idx, row in frame_data.iterrows():
            # get segment points
            x1, x2, y1, y2 = row["X1"], row["X2"], row["Y1"], row["Y2"]
            p1 = (x1, y1)
            p2 = (x2, y2)
            # rescale points
            p1 = resized_coordinates(p1, (112, 112), output_size)
            p2 = resized_coordinates(p2, (112, 112), output_size)
            if len(segs) == 0 and p1[1] > p2[1]: # make sure, for the long axis, that p1 is the top point
                tmp = p2
                p2 = p1
                p1 = tmp
            if len(segs) > 0 and p1[0] > p2[0]: # make sure, for other segments, that p1 is the point on the left side (i,e x-axis value lower)
                tmp = p2
                p2 = p1
                p1 = tmp
            segs.append((p1, p2))
        
        lax = segs[0]
        segs = segs[1:]
        
        left_points = [seg[0] for seg in segs]
        right_points = [seg[1] for seg in segs[::-1]]
        contour_points = np.array([lax[0]] + left_points + right_points)
        # print(contour_points)
        
        # make tracing
        echo_frame = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
        echo_frame = cv2.fillPoly(echo_frame, pts =[contour_points], color=WHITE)
        
        # turn to black and white
        echo_frame = cv2.cvtColor(echo_frame, cv2.COLOR_BGR2GRAY)
        echo_frame = cv2.threshold(echo_frame, 0, 255, cv2.THRESH_BINARY)[1]
        # save data
        echo_laxes.append(lax)
        echo_frames_nbs.append(frame_nb)
        echo_lv_frames.append(echo_frame)
    
    # EDF first
    if echo_frames_nbs[0] != echo_edf:
        # reverse echo_lv_frames, echo_laxes and echo_frame_nbs lists to get EDF first then ESF
        echo_lv_frames = echo_lv_frames[::-1]
        echo_laxes = echo_laxes[::-1]
        echo_frames_nbs = echo_frames_nbs[::-1]
    
    # return 3 lists of length 2 containing respectively the "lv tracings", "long axis" and "frame nb"
    # for each of EDF and ESF (edf first)
    return echo_lv_frames, echo_laxes, echo_frames_nbs

def get_echo_long_axis_points(echo_filename, volume_tracings, new_size=None):
    if new_size is None:
        new_size = (112, 112)
    
    frames_coordinates = volume_tracings[volume_tracings["FileName"] == echo_filename]
    lax = {}
    # manually add the odd cases
    # mapping frame number to idx of lax, this mapping is mapped to the echo_filename
    lax_idx = {}
    lax_idx["0X3A6E1BAA9065D40"] = {35: 148974, 45: 149016} 
    lax_idx["0X44C18287CA978438"] = {51: 185828, 63: 185912}
    lax_idx["0X57AF4D24B154C573"] = {1: 253492, 15: 253488}
    lax_idx["0X65E605F203321860"] = {32: 305319, 53: 305236}
    lax_idx["0X354B37A25C64276F"] = {31: 130723, 38: 130618}
    lax_idx["0X973E4A9DAADDF9F"] = {40: 402149, 49: 402045}
    lax_idx["0X12430512E2BBCD55"] = {38: 8357, 50: 8503}

    
    for frame, coordinates in frames_coordinates.groupby(["Frame"]):
        max_length = -1
        p1_lax = None
        p2_lax = None
        # loop over the segments, get the longest one as the long axis
        for idx, row in coordinates.iterrows():
            # if lax manually labeled
            if echo_filename in lax_idx.keys():
                # if this line segment is not the one of the lax at this frame, skip it
                if idx != lax_idx[echo_filename][frame]:
                    continue
            
            p1 = np.array([row["X1"], row["Y1"]])
            p2 = np.array([row["X2"], row["Y2"]])
            length = np.linalg.norm(p1 - p2)
            if length > max_length:
                p1_lax = p1
                p2_lax = p2
                max_length = length
        
        # resize and select top and bottom points of lax
        p1_lax = resized_coordinates(p1_lax, (112, 112), new_size)
        p2_lax = resized_coordinates(p2_lax, (112, 112), new_size)
        if p1_lax[1] < p2_lax[1]:
            p_top = p1_lax
            p_bottom = p2_lax
        else:
            p_top = p2_lax
            p_bottom = p1_lax
        # save lax of this echo frame
        lax[frame] = (p_top, p_bottom)

    return lax

def get_alive_jobs(parallel_jobs):
    # close jobs not alive and get alive jobs
    alive_jobs = []
    for process in parallel_jobs:
        if process.is_alive():
            alive_jobs.append(process)
        else:
            process.close() # close job and release ressources
    return alive_jobs
    
def wait_parallel_job_completion(parallel_jobs, max_parallel_jobs, verbose=False):
    if len(parallel_jobs) < max_parallel_jobs: # not waiting
        return parallel_jobs
    
    first = True
    while len(parallel_jobs) >= max_parallel_jobs: # if reached max number of parallel jobs
                                                    # wait for some jobs to finish before continuing
        if first and verbose:
            logger.info("Waiting for some parallel jobs to finish...")
            first = False
        
        parallel_jobs = get_alive_jobs(parallel_jobs)
    
    if verbose:
        logger.info("Done waiting parallel jobs completion.")
    return parallel_jobs

def swap(a, b):
    return b, a

def skip_dict_keys(d, ks_to_skip):
    d_new = {}
    for k in d: # fro every key in dict
        if k not in ks_to_skip: #  if key not in list of keys to skip
            d_new[k] = d[k] # add mapping
    return d_new

def transform_point(point, transform_matrices):
    for transform_mat in transform_matrices:
        point = np.append(point, 1)
        point = transform_mat.dot(point)
    point = np.rint(point).astype('int32')
    return tuple(point)
    
def get_line_params(p1 , p2):
    x1, y1 = p1
    x2, y2 = p2
    m = (y1 - y2) / (x1 - x2) # slope
    b = y1 - m * x1 # bias
    return m, b

def overlay_non_black_pixels(image, overlay):
    if image.shape[0] != overlay.shape[0] or image.shape[1] != overlay.shape[1]:
        logger.info("Error in overlay_non_black_pixels method in utils.py")
        return None
    b_overlay, g_overlay, r_overlay = overlay[:, :, 0], overlay[:, :, 1], overlay[:, :, 2]

    locations = np.logical_and(np.logical_and(b_overlay == 0, g_overlay == 0), r_overlay == 0) # locations where pixel is black in the overlay
    mask = np.repeat(np.expand_dims(locations, axis=-1), 3, axis=-1)
    return np.where(mask, image, overlay)

def black_white_to_color(image, color):
    # image is expected to be black and white
    # the white pixels are turned to the specified color
    b, g, r = color
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    b_channel, g_channel, r_channel = (image[:, :, 0] / 255) * b, (image[:, :, 1] / 255) * g, (image[:, :, 2] / 255) * r
    return cv2.merge([b_channel, g_channel, r_channel], 3).astype(np.uint8)

def resized_coordinates(point, old_size, new_size):
    x, y = point
    old_w, old_h = old_size
    new_w, new_h = new_size
    ratio_x = new_w / old_w
    ratio_y = new_h / old_h
    new_point = np.rint((ratio_x * x, ratio_y * y)).astype(np.int32)
    return new_point

def distance_vectors(v1, v2):
    x1, y1 = v1[0], v1[1]
    x2, y2 = v2[0], v2[1]
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def rotate_scale_point(point, rot_scale_mat):
    point = np.append(point, 1)
    point = rot_scale_mat.dot(point)
    point = np.rint(point).astype('int32')
    return point
    
def signed_angle_v2_to_v1(v1, v2):
    # atan2(y1, x1) is the angle from the x-axis to the vector v1 = (x1, y1)^T
    # given 2 vectors v1 and v2, let the angle "theta" from x-axis to v1 and "beta" the angle from x-axis to v2.
    # beta and theta in [0, 360)
    # then "beta - theta" is the signed angle from v2 to v1
    x1, y1 = v1[0], v1[1]
    x2, y2 = v2[0], v2[1]
    theta = (np.arctan2(y1, x1) * 180)  / np.pi
    beta = (np.arctan2(y2, x2) * 180)  / np.pi
    if theta < 0:
        theta += 360
    if beta < 0:
        beta += 360
    return beta - theta

def consecutive_groups(arr):
    ranges =[]

    for k,g in groupby(enumerate(arr),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append((group[0],group[-1]))
    
    return ranges

def spaced_values_indices(arr, nb_indices):
    arr = list(arr) # convert to list
    sorted_arr = sorted(arr) # sort
    # pick arr values evenly spaced
    sorted_arr_indices = np.rint(np.linspace(0, len(sorted_arr)-1, num=nb_indices)).astype("int64")
    picked_vals = [sorted_arr[i] for i in sorted_arr_indices]
    # find indices of the picked values in the original array
    indices = [arr.index(val) for val in picked_vals]

    return indices

def spaced_efs_indices(efs, nb_indices):

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    min_ef, max_ef = min(efs), max(efs)
    spaced_efs = np.linspace(min_ef, max_ef, num=nb_indices) # pick efs as spaced out as possible
    # find nearest efs in array and pick them
    picked_efs_indices = [find_nearest(efs, ef) for ef in spaced_efs]

    # return unique indices in ascending ef value
    picked_efs_indices_unique = []
    for idx in picked_efs_indices:
        if idx not in picked_efs_indices_unique:
            picked_efs_indices_unique.append(idx)
    return np.array(picked_efs_indices_unique)

def get_HH_MM_SS_from_sec(sec):
    # takes an amount of time in seconds and converts it to
    # hours (HH), minutes (MM) and seconds (SS)
    SS = sec % 60
    rest_MM = (sec - SS) / 60
    MM = int(rest_MM % 60)
    HH = int((rest_MM - MM) / 60)
    return HH, MM, SS


def set_gpu(local_gpu_name='fabian'):
    # get first available gpu on local machine
    if local_gpu_name in str(socket.gethostname()):
        gpu_str = str(GPU.getFirstAvailable(order="load")[0])
        logger.info(f"local gpu: {gpu_str}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        return

    try:
        gpu_str = str(GPU.getFirstAvailable(order="load",
                                            maxLoad=10 ** -6,
                                            maxMemory=10 ** -1)[0])
        logger.info(f"server gpu: {gpu_str}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    except Exception as err:
        logger.warning(f"No GPU founded on this device: {err}")


def setup_logger(results_path, create_stdlog):
    """Setup a general logger which saves all logs in the experiment folder"""

    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler = logging.FileHandler(str(results_path))
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(f_format)

    root_logger = logging.getLogger(ROOT_LOGGER_STR)
    root_logger.handlers = []
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(f_handler)

    if create_stdlog:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)


def get_data_handler_cls(model_name):
    return _get_class_from_package(model_name, package='source')


def get_conv_layer(layer_name):
    return _get_class_from_package(layer_name, package="spektral.layers")


def get_regularizer(reg_name):
    return _get_class_from_package(reg_name,
                                   package="tensorflow.keras.regularizers")


def _get_class_from_package(obj_name, package):
    modellib = importlib.import_module(package)
    obj = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == obj_name.lower():
            obj = cls
    return obj


def plot_to_image(figure):

    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_input_output(input_img, output_img):

    if len(input_img.shape) == 4:
        input_img = input_img[0, ..., 0]
        output_img = output_img[0, ..., 0]

    if input_img.shape[-1] == 1:
        input_img = input_img[..., 0]
        output_img = output_img[..., 0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(input_img, cmap='gray')
    axes[1].imshow(output_img, cmap='gray')

    return fig
