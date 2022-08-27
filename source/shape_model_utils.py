import logging
import lzma
import pickle
import random
import string
import time
from itertools import combinations
from pathlib import Path

import cv2
import imageio
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import pandas as pd
import networkx as nx
import vtk
from scipy.signal import find_peaks

import source.utils as utils
import source.constants as constants
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from source.constants import CONRAD_COMPONENTS, CONRAD_IDS_TO_COMPONENTS, \
    CONRAD_VIEWS
from source.constants import ROOT_LOGGER_STR
import sys

# get the logger
logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

""" 
Notes:
-In a poly data, the "cells" (e,g poly.GetCellData()) are the polygons i,e the triangles, so each cell is a triangle
-Inspected the polygones of all 10 phases of the left ventricle (mean shapes saved in: heart_mesh/shape_models/CardiacModel/vtkPolys/leftVentricle).
All polygones are the same, i,e every 3 vertices making up a triangle in phase i are also making up a triangle in phase j.
So the adjacency matrices are the same in all phases.
Generalizing to all other components, we have 1 adjacency matrix per heart component for all phases, thus 1 adjacency matrix
for the full heart for all phases 
"""

# -------------------------------------------- VID Generation --------------------------------------------
def save_mesh_vid(mesh_vid, mesh_data_handler, vid_duration=-1, vid_out_dir=None, vid_name=None, rescale=True, save_vtps=False, vtps_out_dir=None):
    # denormalize the mesh video features
    mesh_dataset_scale = mesh_data_handler.dataset_scale
    mesh_dataset_min = mesh_data_handler.dataset_min
    mesh_ref_poly = mesh_data_handler.reference_poly

    if rescale:
        mesh_vid = (mesh_vid * mesh_dataset_scale) + mesh_dataset_min

    frames = []
    for j, mesh_frame in enumerate(mesh_vid):
        vtp_name = f"{vid_name}_{str(j).zfill(3)}"
        mesh_frame_poly = overwrite_vtkpoly(mesh_ref_poly, points=mesh_frame, save=save_vtps,
                                            output_dir=vtps_out_dir, name=vtp_name)
        # render poly and save the rendered image
        # camera params and rendering size are custom values
        camera_params = {
            "position": (-58.52, 268.59, -258.29),
            "focal_point": (-1.415, 1.354, -0.106),
            "up_vect": (0.8, -0.32, -0.5)
        }
        size = (500, 500)
        image = render_poly_from_camera_params(mesh_frame_poly, camera_params, size,
                                               colors=constants.CONRAD_HEART_COLORS,
                                               return_bgr=True)
        frames.append(image)
    
    if vid_out_dir is not None and vid_name is not None and vid_duration > 0:
        frames_to_vid(vid_out_dir, frames, vid_duration, vid_name)
    return frames

def save_mesh_vid_slice(mesh_vid, mesh_data_handler, mesh_lax_new=None, vid_duration=-1, vid_out_dir=None, vid_name=None):
    output_size = (112, 112)

    # denormalize the mesh video features
    mesh_dataset_scale = mesh_data_handler.dataset_scale
    mesh_dataset_min = mesh_data_handler.dataset_min
    mesh_ref_poly = mesh_data_handler.reference_poly

    mesh_vid = (mesh_vid * mesh_dataset_scale) + mesh_dataset_min

    lv_volumes = compute_volumes_feats(mesh_vid, mesh_ref_poly, components_list=["leftVentricle"])["leftVentricle"]
    ed_mesh_idx = np.argmax(lv_volumes)
    ed_mesh_feats = mesh_vid[ed_mesh_idx]

    ed_mesh = overwrite_vtkpoly(mesh_ref_poly, points=ed_mesh_feats)
    origin, normal, points, _, _ = get_4ch_slicing_plane(ed_mesh)

    lax_p1, lax_p2, _ = points
    lax_vect = lax_p2 - lax_p1 # vector around which to rotate the normal
    lax_vect = lax_vect / np.linalg.norm(lax_vect) # normalize

    slicing_plane = {"origin": origin, "normal": normal}
    camera_params = (origin, normal, -lax_vect)
    _, mesh_lax = render_lv_get_lax(ed_mesh, slicing_plane, camera_params, output_size)

def vtp_to_png_clip(vtp_file_path, output_dir, do_clip=False):
    RemoveViewsAndLayouts()

    mesh_000vtp = XMLPolyDataReader(FileName=[vtp_file_path])

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    renderView1.ViewSize = [2*450, 2*600]

    # get color transfer function/color map for 'components'
    componentsLUT = GetColorTransferFunction('components')

    # get opacity transfer function/opacity map for 'components'
    componentsPWF = GetOpacityTransferFunction('components')

    # create a new 'Clip'
    clip1 = Clip(Input=mesh_000vtp)

    # Properties modified on clip1.ClipType
    #clip1.ClipType.Origin = [0.7355424310656785, 0.6042084639503627, 0.6704436456614302]
    #clip1.ClipType.Normal = [0.859548331518909, 0.2845705895305824, 0.4244952830801266]

    clip1.ClipType.Origin = [0.46416302342161947, 0.5234455100636909, 0.48970351287734637]
    clip1.ClipType.Normal = [0.2604502561771543, -0.09192058065292552, 0.9611015923978504]

    # show data in view
    clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    clip1Display.Representation = 'Surface'

    # Properties modified on renderView1
    renderView1.CameraParallelProjection = 1

    # Hide orientation axes
    renderView1.OrientationAxesVisibility = 0

    # hide data in view
    if do_clip:
        Hide(mesh_000vtp, renderView1)
    else:
        Hide(clip1, renderView1)
        Show(mesh_000vtp, renderView1)

    # hide color bar/color legend
    clip1Display.SetScalarBarVisibility(renderView1, False)

    # set active source
    SetActiveSource(mesh_000vtp)

    # toggle 3D widget visibility (only when running from the GUI)
    Hide3DWidgets(proxy=clip1.ClipType)

    # Properties modified on componentsLUT
    componentsLUT.InterpretValuesAsCategories = 1

    ImportPresets(filename='heart_mesh/shape_models/CONRAD/color_pallete_w_aorta_softer.json')

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    componentsLUT.ApplyPreset('Preset 1', True)

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    componentsLUT.ApplyPreset('Preset 1', False)

    # Properties modified on componentsLUT
    componentsLUT.IndexedOpacities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # current camera placement for renderView1
    #renderView1.CameraPosition = [2.1919264259410123, 1.4253141394435989, 2.075289674903951]
    renderView1.CameraPosition = [1.2080827131288339, 0.2773846043817438, 3.5054295546550587]
    renderView1.CameraFocalPoint = [0.50629213452339, 0.5372298881411527, 0.5138907697831137]
    #renderView1.CameraViewUp = [-0.7250119074561387, 0.45321139859442805, 0.5186108003415412]
    renderView1.CameraViewUp = [-0.8510364546312318, 0.46694586159551016, 0.2402051523750693]
    renderView1.CameraParallelScale = 0.5294938099984521
    renderView1.CameraParallelProjection = 1

    # save screenshot
    filename = vtp_file_path.split('/')[-1].split('.')[0]
    out_file = str(output_dir / filename) + ".png"
    SaveScreenshot(out_file, renderView1, TransparentBackground=1, CompressionLevel=0)

    return Path(out_file)


def vtp_to_png(vtps_folder, pngs_folder, filename, state_file="paraview_state.pvsm"):
    vtp_file_path = str(vtps_folder / filename) + ".vtp"

    pngs_folder.mkdir(parents=True, exist_ok=True)
    png_file_path = str(pngs_folder / filename) + ".png"

    LoadState(state_file)

    # position camera
    # view = GetActiveView()
    # set image size
    # view.ViewSize = [567, 751] #[width, height]

    # read a vtp file
    reader = OpenDataFile(vtp_file_path)
    if not reader:
        logger.info("ERROR: Could not open vtk/vtp file {}".format(vtp_file_path))
        exit(1)
    Show()
    # save screenshot
    WriteImage(png_file_path)
    # Delete()


# -------------------------------------------- GIF Generation --------------------------------------------
def pngs_to_gif(pngs_folder, gif_folder):
    gif_filename = pngs_folder.name + ".gif"

    gif_folder.mkdir(parents=True, exist_ok=True)
    gif_file_path = str(gif_folder / gif_filename)

    # list png files
    images = []
    for filename in pngs_folder.iterdir():
        if filename.is_file():
            images.append(str(filename))

    images = sorted(images)

    images = list(map(lambda image: imageio.imread(image), images))
    per_frame = 2.0 / len(images)  # duration per frame, total duration = 2.0 sec
    imageio.mimsave(gif_file_path, images, format='GIF', duration=per_frame)


def vtps_to_gif(vtps_folder, filenames):
    # keep same folder structure for png files, except use reconstruction_png for the folder containing all pngs
    # likewise for the generated gif
    pngs_folder = Path(str(vtps_folder).replace("/reconstruction", "/reconstruction_png"))
    gif_folder = Path(str(vtps_folder).replace("/reconstruction", "/reconstruction_gif")).parent

    # convert vtps to pngs and save
    for filename in filenames:
        vtp_to_png(vtps_folder, pngs_folder, filename)

    # convert pngs under same folder to gif
    pngs_to_gif(pngs_folder, gif_folder)


# --------------------------------------------------------------------------------------------------------

# -------------------------------------------- VID Generation --------------------------------------------
def pngs_to_vid(pngs_folder, vid_folder, vid_duration):
    vid_filename = pngs_folder.name + ".avi"

    vid_folder.mkdir(parents=True, exist_ok=True)
    vid_file_path = str(vid_folder / vid_filename)

    # list png files
    images = []
    for filename in pngs_folder.iterdir():
        if filename.is_file():
            images.append(str(filename))

    images = sorted(images)

    # read a frame to get video size
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    fps = len(images) / vid_duration

    fourcc = cv2.VideoWriter_fourcc(
        *'FFV1')  # using lossless video codec FFV1 for better video quality
    video = cv2.VideoWriter(vid_file_path, fourcc, fps, (width, height))

    # Appending the images to the video one by one
    for image in images:
        video.write(cv2.imread(image))

        # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


def vtps_to_vid(vtps_folder, filenames, vid_duration):
    # keep same folder structure for png files, except use reconstruction_png for the folder containing all pngs
    # likewise for the generated video
    pngs_folder = Path(str(vtps_folder).replace("/reconstruction", "/reconstruction_png"))
    vid_folder = Path(str(vtps_folder).replace("/reconstruction", "/reconstruction_vid")).parent

    # convert vtps to pngs and save
    for filename in filenames:
        vtp_to_png(vtps_folder, pngs_folder, filename)

    # convert pngs under same folder to video
    pngs_to_vid(pngs_folder, vid_folder, vid_duration)


def feats_to_seg_map_vid(ref_poly, feats, dataset_scale, dataset_min, vid_duration, output_dir,
                         prefix, heart_components=None, views=None):
    if heart_components is None:  # use all heart components
        heart_components = list(CONRAD_COMPONENTS)

    feats = (feats * dataset_scale) + dataset_min
    images = {}
    for feat in feats:
        poly = overwrite_vtkpoly(ref_poly, points=feat)
        seg_maps = generate_seg_maps(poly, heart_components, views=views, save=False)
        for view in seg_maps:
            seg_map_image = seg_maps[view]["map_rgb"]
            if view in images:
                images[view].append(seg_map_image)
            else:
                images[view] = [seg_map_image]

    output_dir.mkdir(parents=True, exist_ok=True)  # make output dir if not exists

    for view in images:
        view_images = images[view]
        seg_map_file_path = str(output_dir / (prefix + "_" + view + ".avi"))

    mesh_4ch_frames = []
    for mesh_frame in mesh_vid:
        mesh = overwrite_vtkpoly(mesh_ref_poly, points=mesh_frame)
        mesh_slice = slice_poly(mesh, origin, normal)
        mesh_4ch_frame = render_poly_lax_vertical(mesh_slice, origin, normal, -lax_vect, size=output_size)
        # turn to black and white and fill holes
        mesh_4ch_frame = cv2.cvtColor(mesh_4ch_frame, cv2.COLOR_BGR2GRAY)
        mesh_4ch_frame = cv2.threshold(mesh_4ch_frame, 0, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mesh_4ch_frame = cv2.morphologyEx(mesh_4ch_frame, cv2.MORPH_GRADIENT, kernel)

        assert len(np.unique(mesh_4ch_frame)) == 2 # make sure it's black and white

        mesh_4ch_frames.append(mesh_4ch_frame)

    # read mask
    mask_echo_device = cv2.imread("mask_echo_device.png", cv2.IMREAD_GRAYSCALE)
    mask_echo_device = mask_echo_device / 255

    if mesh_lax_new is not None:
        rot_scale_mat, translation_mat = get_transform_matrices(mesh_lax, mesh_lax_new, output_size, output_size)

        mesh_4ch_frames_new = []
        for mesh_4ch_frame in mesh_4ch_frames:
            # transform mesh_4ch_frame by translation, rotation and scaling
            mesh_4ch_frame = cv2.warpAffine(src=mesh_4ch_frame, M=rot_scale_mat, dsize=output_size)
            mesh_4ch_frame = cv2.warpAffine(src=mesh_4ch_frame, M=translation_mat, dsize=output_size)
            mesh_4ch_frame[np.where(mask_echo_device == 0)] = 0
            mesh_4ch_frames_new.append(mesh_4ch_frame)

        mesh_4ch_frames = mesh_4ch_frames_new
        mesh_lax = mesh_lax_new

    if vid_out_dir is not None and vid_name is not None and vid_duration > 0:
        frames_to_vid(vid_out_dir, mesh_4ch_frames, vid_duration, vid_name, isColor=False)

    return mesh_4ch_frames, mesh_lax

def frames_to_vid(output_dir, frames, vid_duration, video_name, denormalize=False, isColor=True): # save video from it's frames
    
    output_dir.mkdir(parents=True, exist_ok=True)

    nb_frames = len(frames)
    height = int(frames[0].shape[0])
    width = int(frames[0].shape[1])
    fps = (nb_frames / vid_duration)

    # write out reconstructions videos
    fourcc = cv2.VideoWriter_fourcc(
        *'FFV1')  # using lossless video codec FFV1 for better video quality
    vid_file_path = str(output_dir / (video_name + ".avi"))
    video = cv2.VideoWriter(vid_file_path, fourcc, fps, (width, height), isColor=isColor)
    # Appending the images to the video one by one
    for frame in frames:
        if denormalize:
            frame = np.clip(frame * 255.0, 0, 255.0).astype(np.uint8)
        video.write(frame)

    # Deallocating memories taken for window creation
    video.release()  # releasing the video generated
    cv2.destroyAllWindows()


# -------------------------------------- Echo/Mesh Helper Functions  -------------------------------------
def echo_edf_mesh_edf(echo_edf, mesh_feats, ref_poly, mesh_dataset_scale, mesh_dataset_min,
                      parallel_process=False, echo_filename="", return_dict=None):
    mesh_feats = (mesh_feats * mesh_dataset_scale) + mesh_dataset_min
    # align echo video and mesh video by aligning ED Frames
    lv_volumes = compute_volumes_feats(mesh_feats, ref_poly, components_list=["leftVentricle"])[
        "leftVentricle"]
    peaks = find_peaks(lv_volumes)[0]  # find peaks in lv volumes
    mesh_edf = 0  # first frame of the mesh video is the peak by default
    if len(peaks) > 0 and lv_volumes[mesh_edf] < lv_volumes[peaks[
        0]]:  # if the volume of the 1st frame lower than that of the 1st peak, take the first peak
        mesh_edf = peaks[0]

    # return results
    if parallel_process:
        return_dict[echo_filename] = (echo_edf, mesh_edf)
    else:
        return (echo_edf, mesh_edf)

def get_transform_matrices(src_lax, dest_lax, src_frame_size, dest_frame_size):
    src_lax_top, src_lax_bottom = src_lax
    dest_lax_top, dest_lax_bottom = dest_lax
    # transform dest lax points to match the src frame size since we apply the
    # transformations on the src frame
    dest_lax_top = utils.resized_coordinates(dest_lax_top, dest_frame_size, src_frame_size)
    dest_lax_bottom = utils.resized_coordinates(dest_lax_bottom, dest_frame_size, src_frame_size)
    # get translation vector
    src_lax_mid = (src_lax_top + src_lax_bottom)/2
    dest_lax_mid = (dest_lax_top + dest_lax_bottom)/2
    translation_vect = dest_lax_mid - src_lax_mid

    v2 = src_lax_top - src_lax_mid
    v1 = dest_lax_top - dest_lax_mid
    angle = utils.signed_angle_v2_to_v1(v1, v2) # get andgle from v1 to v2

    # get scaling factor
    scale = np.linalg.norm(dest_lax_top - dest_lax_bottom) / np.linalg.norm(src_lax_top - src_lax_bottom)
    
    # since we first apply rotation and scaling then translation, so the rotation center is mesh_lax_mid
    rot_scale_mat = cv2.getRotationMatrix2D(center=tuple(src_lax_mid), angle=angle, scale=scale)
    tx, ty = translation_vect
    translation_mat = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    return rot_scale_mat, translation_mat

def save_mesh_vid_slice_with_data_as_np(mesh_vid, mesh_data_handler, times, params, mesh_lax_new, np_out_dir, vid_name, save_avi=False):
    frames, ed_lax = save_mesh_vid_slice(mesh_vid, mesh_data_handler, mesh_lax_new=mesh_lax_new)
    if save_avi:
        avi_out_dir = np_out_dir / "videos"
        vid_duration = times[-1]
        frames_to_vid(avi_out_dir, frames, vid_duration, vid_name, isColor=False)

    ed_lax = np.array([ed_lax[0][0], ed_lax[0][1], ed_lax[1][0], ed_lax[1][1]])

    np_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f'{np_out_dir / vid_name}.npz', frames=frames, times=times, params=params, ed_lax=ed_lax)

# ------------------------------------ Echo/Mesh Overlay VID generation  ---------------------------------
def unicolor_image(image, rgb_color=(41, 120, 142), dilate=False, transparent_background=False):
    # expects image with black background
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)
    # make borders thicker
    if dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 3x3 kernel
        thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = thresh // 255
    r, g, b = rgb_color
    if transparent_background:
        image = [thresh * b, thresh * g, thresh * r, thresh * 255]
        image = cv2.merge(image, 4)
    else:
        image = [thresh * b, thresh * g, thresh * r]
        image = cv2.merge(image, 3)
    return image


def echo_mesh_4ch_overlay(echo_filename, echo_frames, edf, echo_lax, mesh_feats, mesh_data_handler, vid_duration, output_dir, overlay_vid_prefix, output_size=None):
    if output_size is None:
        output_size = (112, 122)

    # mesh dataset params
    mesh_dataset_scale = mesh_data_handler.dataset_scale
    mesh_dataset_min = mesh_data_handler.dataset_min
    ref_poly = mesh_data_handler.reference_poly

    mesh_feats = (mesh_feats * mesh_dataset_scale) + mesh_dataset_min

    ed_mesh = overwrite_vtkpoly(ref_poly, points=mesh_feats[edf])
    origin, normal, points, _, _ = get_4ch_slicing_plane(ed_mesh)

    lax_p1, lax_p2, _ = points
    lax_vect = lax_p2 - lax_p1  # vector around which to rotate the normal
    lax_vect = lax_vect / np.linalg.norm(lax_vect)  # normalize

    slicing_plane = {"origin": origin, "normal": normal}
    camera_params = (origin, normal, -lax_vect)
    _, mesh_lax = render_lv_get_lax(ed_mesh, slicing_plane, camera_params, output_size)

    mesh_4ch_frames = []
    for mesh_feat in mesh_feats:
        mesh = overwrite_vtkpoly(ref_poly, points=mesh_feat)
        mesh_slice = slice_poly(mesh, origin, normal)
        mesh_frame = render_poly_lax_vertical(mesh_slice, origin, normal, -lax_vect,
                                              size=output_size)
        # dilate countour and color
        mesh_frame = unicolor_image(mesh_frame, dilate=True)
        mesh_4ch_frames.append(mesh_frame)

    # get translation, rotation and scale matrices
    mesh_frame_size = echo_frame_size = output_size
    rot_scale_mat, translation_mat = get_transform_matrices(mesh_lax, echo_lax, mesh_frame_size,
                                                            echo_frame_size)

    echo_mesh_4ch_overlay_images = []
    mesh_4ch_frames_aligned = []
    # resize echo frames and overlay 4ch view
    for i, (echo_frame, mesh_4ch_frame) in enumerate(zip(echo_frames, mesh_4ch_frames)):
        # resize echo frame and put in RGB mode
        echo_frame = np.clip(echo_frame * 255.0, 0.0, 255.0).astype(np.uint8)  # denormalize
        echo_frame = cv2.resize(echo_frame, output_size, interpolation=cv2.INTER_CUBIC)
        echo_frame = cv2.cvtColor(echo_frame, cv2.COLOR_GRAY2BGR)

        # transform mesh_4ch_frame by translation, rotation and scaling
        mesh_4ch_frame = cv2.warpAffine(src=mesh_4ch_frame, M=rot_scale_mat, dsize=output_size)
        mesh_4ch_frame = cv2.warpAffine(src=mesh_4ch_frame, M=translation_mat, dsize=output_size)
        mesh_4ch_frames_aligned.append(mesh_4ch_frame)

        # overlay mesh 4ch frame on echo
        # replace all black pixels in mesh_4ch_frame by the corresponding pixel in echo_frame
        echo_mesh_4ch_overlay = utils.overlay_non_black_pixels(echo_frame, mesh_4ch_frame)
        echo_mesh_4ch_overlay_images.append(echo_mesh_4ch_overlay)
    
    frames_to_vid(output_dir / "overlay", echo_mesh_4ch_overlay_images, vid_duration, f"{overlay_vid_prefix}_4ch")
    # frames_to_vid(output_dir / "mesh_slice_aligned", mesh_4ch_frames_aligned, vid_duration, f"{echo_filename}")

# ----------------------------------------------- 3D Volumes ---------------------------------------------
def compute_volumes_poly(polydata, components_list=None, return_mesh_volume=False):
    """
        Compute volume of each heart component found in the polydata and return mapping from heart component to its volume
        Return volume of polydata too
    """
    # t1 = time.time()
    points_components = vtk_to_numpy(polydata.GetPointData().GetAbstractArray(
        'components'))  # i'th elem is the component index of the i'th point

    # by default, find all components in shape and compute volume on them
    components_ids = list(np.unique(points_components))
    component_names = [CONRAD_IDS_TO_COMPONENTS[id] for id in components_ids]

    # if specified a list of components to compute volume on, use it
    if components_list is not None:
        used_ids = []
        used_names = []
        for c, name in zip(components_ids, component_names):
            if name in components_list:
                used_ids.append(c)
                used_names.append(name)
        components_ids = used_ids
        component_names = used_names
    # print(f"Found components {component_names}")

    # """
    #     Test on the mean meshes that the sum of volumes of each component of heart is volume of full heart
    #     Result: There is a small difference
    # """
    # project_dir = Path(".").absolute() # absolute path to repo

    # LEOMED_RUN = "--leomed" in sys.argv

    # data_dir = project_dir / 'heart_mesh' / 'shape_models'

    # vtk_path = data_dir / CONRAD_FOLDER / CONRAD_VTK_FOLDER

    # components_paths = [vtk_path / c / CONRAD_MEAN_FILE.format(0, c, 'vtk') for c in CONRAD_COMPONENTS] # mean shape at phase 0 for each component c
    # components_polys = [load_polydata(path) for path in components_paths]
    # merged_polys = merge_polys(components_polys)

    # volumes = []
    # for poly in components_polys:
    #     Mass = vtk.vtkMassProperties()
    #     Mass.SetInputData(poly)
    #     volume = Mass.GetVolume()
    #     volumes.append(volume)

    # Mass = vtk.vtkMassProperties()
    # Mass.SetInputData(merged_polys)
    # full_heart_volume = Mass.GetVolume()
    # total_volume = sum(volumes)

    # print(f"Mesh volume: {full_heart_volume}")
    # print(f"Sum volumes: {total_volume}")
    # print(f"Diff: {abs(full_heart_volume - total_volume)}")
    # return

    # get the points of the polydata
    points = polydata.GetPoints()

    component_volumes = {}  # one volume per component
    Mass = vtk.vtkMassProperties()  # vtk filter to compte volumes and other quantities
    Mass.SetInputData(polydata)

    # convert point to numpy array
    points_arr = vtk_to_numpy(points.GetData())

    for c, name in zip(components_ids, component_names):  # for each component
        points_arr_copy = points_arr.copy()  # copy original points
        points_arr_copy[points_components != c] = (
        0.0, 0.0, 0.0)  # set all points not part of component "c" to (0.0, 0.0, 0.0)

        # set point of polydata
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points_arr_copy))
        polydata.SetPoints(vtk_points)

        # write_vtk_xml(polydata, save=True, output_dir=Path(".").absolute(), name=name)

        # get the volume of the polydata
        volume = Mass.GetVolume()
        component_volumes[name] = volume

    if return_mesh_volume:
        polydata.SetPoints(points)  # insert back original points
        mesh_volume = Mass.GetVolume()
    else:
        mesh_volume = 0

    # t2 = time.time()
    # h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
    # print("Computed volumes in {}:{}:{}\n".format(h, m, s))

    return component_volumes, mesh_volume


def compute_volumes_vtp(vtp_file_path, components_list=None, return_mesh_volume=False):
    # read vtp file and get the indices of the components it contains
    polydata = load_polydata(vtp_file_path)
    return compute_volumes_poly(polydata, components_list, return_mesh_volume)


def compute_volumes_feat(feat, poly, components_list=None, return_mesh_volume=False):
    # set points of the poly to be the new points "feat"
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(feat))
    poly.SetPoints(vtk_points)

    return compute_volumes_poly(poly, components_list, return_mesh_volume)  # compute volumes

def compute_volumes_feats(feats, ref_poly, components_list=None, parallel_process=False, return_dict=None, return_dict_key=None):
    # create a new poly once for all feats
    # this new poly is used and modifed by "compute_volumes_feat"
    newpoly = vtk.vtkPolyData()
    newpoly.DeepCopy(ref_poly)

    # loop through meshes. For each one, compute volume of each chamber 
    # in components_list and append it to the corresponding list in "all_volumes"
    all_volumes = {}
    for feat in feats:
        component_volumes, _ = compute_volumes_feat(feat, newpoly, components_list)
        for c, v in component_volumes.items():
            if c in all_volumes:
                all_volumes[c].append(v)
            else:
                all_volumes[c] = [v]
    if parallel_process:
        return_dict[return_dict_key] = all_volumes
    else:
        return all_volumes

# -------------------------------------------- Disks method EF -------------------------------------------
def get_4ch_slicing_plane(poly):
    # read indices of annulus points and apex cells (LV and RV)
    lv_annulus_data = pd.read_csv("lv_annulus_cells.csv")
    lv_annulus_cells_ids = list(lv_annulus_data["vtkOriginalCellIds"])

    lv_apex_data = pd.read_csv("lv_apex_cells.csv")
    lv_apex_cells_ids = list(lv_apex_data["vtkOriginalCellIds"])

    rv_apex_data = pd.read_csv("rv_apex_cells.csv")
    rv_apex_cells_ids = list(rv_apex_data["vtkOriginalCellIds"])

    # extract annulus and apex cells of LV and RV
    lv_component_id = constants.CONRAD_COMPONENTS_TO_IDS["leftVentricle"]
    rv_component_id = constants.CONRAD_COMPONENTS_TO_IDS["rightVentricle"]
    lv, lv_annulus_poly = extract_component_cells(poly, lv_component_id, lv_annulus_cells_ids)
    lv, lv_apex_poly = extract_component_cells(poly, lv_component_id, lv_apex_cells_ids)
    rv, rv_apex_poly = extract_component_cells(poly, rv_component_id, rv_apex_cells_ids)

    # get mean points for annulus and apex of LV and RV
    lv_annulus_mean = get_mean_point([lv_annulus_poly])
    lv_apex_mean = get_mean_point([lv_apex_poly])
    rv_apex_mean = get_mean_point([rv_apex_poly])

    # get normal and origin
    v1 = lv_annulus_mean - lv_apex_mean  # vector in the slicing plane
    v2 = rv_apex_mean - lv_apex_mean  # another vector in the slicing plane
    normal = np.cross(v1, v2)  # use cross product to get vector perpendicular to both v1 and v2
    normal = normal / np.linalg.norm(normal)  # normalize vector
    origin = (
                         lv_annulus_mean + lv_apex_mean) / 2  # origin in middle between apex and annulus center

    return origin, normal, (lv_annulus_mean, lv_apex_mean, rv_apex_mean), lv, rv


def disks_volume_from_segs(views_segs, view_pair):
    view1_segs = views_segs[view_pair[0]]
    view2_segs = views_segs[view_pair[1]]

    # extract lax from view1 and view2
    lax1 = view1_segs[0]
    lax2 = view2_segs[0]
    view1_segs = view1_segs[1:]
    view2_segs = view2_segs[1:]
    assert len(view1_segs) == len(view2_segs)
    nb_segs = len(view1_segs)

    lax1_p1, lax1_p2 = np.array(lax1[0]), np.array(lax1[1])
    lax2_p1, lax2_p2 = np.array(lax2[0]), np.array(lax2[1])
    length1 = np.linalg.norm(lax1_p1 - lax1_p2)
    length2 = np.linalg.norm(lax2_p1 - lax2_p2)
    if view_pair[0] == "4CH":
        length = length1
    else:
        length = length2
    # length = (length1 + length2) / 2
    height = length / nb_segs

    vol = 0
    for seg1, seg2 in zip(view1_segs, view2_segs):
        seg1_p1, seg1_p2 = np.array(seg1[0]), np.array(seg1[1])
        seg2_p1, seg2_p2 = np.array(seg2[0]), np.array(seg2[1])
        seg1_length = np.linalg.norm(seg1_p1 - seg1_p2)
        seg2_length = np.linalg.norm(seg2_p1 - seg2_p2)

        vol += height * np.pi * (seg1_length / 2) * (seg2_length / 2)
    return vol


def get_disk_method_ef(ed_feats, es_feats, ref_poly, view_pair, slicing_ref_frame="EDF",
                       output_dir=None):
    """
    Finds approximately the left ventricle long axis in the 3D mesh of the ED Frame or ES Frame
    Finds 4CH slicing plane passing through the long axis at ED or ES
    Finds 2CH slicing plane from 4CH slicing plane
    For each view in view_pair:
        Slices the LV mesh at ED and ES using the view's slicing plane
        Get the segments (long axis and other segments) from the LV view for ED and ES
    Computes the EDV and ESV from the ED and ES segments
    """

    def draw_hull(frame, hull, color=(255, 255, 255)):
        # draw hull on edf
        hull_rot = np.concatenate([hull[1:], hull[0, None]], axis=0)
        for p1, p2 in zip(hull, hull_rot):
            p1 = tuple(p1)
            p2 = tuple(p2)
            frame = cv2.line(frame, p1, p2, color, 1)
        return frame

    # get mesh from feats
    ed_mesh = overwrite_vtkpoly(ref_poly, points=ed_feats)
    es_mesh = overwrite_vtkpoly(ref_poly, points=es_feats)

    if slicing_ref_frame == "EDF":
        origin, normal_4ch, points, _, _ = get_4ch_slicing_plane(ed_mesh)
    elif slicing_ref_frame == "ESF":
        origin, normal_4ch, points, _, _ = get_4ch_slicing_plane(es_mesh)

    lax_p1, lax_p2, _ = points
    lax_vect = lax_p2 - lax_p1  # vector around which to rotate the normal
    lax_vect = lax_vect / np.linalg.norm(lax_vect)  # normalize

    # rotate 4CH normal by 60 degrees around lax_vect to get 2CH normal
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(60, lax_vect)
    normal_2ch = np.array(transform.TransformPoint(normal_4ch))
    normal_2ch = normal_2ch / np.linalg.norm(normal_2ch)  # normalize

    edv = 0
    esv = 0
    for which_frame in ["EDF", "ESF"]:
        if which_frame == "EDF":
            mesh = ed_mesh
        elif which_frame == "ESF":
            mesh = es_mesh

        views_segs = {}
        for view in view_pair:
            assert view == "4CH" or view == "2CH"
            if view == "4CH":
                normal = normal_4ch
            elif view == "2CH":
                normal = normal_2ch
            if view in views_segs:  # do not recompute view segments
                continue
            # Slice LV ED Mesh and ES Mesh using slicing plane and render (get image)
            lv_component_id = constants.CONRAD_COMPONENTS_TO_IDS["leftVentricle"]
            mesh_lv = extract_component(mesh, lv_component_id)
            slice_lv = slice_poly(mesh_lv, origin, normal)

            lv_frame = render_poly_lax_vertical(slice_lv, origin, normal, -lax_vect)
            lax_bottom_p1, lax_bottom_p2, lax_top, hull = get_lax_points(lv_frame)
            lax_bottom_mid = np.rint((lax_bottom_p1 + lax_bottom_p2) / 2).astype('int32')

            # turn to black and white
            lv_frame = cv2.cvtColor(lv_frame, cv2.COLOR_BGR2GRAY)
            lv_frame = cv2.threshold(lv_frame, 0, 255, cv2.THRESH_BINARY)[1]
            # draw bottom line
            lv_frame = cv2.line(lv_frame, tuple(lax_bottom_p1), tuple(lax_bottom_p2), 255, 1)

            nb_segs = 20
            lv_segs = get_mesh_lv_segments(lv_frame, lax_top, lax_bottom_mid, nb_segs)
            if lv_segs is None:  # error occurred while finding some segment, use convex hull for finding segments
                lv_frame = draw_hull(lv_frame, hull)
                lv_segs = get_mesh_lv_segments(lv_frame, lax_top, lax_bottom_mid, nb_segs)

            # add lax to list of segments
            lax_seg = (lax_top, lax_bottom_mid)
            lv_segs = [lax_seg] + lv_segs

            views_segs[view] = lv_segs

            if output_dir is not None:
                # colors
                BLUE = (255, 0, 0)
                GREEN = (0, 255, 0)
                RED = (0, 0, 255)
                WHITE = (255, 255, 255)
                GREY = (128, 128, 128)
                ORANGE = (0, 215, 255)
                circle_marker_r = 2
                circle_marker_thickness = 5
                # turn to tuples
                lax_top = tuple(lax_top)
                lax_bottom_p1 = tuple(lax_bottom_p1)
                lax_bottom_p2 = tuple(lax_bottom_p2)
                lax_bottom_mid = tuple(lax_bottom_mid)
                # turn to color
                lv_frame = cv2.cvtColor(lv_frame, cv2.COLOR_GRAY2BGR)
                lv_frame = draw_hull(lv_frame, hull, GREEN)

                # draw lax
                lv_frame = cv2.line(lv_frame, lax_top, lax_bottom_mid, RED, 1)

                lv_frame = cv2.circle(lv_frame, lax_bottom_p1, circle_marker_r, BLUE,
                                      circle_marker_thickness)
                lv_frame = cv2.circle(lv_frame, lax_bottom_p2, circle_marker_r, RED,
                                      circle_marker_thickness)
                lv_frame = cv2.circle(lv_frame, lax_top, circle_marker_r, GREEN,
                                      circle_marker_thickness)
                lv_frame = cv2.circle(lv_frame, lax_bottom_mid, circle_marker_r, WHITE,
                                      circle_marker_thickness)

                # draw lv segs
                for seg in lv_segs:
                    p1, p2 = seg
                    lv_frame = cv2.line(lv_frame, tuple(p1), tuple(p2), ORANGE, 1)

                images_dir = output_dir / "views" / view
                images_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(images_dir / f"{which_frame}_{view}.png"), lv_frame)

        vol = disks_volume_from_segs(views_segs, view_pair)
        if which_frame == "EDF":
            edv = vol
        elif which_frame == "ESF":
            esv = vol
    if esv > edv:
        tmp = edv
        edv = esv
        esv = tmp
    ef = ((edv - esv) / edv) * 100 if edv != 0 else -1
    return ef, edv, esv


def mesh_ef_disks_method(echo_filename, view_pair, mesh_feats, edf, esf, mesh_ref_poly,
                         mesh_dataset_scale, mesh_dataset_min, parallel_process=False,
                         return_dict=None):
    mesh_feats = (mesh_feats * mesh_dataset_scale) + mesh_dataset_min

    ed_feats = mesh_feats[edf]
    es_feats = mesh_feats[esf]
    ef_disks, _, _ = get_disk_method_ef(ed_feats, es_feats, mesh_ref_poly, view_pair)

    if parallel_process:
        return_dict[echo_filename] = ef_disks
    else:
        return ef_disks

# -------------------------------------------- EF Vol and Disks -------------------------------------------
def compute_ef_mesh_vid(feats, mesh_data_handler, ef_disks_params=None, parallel_process=False, return_dict=None, return_dict_key=None):

    # mesh dataset params
    dataset_scale = mesh_data_handler.dataset_scale
    dataset_min = mesh_data_handler.dataset_min
    ref_poly = mesh_data_handler.reference_poly

    # denormalize feats
    feats = (feats * dataset_scale) + dataset_min

    # compute LV volumes and EF_Vol
    volumes_lv = compute_volumes_feats(feats, ref_poly, ["leftVentricle"])["leftVentricle"]

    max_volume = max(volumes_lv)
    min_volume = min(volumes_lv)
    ef_vol = ((max_volume - min_volume) / max_volume) * 100.0 if max_volume != 0 else -1

    if ef_disks_params is not None: # compute the EF_Disks as well
        view_pair, slicing_ref_frame = ef_disks_params["view_pair"], ef_disks_params["slicing_ref_frame"]
        ed_idx, es_idx = np.argmax(volumes_lv), np.argmin(volumes_lv)
        ed_feats, es_feats = feats[ed_idx], feats[es_idx]
        ef_disks, _, _ = get_disk_method_ef(ed_feats, es_feats, ref_poly, view_pair, slicing_ref_frame)
    else:
        ef_disks = -1

    if parallel_process:
        return_dict[return_dict_key] = (ef_vol, ef_disks)
    else:
        return ef_vol, ef_disks

# ------------------------------------------- IoU functions ----------------------------------------------

def trace_lax(frame, lax, p1_color, p2_color, lax_color, frame_color=None):
    # frame is grayscale
    circle_marker_r = 1
    circle_marker_thickness = 1
    line_thickness = 1

    if frame_color is not None:
        b, g, r = frame_color
        _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)
        frame = frame // 255
        frame = [frame * b, frame * g, frame * r]
        frame = cv2.merge(frame, 3)
    else:
        _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    frame = cv2.circle(frame, tuple(lax[0]), circle_marker_r, p1_color, circle_marker_thickness)
    frame = cv2.circle(frame, tuple(lax[1]), circle_marker_r, p2_color, circle_marker_thickness)
    frame = cv2.line(frame, tuple(lax[0]), tuple(lax[1]), lax_color, line_thickness)

    return frame


def get_outline(frame):
    outline = np.zeros(frame.shape, dtype=np.uint8)

    for i, row in enumerate(frame):
        white_pixels = np.where(row == 255)[0]
        if len(white_pixels) > 0:
            min_x = white_pixels.min()
            max_x = white_pixels.max()
            outline[i, min_x] = 255
            outline[i, max_x] = 255

    return outline


def mesh_echo_iou_and_dice(echo_filename, echo_frames, echo_laxes, mesh_feats, mesh_data_handler,
                           mean_mesh_iou_dice, output_size, use_volumes=False, compute_ef=False,
                           parallel_process=False, return_dict=None, output_dir=None):
    # computing IoU and Dice on EDF and ESF

    mesh_dataset_scale = mesh_data_handler.dataset_scale
    mesh_dataset_min = mesh_data_handler.dataset_min
    mesh_ref_poly = mesh_data_handler.reference_poly

    mean_meshes = mesh_data_handler._mean_polydata
    mean_ed_mesh = mean_meshes[0]
    mean_es_mesh = mean_meshes[4]
    mean_meshes = [mean_ed_mesh, mean_es_mesh]

    mesh_feats = (mesh_feats * mesh_dataset_scale) + mesh_dataset_min

    if use_volumes:
        lv_volumes = \
        compute_volumes_feats(mesh_feats, mesh_ref_poly, components_list=["leftVentricle"])[
            "leftVentricle"]

        ed_mesh_feats = mesh_feats[np.argmax(lv_volumes)]
        es_mesh_feats = mesh_feats[np.argmin(lv_volumes)]

        mesh_feats = [ed_mesh_feats, es_mesh_feats]

    if compute_ef:
        # Compute EF using volumes
        lv_volumes = compute_volumes_feats(mesh_feats, mesh_ref_poly, components_list=["leftVentricle"])["leftVentricle"]
        edv_3d = max(lv_volumes)
        esv_3d = min(lv_volumes)
        ef_vol = ((edv_3d - esv_3d) / edv_3d) * 100 if edv_3d !=0 else -1

        # compute EF using biplane method
        view_pair = ("4CH", "2CH")
        ef_biplane, edv_biplane, esv_biplane = get_disk_method_ef(mesh_feats[0], mesh_feats[1], mesh_ref_poly, view_pair)
        
        efs = [ef_vol, ef_biplane]
        volumes = [edv_3d, esv_3d, edv_biplane, esv_biplane]

    else:
        efs = [-1, -1]
        volumes = [-1, -1, -1, -1]

    ed_mesh = overwrite_vtkpoly(mesh_ref_poly, points=mesh_feats[0])
    origin, normal, points, _, _ = get_4ch_slicing_plane(ed_mesh)

    if mean_mesh_iou_dice:
        # get mean mesh slicing plane
        origin_mm, normal_mm, _, _, _ = get_4ch_slicing_plane(mean_ed_mesh)

    lax_p1, lax_p2, _ = points
    lax_vect = lax_p2 - lax_p1  # vector around which to rotate the normal
    lax_vect = lax_vect / np.linalg.norm(lax_vect)  # normalize

    mesh_frames = []
    mesh_laxes = []
    mean_mesh_frames = []

    for idx, mesh_feat in enumerate(mesh_feats):
        mesh = overwrite_vtkpoly(mesh_ref_poly, points=mesh_feat)
        slicing_plane = {"origin": origin, "normal": normal}
        camera_params = (origin, normal, -lax_vect)
        mesh_frame, mesh_lax = render_lv_get_lax(mesh, slicing_plane, camera_params, output_size, draw_bottom_line=True, fill=True)
        # save frame and lax
        mesh_frames.append(mesh_frame)
        mesh_laxes.append(mesh_lax)
        if mean_mesh_iou_dice:
            # render mean mesh
            slicing_plane = {"origin": origin_mm, "normal": normal_mm}
            camera_params = (origin_mm, normal_mm, -lax_vect)
            mean_mesh_frame, _ = render_lv_get_lax(mean_meshes[idx], slicing_plane, camera_params, output_size, draw_bottom_line=True, fill=True)
            mean_mesh_frames.append(mean_mesh_frame)
        else:
            mean_mesh_frames.append(None)  # no mean mesh frame

    # get translation, rotation and scale matrices
    echo_edf_lax, echo_esf_lax = echo_laxes
    mesh_edf_lax, mesh_esf_lax = mesh_laxes
    echo_frame_size = mesh_frame_size = output_size
    rot_scale_mat, translation_mat_edf = get_transform_matrices(mesh_edf_lax, echo_edf_lax,
                                                                mesh_frame_size, echo_frame_size)
    _, translation_mat_esf = get_transform_matrices(mesh_esf_lax, echo_esf_lax, mesh_frame_size,
                                                    echo_frame_size)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    IoUs = []
    Dices = []
    IoUs_MM = []
    Dices_MM = []
    for i, (echo_frame, mesh_frame, mean_mesh_frame) in enumerate(
            zip(echo_frames, mesh_frames, mean_mesh_frames)):

        translation_mat = translation_mat_edf

        original_frames = (echo_frame, mesh_frame, mean_mesh_frame)

        # transform mesh_frame by translation, rotation and scaling then resize
        mesh_frame = cv2.warpAffine(src=mesh_frame, M=rot_scale_mat, dsize=mesh_frame_size)
        mesh_frame = cv2.warpAffine(src=mesh_frame, M=translation_mat, dsize=mesh_frame_size)
        mesh_frame = cv2.threshold(mesh_frame, 0, 255, cv2.THRESH_BINARY)[1]

        # mesh IoU and Dice
        IoU, Dice = utils.compute_iou_and_dice(mesh_frame, echo_frame)
        IoUs.append(IoU)
        Dices.append(Dice)

        if mean_mesh_iou_dice:
            # transform mean_mesh_frame by translation, rotation and scaling then resize
            mean_mesh_frame = cv2.warpAffine(src=mean_mesh_frame, M=rot_scale_mat,
                                             dsize=mesh_frame_size)
            mean_mesh_frame = cv2.warpAffine(src=mean_mesh_frame, M=translation_mat,
                                             dsize=mesh_frame_size)
            mean_mesh_frame = cv2.threshold(mean_mesh_frame, 0, 255, cv2.THRESH_BINARY)[1]
            # mean mesh IoU and Dice
            IoU, Dice = utils.compute_iou_and_dice(mean_mesh_frame, echo_frame)
            IoUs_MM.append(IoU)
            Dices_MM.append(Dice)
        else:  # do not compute mean mesh iou and dice
            IoUs_MM.append(-1)
            Dices_MM.append(-1)

        transformed_frames = (echo_frame, mesh_frame, mean_mesh_frame)

        # save image overlaying echo tracing, mesh slice (and mean mesh slice)
        if output_dir is not None:
            # original frames
            echo_frame, mesh_frame, mean_mesh_frame = original_frames

            mesh_frame_outline = get_outline(mesh_frame)
            if mean_mesh_frame is not None:
                mean_mesh_frame_outline = get_outline(mean_mesh_frame)

            frames = []
            # trace laxes in original frames
            echo_lax = echo_laxes[i]
            mesh_lax = mesh_laxes[i]
            frames.append(trace_lax(echo_frame, echo_lax, utils.RED, utils.BLUE, utils.GREEN))
            frames.append(trace_lax(mesh_frame_outline, mesh_lax, utils.RED, utils.BLUE, utils.VIOLET_BLUE, utils.VIOLET_BLUE))
            if mean_mesh_frame is not None:
                frames.append(utils.black_white_to_color(mean_mesh_frame_outline, utils.ORANGE))

            # transformed frames
            echo_frame, mesh_frame, mean_mesh_frame = transformed_frames

            # get outline
            mesh_frame_outline = get_outline(mesh_frame)
            if mean_mesh_frame is not None:
                mean_mesh_frame_outline = get_outline(mean_mesh_frame)

            # transform mesh lax by rot_scale and translate, then resize
            mesh_lax = mesh_laxes[i]
            mesh_lax_top_transformed = utils.transform_point(mesh_lax[0],
                                                             [rot_scale_mat, translation_mat])
            mesh_lax_bottom_transformed = utils.transform_point(mesh_lax[1],
                                                                [rot_scale_mat, translation_mat])
            mesh_lax_transformed = (mesh_lax_top_transformed, mesh_lax_bottom_transformed)
            frames.append(trace_lax(mesh_frame_outline, mesh_lax_transformed, utils.RED, utils.BLUE,
                                    utils.GREEN, utils.GREEN))
            if mean_mesh_frame is not None:
                frames.append(utils.black_white_to_color(mean_mesh_frame_outline, utils.PINK))

            # overlay frames on each other and save
            output_frame = frames[0]

            for frame in frames[1:]:
                output_frame = utils.overlay_non_black_pixels(output_frame, frame)

            if i == 0:
                phase = "end-diastole"
            else:
                phase = "end-systole"
            overlays_dir = output_dir / "overlays" / phase
            overlays_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlays_dir / f"{echo_filename}.png"), output_frame)
    
    results = [IoUs, Dices, IoUs_MM, Dices_MM, efs, volumes]
    
    if parallel_process:
        return_dict[echo_filename] = results
    else:
        return results


# -------------------------------------- Other (Helper) Functions ----------------------------------------
def rotate_points_lax_vertical(mesh_lax_top, mesh_lax_bottom, points):
    mesh_lax_mid = np.rint((mesh_lax_top + mesh_lax_bottom) / 2).astype(np.int32)
    v2 = mesh_lax_top - mesh_lax_bottom
    v1 = np.asarray((0, -1))
    angle = utils.signed_angle_v2_to_v1(v1, v2)
    mesh_lax_mid = (int(mesh_lax_mid[0]), int(mesh_lax_mid[1]))
    rot_mat = cv2.getRotationMatrix2D(center=mesh_lax_mid, angle=angle, scale=1)
    inverse_rot_mat = cv2.getRotationMatrix2D(center=mesh_lax_mid, angle=-angle, scale=1)
    new_points = []
    for point in points:
        new_point = utils.rotate_scale_point(point, rot_mat)
        new_points.append(new_point)

    return rot_mat, inverse_rot_mat, new_points


def get_mesh_lv_segments(mesh_lv, mesh_lax_top, mesh_lax_bottom, nb_segs=20):
    # find segments
    points = [mesh_lax_top, mesh_lax_bottom]
    rot_mat, inverse_rot_mat, (mesh_lax_top, mesh_lax_bottom) = rotate_points_lax_vertical(
        mesh_lax_top, mesh_lax_bottom, points)
    # rotate the mesh lv image then get the lv border points row by row
    width, height = mesh_lv.shape[1], mesh_lv.shape[0]
    mesh_lv = cv2.warpAffine(src=mesh_lv, M=rot_mat, dsize=(width, height))
    mesh_lv = cv2.threshold(mesh_lv, 0, 255, cv2.THRESH_BINARY)[
        1]  # threshold to only have black and white pixels

    lax_length = mesh_lax_bottom[1] - mesh_lax_top[1]

    # get row indices of the segments
    step = lax_length / nb_segs
    row = mesh_lax_top[1] + step
    row_indices = []
    for i in range(nb_segs):
        row_idx = np.rint(row).astype("int32")
        row_indices.append(row_idx)
        row += step

    # return mesh_lv
    # pick segment points
    segs = []
    for row_idx in row_indices:
        row = mesh_lv[row_idx]
        pixel_indices = np.argwhere(row == 255)
        if len(pixel_indices) == 0:
            return None
        groups = utils.consecutive_groups(pixel_indices)
        left_x = np.rint(np.mean(groups[0])).astype("int32")
        right_x = np.rint(np.mean(groups[-1])).astype("int32")
        left = (left_x, row_idx)
        right = (right_x, row_idx)
        left = utils.rotate_scale_point(left, inverse_rot_mat)
        right = utils.rotate_scale_point(right, inverse_rot_mat)
        segs.append((left, right))

    return segs


def get_lax_points(lv_countour, is_thresholded=False):
    if not is_thresholded:
        lv_countour = cv2.cvtColor(lv_countour, cv2.COLOR_BGR2GRAY)
        _, lv_countour = cv2.threshold(lv_countour, 0, 255, cv2.THRESH_BINARY)

    # (x, y) coordinates of white pixels
    pixel_indices = np.argwhere(lv_countour == 255)  # indices are (y, x)
    pixel_indices = np.concatenate([pixel_indices[:, 1, None], pixel_indices[:, 0, None]],
                                   axis=1)  # swap colums

    # hull points in counter-clockwise direction
    hull = np.squeeze(cv2.convexHull(pixel_indices))
    # create list of point pairs representing the hull lines
    hull_rot = np.concatenate([hull[1:], hull[0, None]], axis=0)
    points_pairs = list(zip(hull, hull_rot))
    # filter point pairs, only keep those below the mid line (horizontal line in 
    # the middle between the top and bottom point of the hull)
    y_dist = hull[:, 1].max() - hull[:, 1].min()
    mid_y = np.rint(hull[:, 1].min() + y_dist / 2).astype("int32")
    points_pairs = [pair for pair in points_pairs if pair[0][1] >= mid_y and pair[1][1] >= mid_y]
    # find point pair that is furtherest apart in the x-axis
    x_dists = []
    for p1, p2 in points_pairs:  # for each point pair
        x1 = p1[0]
        x2 = p2[0]
        x_dists.append(np.abs(x1 - x2))  # calculate the distance in the x-axis
    points_pairs.sort
    idx = np.argmax(x_dists)  # index of points pair furtherest apart in the x-axis
    lax_bottom_p1, lax_bottom_p2 = points_pairs[idx]

    # find top point
    top_y = np.min(pixel_indices[:, 1])
    top_row = lv_countour[top_y]
    indices = np.argwhere(top_row == 255)
    top_x = np.rint((np.min(indices) + np.max(indices)) / 2).astype('int32')
    lax_top = np.array((top_x, top_y))
    return lax_bottom_p1, lax_bottom_p2, lax_top, hull

def render_poly_lax_vertical(poly, origin, normal, lax_vect, size=None):
    if size is None:
        size = (500, 500)

    camera_params = {"position": origin - 300 * normal,
                     "focal_point": -normal,
                     "up_vect": lax_vect} # long axis vector to orient the up vector of the camera

    image = render_poly_from_camera_params(poly, camera_params, size, parallel_zoom=True)
    return image

def render_poly_from_camera_params(poly, camera_params, size, colors=None, parallel_zoom=False, p_zoom=90, return_bgr=False):

    avail_comps = np.unique(vtk_to_numpy(poly.GetPointData().GetArray('components'))).astype(np.int32)
    # set up rendered and add polys to render to it
    renderer = vtk.vtkRenderer()
    for comp_id in avail_comps:
        comp_name = constants.CONRAD_IDS_TO_COMPONENTS[comp_id]
        if colors is not None and comp_name in colors:
            color = colors[comp_name]
        else:
            color = (255, 255, 255)

        comp = extract_component(poly, comp_id)
        comp.GetPointData().RemoveArray("components")
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(comp)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(color[0]/255, color[1]/255, color[2]/255)
        renderer.AddActor(actor)

    if colors is not None and "background" in colors:
        bg_color = colors["background"]
    else:
        bg_color = (0, 0, 0)

    renderer.SetBackground(bg_color[0]/255, bg_color[1]/255, bg_color[2]/255)  # Background color
    # set up render window
    render_window = vtk.vtkRenderWindow()
    render_window.OffScreenRenderingOn()
    render_window.AddRenderer(renderer)
    render_window.SetSize(size)

    camera = vtk.vtkCamera()
    camera.SetViewUp(camera_params["up_vect"])
    camera.SetPosition(camera_params["position"])
    camera.SetFocalPoint(camera_params["focal_point"])
    if parallel_zoom:
        camera.ParallelProjectionOn()
        camera.SetParallelScale(p_zoom)

    renderer.SetActiveCamera(camera)
    render_window.Render()

    vtk_win_im = vtk.vtkWindowToImageFilter()
    vtk_win_im.SetInput(render_window)
    vtk_win_im.Update()
    vtk_image = vtk_win_im.GetOutput()

    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    image = vtk_to_numpy(vtk_array).reshape(height, width, components)
    image = np.fliplr(image)
    if return_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def render_lv_get_lax(mesh, slicing_plane, camera_params, output_size=None, draw_bottom_line=False, fill=False):
    if output_size is None:
        output_size = (500, 500)
    
    lv_component_id = constants.CONRAD_COMPONENTS_TO_IDS["leftVentricle"]
    mesh_lv = extract_component(mesh, lv_component_id)
    mesh_slice_lv = slice_poly(mesh_lv, slicing_plane["origin"], slicing_plane["normal"])
    origin, normal, up_vect = camera_params
    mesh_lv_frame = render_poly_lax_vertical(mesh_slice_lv, origin, normal, up_vect, size=output_size)
    # turn to black and white
    mesh_lv_frame = cv2.cvtColor(mesh_lv_frame, cv2.COLOR_BGR2GRAY)
    mesh_lv_frame = cv2.threshold(mesh_lv_frame, 0, 255, cv2.THRESH_BINARY)[1]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mesh_lv_frame = cv2.morphologyEx(mesh_lv_frame, cv2.MORPH_GRADIENT, kernel)

    assert len(np.unique(mesh_lv_frame)) == 2
    
    mesh_lax_bottom_p1, mesh_lax_bottom_p2, mesh_lax_top, mesh_hull = get_lax_points(mesh_lv_frame, is_thresholded=True)
    mesh_lax_bottom_mid = np.rint((mesh_lax_bottom_p1 + mesh_lax_bottom_p2) / 2).astype('int32')

    if draw_bottom_line: # draw bottom line
        mesh_lv_frame = cv2.line(mesh_lv_frame, tuple(mesh_lax_bottom_p1), tuple(mesh_lax_bottom_p2), 255, 1)
    if fill: # fill contour
        mesh_lv_frame = utils.fill_contour(mesh_lv_frame)
    
    mesh_lax = (mesh_lax_top, mesh_lax_bottom_mid)

    return mesh_lv_frame, mesh_lax


def slice_poly(poly, origin, normal):
    # cut heart
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin[0], origin[1], origin[2])
    plane.SetNormal(normal[0], normal[1], normal[2])

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)

    cutter.SetInputData(poly)
    cutter.Update()
    return cutter.GetOutput()


def extract_component(poly, component):
    thresholdfilter = vtk.vtkThreshold()
    thresholdfilter.SetInputData(poly)
    thresholdfilter.ThresholdBetween(component, component)
    thresholdfilter.Update()
    component_poly = thresholdfilter.GetOutput()
    return component_poly


def get_mean_point(polys):
    all_points = []
    for poly in polys:
        all_points.append(vtk_to_numpy(poly.GetPoints().GetData()))
    all_points = np.concatenate(all_points, axis=0)
    return np.mean(all_points, axis=0)


def extract_component_cells(poly, component, cells_ids):
    thresholdfilter = vtk.vtkThreshold()
    thresholdfilter.SetInputData(poly)
    thresholdfilter.ThresholdBetween(component, component)
    thresholdfilter.Update()
    component_poly = thresholdfilter.GetOutput()

    # extract selection
    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
    selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
    selectionNode.SetSelectionList(numpy_to_vtk(cells_ids))

    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)

    extractSelectedIds = vtk.vtkExtractSelectedIds()
    extractSelectedIds.SetInputConnection(0, thresholdfilter.GetOutputPort())
    extractSelectedIds.SetInputData(1, selection)
    extractSelectedIds.Update()

    extracted_poly = extractSelectedIds.GetOutput()

    return component_poly, extracted_poly

def save_feats_as_vtps(feats, mesh_data_handler, output_dir, prefix):
    ref_poly = mesh_data_handler.reference_poly

    output_dir.mkdir(parents=True, exist_ok=True)
    for j, feat in enumerate(feats):  # for each feat

        filename = prefix + f"_{str(j).zfill(3)}"
        overwrite_vtkpoly(ref_poly, points=feat, save=True, output_dir=output_dir, name=filename)


def load_polydata(vtk_path):
    """Load vtk or vtp mesh. (vtk_path must be a Path object)"""

    # We assume .vtp is in xml format, otherwise it's the old format
    if not vtk_path.exists():
        raise FileNotFoundError(f'Following file does not exist: {vtk_path}')
    if vtk_path.suffix == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
    reader.SetFileName(str(vtk_path))
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


# save vtk poly to .vtp file (which is an xml file)
def write_vtk_xml(polydata, save_path=None, overwrite=False, **saving_params):
    """Saves `polydata` into `save_path` as .vtp file.

    If `save_path` is not given then the `name` is appended to the global
    `output_dir`. If `name` is also omitted a random string is generated to
    replace the `name`.

    :param overwrite:
    :param save_path:
    :type save_path:
    :param polydata: vtk polydata to be saved
    :type polydata: vtkPolyData

    """

    if not ('save' in saving_params) or not saving_params['save']:
        logger.debug('Not saving')
        return None

    if save_path:
        out_file = save_path
    elif 'name' in saving_params:
        out_file = (saving_params['output_dir'] /
                    f"{saving_params['name']}.vtp")
    else:
        name = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=5))
        out_file = saving_params['output_dir'] / f"{name}.vtp"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not out_file.exists() or overwrite:
        # if out_file.exists():
        #     logger.warning('Overwriting file')
        # logger.debug("Saving vtk polygone to disk...")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(out_file))
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)
        writer.SetDataModeToAscii()
        writer.Write()
        # logger.debug(f"vtk polygone saved to {out_file}")
        return out_file
    else:
        logger.warning(f"file {out_file} already exists. Kept the old one.")
        return out_file


def generate_shape_from_mode(polydata, pc_i, eigen, std_num):
    newpoly = vtk.vtkPolyData()
    newpoly.DeepCopy(polydata)

    points = newpoly.GetPoints()
    n_points = newpoly.GetNumberOfPoints()

    # Compute the varying shape
    for j in range(0, n_points):
        p = points.GetPoint(j)
        p += std_num * eigen * pc_i[j]
        points.SetPoint(j, p)
    return newpoly


# apply weight to each PC then add them to the mean shape vertices
def generate_shape_from_modes(mean_polydata, pcs, eigens, weights):
    """Generate a new polydata based on `mean_polydata` and eigen information

    Each new point in the new polydata will be the addition of the point in
    `mean_polydata` and a linear combination of principal components `pcs`
    weighted by eigenvalues `eigens` and `weights`.
    For n points and m principal components we have:

    p_i = p_old_i + (pcs_i_0 * eigens_0 * w_0 + ... + pcs_i_m * eigens_m * w_m)

    Therefore `pcs` is matrix of n x m and `eigens` and `weights` are arrays
    of length m.

    When m = 0, then this function behaves as the same as
    `generate_shape_from_mode`.

    :param mean_polydata: The mean vtk polydata
    :type mean_polydata: vtk.vtkPolyData
    :param pcs: n x m array where n is the same as number of points in
    `mean_polydata` and m is the number of principal components.
    :type pcs: list of list or np.array of np.array
    :param eigens: eigenvalues corresponding to eigenvectors `pcs`. Array of
    length m
    :type eigens: list or np.array
    :param weights:(optional) weights multiplied by eigenvalues `eigens`.
    Array of length m
    :type weights: list or np.array
    :return: new generated vtk polydata
    """
    pcs = np.array(pcs)
    eigens = np.array(eigens)
    weights = np.array(weights)
    n_pcs = len(pcs)
    assert n_pcs == len(eigens)
    assert weights is None or n_pcs == len(weights)

    if weights is None:
        weights = np.ones(n_pcs)

    # Create new poly
    newpoly = vtk.vtkPolyData()
    newpoly.DeepCopy(mean_polydata)
    vtk_points = vtk.vtkPoints()

    points = vtk_to_numpy(newpoly.GetPoints().GetData())
    n_points = newpoly.GetNumberOfPoints()

    w = eigens * weights

    tile_w = np.tile(w, [3, n_points, 1]).T
    c = np.sum(pcs * tile_w, axis=0)
    new_points = points + c

    # Set the new points
    vtk_points.SetData(numpy_to_vtk(new_points))
    newpoly.SetPoints(vtk_points)

    return newpoly


# get poly's points
def _get_poly_scalars(polydata):
    return vtk_to_numpy(polydata.GetPointData().GetScalars())


# get poly's normal vectors
def _get_poly_normals(polydata):
    return vtk_to_numpy(polydata.GetPointData().GetNormals())


# convert vtk list of ids to python list of ids
def _vtkidlist_to_list(vtkidlist):
    id_list = []
    for idx in range(vtkidlist.GetNumberOfIds()):
        id_list.append(vtkidlist.GetId(idx))
    return id_list


def vtkpoly_to_adj_feats(polydata, verbose=False, **saving_params):
    logger.debug("Getting adjacency matrix...")
    polys = polydata.GetPolys()  # the triangles (polygones)
    points = polydata.GetPoints()  # the vertices
    n_polys = polydata.GetNumberOfPolys()  # the number triangles
    n_points = polydata.GetNumberOfPoints()  # the number of vertices

    if verbose:
        logger.debug(f'n_polys: {n_polys}')
        logger.debug(f'n_points: {n_points}')
    assert n_polys == polys.GetNumberOfCells()  # number of cells should be number of polygones/triangles since each cell is a triangle
    assert n_points == points.GetNumberOfPoints()

    a_mat = np.zeros((n_points, n_points), dtype=np.bool)  # adjacency matrix
    x_mat = np.zeros((n_points, 3), dtype=np.float32)  # 3D vertex coordinates
    polys.InitTraversal()  # traverse the list of triangles
    for _ in range(n_polys):  # for each triangle
        # get the indices of the 3 vertices of the triangle
        id_list = vtk.vtkIdList()
        polys.GetNextCell(id_list)
        point_ids = _vtkidlist_to_list(id_list)
        # create all possible pairs of these 3 vertices (6 pairs). Each pair is connected
        combs = set(combinations(point_ids, 2))
        # for each pair, set the corresponding value in the adjacency matrix to 1
        for i, j in combs:
            a_mat[i, j] = 1
            a_mat[j, i] = 1
        # get the 3 coordinates of the point at "p_id" (e,g point 5) and save 
        # the corresponding location in the features matrix x_mat
        for p_id in point_ids:
            x_mat[p_id] = points.GetPoint(p_id)

    if 'save' in saving_params and saving_params['save']:  # not used
        logger.debug("Saving matrices to disk...")

        output_dir = saving_params['output_dir']
        a_out_file = output_dir / f"{saving_params['name']}_adj"
        x_out_file = output_dir / f"{saving_params['name']}_vertices"
        np.savez_compressed(a_out_file, a_mat)
        np.savez_compressed(x_out_file, x_mat)
        logger.debug(f"Matrices saved to {a_out_file} and {x_out_file}")

    return a_mat, x_mat


def vtkpoly_to_adj(polydata):
    logger.debug("Getting adjacency matrix...")
    polys = polydata.GetPolys()  # the triangles (polygones)
    n_polys = polydata.GetNumberOfPolys()  # the number triangles

    logger.debug(f'n_polys: {n_polys}')
    assert n_polys == polys.GetNumberOfCells()  # number of cells should be number of polygones/triangles since each cell is a triangle

    a_mat = np.zeros((n_points, n_points), dtype=np.bool)  # adjacency matrix
    polys.InitTraversal()  # traverse the list of triangles
    for _ in range(n_polys):  # for each triangle
        # get the indices of the 3 vertices of the triangle
        id_list = vtk.vtkIdList()
        polys.GetNextCell(id_list)
        point_ids = _vtkidlist_to_list(id_list)
        # create all possible pairs of these 3 vertices (6 pairs). Each pair is connected
        combs = set(combinations(point_ids, 2))
        # for each pair, set the corresponding value in the adjacency matrix to 1
        for i, j in combs:
            a_mat[i, j] = 1
            a_mat[j, i] = 1

    return a_mat


def vtkpoly_to_feats(polydata, verbose=False):
    points = polydata.GetPoints()  # the vertices
    n_points = polydata.GetNumberOfPoints()  # the number of vertices

    if verbose:
        logger.debug(f'n_points: {n_points}')
    assert n_points == points.GetNumberOfPoints()

    x_mat = np.zeros((n_points, 3), dtype=np.float32)  # 3D vertex coordinates
    for i in range(n_points):
        x_mat[i] = points.GetPoint(i)

    return x_mat

# slow function, not used
def construct_triangles(a_mat, nxgraph=None):
    """Return triangles from adjacency matrix. (Extremely slow)"""

    if not nxgraph:
        nxgraph = adj_to_nxgraph(a_mat)

    triangles = set()
    start = time.time()
    for edge in nxgraph.edges:
        for vertex in nxgraph.nodes:
            if edge[0] == vertex or edge[1] == vertex:
                continue
            if a_mat[vertex][edge[0]] and a_mat[vertex][edge[1]]:
                tri = tuple([3] + sorted((vertex, edge[0], edge[1])))
                triangles.add(tri)
    end = time.time()
    logger.debug(f"{len(triangles)} triangles reconstructed in "
                 f"{end - start:.3f}s from total of {len(nxgraph.edges)} "
                 f"edges and {len(nxgraph.nodes)} nodes")

    return np.array(list(triangles))


# CONRAD Data: return linear interpolation of 2 polys
def vtkpoly_linearly_interpolated(polydata_1, polydata_2, diff):
    interpolated = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    p1 = vtk_to_numpy(polydata_1.GetPoints().GetData())
    p2 = vtk_to_numpy(polydata_2.GetPoints().GetData())

    res = p1 + diff * (p2 - p1)  # linear interpolate
    vtk_points.SetData(numpy_to_vtk(res))
    interpolated.SetPoints(vtk_points)
    return interpolated


def overwrite_vtkpoly(polydata, points=None, data_array_and_name=None, polys=None, point_data=None,
                      cell_data=None, point_scalar=None,
                      poly_scalar='part', **saving_params):
    """Returns a vtk polydata based on `polydata` where some attributes are
    overwritten.

    The returned polydata will have all the information as the based
    `polydata` except other attributes that are given as argument.

    The input `polydata` will remain unchanged.

    This function can be used to create entirely new polydata if `polydata`
    is simply a new instance of vtk polydata: `polydata = vtk.vtkPolyData()`


    :param polydata: The base polydata
    :type polydata: vtk polydata
    :param points: New coordinates of the vertices of polygon to overwrite
    the old coordinates in `polydata`
    :type points: (numpy.ndarray, list) of (numpy.ndarray, list)
    :param polys: New poly data to overwrite the old poly data in `polydata`
    :type polys: (numpy.ndarray, list) of (numpy.ndarray, list)
    :param point_data: New PointData to overwrite the old PointData in
    `polydata`
    :type point_data: dict[str: List]
    :param cell_data: New CellData to overwrite the old CellData in `polydata`
    :type cell_data: dict[str: List]
    :param point_scalar: A string indicating the name of the PointData scalars
    :type point_scalar: str
    :param poly_scalar: A string indicating the name of the PolyData scalars
    :type poly_scalar: str
    :param saving_params: parameters required by `write_vtk_xml`
    :return: vtk polydata where some attributes are overwritten
    """

    assert not point_data or isinstance(point_data, dict)
    assert not cell_data or isinstance(cell_data, dict)

    newpoly = vtk.vtkPolyData()
    newpoly.DeepCopy(polydata)

    if points is not None:
        vtk_points = vtk.vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint(point)
        newpoly.SetPoints(vtk_points)

    if data_array_and_name is not None:
        data_array, name = data_array_and_name
        arr = numpy_to_vtk(data_array)  # create a data array
        arr.SetName(name)  # name the array
        newpoly.GetPointData().AddArray(arr)  # save array to poly object
        newpoly.GetPointData().SetActiveScalars(name)  # set the data array to be the active scalar
    else:
        # set same scalars
        scalars = polydata.GetPointData().GetScalars()
        newpoly.GetPointData().SetScalars(scalars)

    if polys is not None:
        p_dim = polys.shape[1] - 1  # Assume 1st entry is the # of elements
        vtk_polys = vtk.vtkCellArray()
        vtk_polys.Allocate(len(polys), p_dim)
        for poly in polys:
            vtk_polys.InsertNextCell(p_dim)
            for p in poly[1:]:
                vtk_polys.InsertCellPoint(p)
        newpoly.SetPolys(vtk_polys)

    if point_data is not None:
        for key, value in point_data.items():
            attr = numpy_to_vtk(value)
            attr.SetName(key)
            newpoly.GetPointData().AddArray(attr)

    if point_scalar:
        newpoly.GetPointData().SetActiveScalars(point_scalar)

    if cell_data is not None:
        for key, value in cell_data.items():
            attr = numpy_to_vtk(value)
            attr.SetName(key)
            newpoly.GetCellData().AddArray(attr)

    if poly_scalar:
        newpoly.GetCellData().SetActiveScalars(poly_scalar)

    if ('save' in saving_params) and saving_params['save']:
        write_vtk_xml(newpoly, **saving_params)
    return newpoly


def decimate_poly(polydata, reduction=5000, **saving_params):
    if not reduction:  # Can be None to skip
        return polydata

    poly_num_triangles_orig = polydata.GetNumberOfPolys()
    poly_num_points_orig = polydata.GetNumberOfPoints()

    if reduction >= poly_num_points_orig:
        logger.warning("Number of vertices specified for reduction is "
                       "larger than the one in the original mesh.")
        return polydata

    # if reduction specifies the number of vertices to reduce to
    if reduction > 1:
        reduction = 1 - reduction / poly_num_points_orig

    logger.debug(f"Decimating polygon with {poly_num_triangles_orig} "
                 f"triangles and {poly_num_points_orig} points with "
                 f"reduction factor {reduction}...")

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(reduction)

    # TODO (fabian): investigate effect of these properties
    decimate.BoundaryVertexDeletionOff()
    decimate.SplittingOff()
    decimate.SetMaximumError(vtk.VTK_DOUBLE_MAX)

    # important!
    # Adj. matrix should not change for different variations of the model
    decimate.PreserveTopologyOn()

    decimate.Update()

    decimated = vtk.vtkPolyData()
    decimated.ShallowCopy(decimate.GetOutput())

    poly_num_dec = decimated.GetNumberOfPolys()
    poly_num_points = decimated.GetNumberOfPoints()
    r_fact = (poly_num_triangles_orig - poly_num_dec) / poly_num_triangles_orig

    logger.debug(f"Decimated to {poly_num_dec} triangles.")
    logger.debug(f"Reduction factor:{r_fact} vs target reduction: {reduction}")
    logger.debug(f"Number of points decimated mesh {poly_num_points}")

    write_vtk_xml(decimated, **saving_params)
    return decimated


def remove_vertices(adj_matrix, feature_matrix, n_points_to_remove):
    """ Remove entries from adj. and feature matrix"""

    assert n_points_to_remove > 0
    # selected equally indexed points from array
    n_vertices = adj_matrix.shape[0]
    points_to_remove = np.linspace(0, n_vertices - 1, n_points_to_remove)
    points_to_remove = np.round(points_to_remove).astype(int)
    adj_matrix = np.delete(adj_matrix, points_to_remove, 0)
    adj_matrix = np.delete(adj_matrix, points_to_remove, 1)
    feature_matrix = np.delete(feature_matrix, points_to_remove, 0)

    return adj_matrix, feature_matrix


def compute_point_normals(polydata, **saving_params):
    """ Compute normal vectors for each vertex of polygon

    # It must be related to feature angle
    :param polydata: vtk polydata
    :type polydata: vtkPolyData
    :param saving_params: parameters required for `write_vtk_xml`
    :return: polydata where for each vertex a normal vector is computed
    """

    logger.debug(f"computing normals...")
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.SetFeatureAngle(30.0)
    normals.SplittingOff()
    normals.ComputePointNormalsOn()
    normals.Update()

    write_vtk_xml(normals.GetOutput(), **saving_params)
    return normals.GetOutput()


def make_slices(plane_dicts, polys, stripe=False, **saving_params):
    """Make slices of all the `polys` with all the planes given by
    `plane_dicts`

    :param plane_dicts: list of dictionaries where each dictionary represents
    a plane with which the polygons are cut. Each dictionary has two keys
    defining the origin and normal of the plane:
        ``origin``
            List of three elements representing the origin of the plane
        ``normal``
            List of three elements representing the normal of the plane
    :type plane_dicts: list of dict
    :param polys: List of vtk polydata where each polydata will be cut with
    all planes
    :type list of vtkPolyData
    :param stripe: Whether to make stripe triangles on the cut data or not
    :type stripe: bool
    :param saving_params: parameters required by `write_vtk_xml`
    :return: list of list containing all the cuts.
    """

    outs = []
    for pl_i, plane in enumerate(plane_dicts):
        origin = plane['origin']
        normal = plane['normal']

        plane = vtk.vtkPlane()
        plane.SetOrigin(origin[0], origin[1], origin[2])
        plane.SetNormal(normal[0], normal[1], normal[2])

        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)

        cut_strips = vtk.vtkStripper()
        plane_outs = []
        for p_i, poly_data in enumerate(polys):
            start = time.time()
            cutter.SetInputData(poly_data)
            cutter.Update()
            cut_poly = cutter.GetOutput()
            if stripe:
                cut_strips.SetInputData(cutter.GetOutput())
                cut_strips.Update()
                cut_poly = cut_strips.GetOutput()
            end = time.time()
            plane_outs.append(cut_poly)

            # Save data
            this_saving_params = saving_params.copy()
            if 'name' in saving_params:
                this_saving_params['name'] += f'_plane{pl_i}_poly{p_i}'
            write_vtk_xml(cut_poly, **this_saving_params)
            logger.debug(f"One cut done in {end - start:.3f} seconds")

        outs.append(plane_outs)
    return outs


def merge_polys(polys):
    """Returns a polydata which contains all vtk polys in `polys`"""

    # create an empty "poly data list" (type of list for polydata)
    append_filter = vtk.vtkAppendPolyData()
    # add the polys from the "python list called polys" to the "poly data list"
    for poly_data in polys:
        append_filter.AddInputData(poly_data)
    # apply changes
    append_filter.Update()

    # create a new poly data and merge all polys in "poly data list", result saved in the new poly data
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(append_filter.GetOutput())
    # apply changes and return output poly
    clean_filter.Update()
    return clean_filter.GetOutput()


def visualize_poly(poly_data,
                   representation='wireFrame',
                   scalar_mode='point_data'):
    """Display the poly_data.

    This requires user interaction.

    :param poly_data: polydata to display
    :type poly_data: vtk polydata
    :param representation: Type of representation to be shown in case
    `display` is True. This can be the followings:
        ``wireFrame``
        ``surface``
        ``points``
    :type representation: str
    :param scalar_mode: whther to use CellData or PointData as scalar to
    render color. It can be the following:
        ``point_data``
        ``cell_data``
    :type: str
    """
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    if scalar_mode == 'point_data':
        mapper.SetScalarModeToUsePointData()
        mapper.SetScalarRange(poly_data.GetPointData().GetScalars().GetRange())
    elif scalar_mode == 'cell_data':
        mapper.SetScalarModeToUseCellData()
        mapper.SetScalarRange(poly_data.GetCellData().GetScalars().GetRange())
    mapper.SetColorModeToMapScalars()

    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if representation == 'wireFrame':
        actor.GetProperty().SetRepresentationToWireframe()
    elif representation == 'surface':
        actor.GetProperty().SetRepresentationToSurface()
    elif representation == 'points':
        actor.GetProperty().SetRepresentationToPoints()

    # Visualize
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.AddActor(actor)
    renderer.SetBackground(.5, .3, .31)  # Background color salmon
    render_window.SetSize([1328, 956])

    render_window.Render()
    render_window_interactor.Start()


# --------------------------------------- Segmentation maps generation -----------------------------------

def render_poly(poly_data,
                representation='wireFrame',
                scalar_mode='point_data',
                view=None,
                view_plane=None,
                do_parallel_projection=True,
                zoom_parallel_proj=85,
                zoom_perspective_proj=300,
                size=(400, 400)):
    """Returns rendered image array from a poly_data or unstructured grid object.

    This requires user interaction.

    :param poly_data: polydata to display
    :type poly_data: vtk polydata

    :param representation: Type of representation to be used
    This can be the followings:
        ``wireFrame``
        ``surface``
        ``points``
    :type representation: str

    :param scalar_mode: whether to use CellData or PointData as scalar to
    render color. It can be the following:
        ``point_data``
        ``cell_data``
    :type: str

    :param view: the chamber view
    :type view: str

    :param view_plane: the segmentation plane defined with origin and normal vector.
    If this parameter is set, an image corresponding to the passed slicing plane is generated.
    It should correspond to an echocardiographic view
    :type: dict

    :param do_parallel_projection: if parallel project  should be used for the image generation
    :type: bool

    :param zoom_perspective_proj: the distance of the camera in case of perspective projection
    :type: int

    :param zoom_parallel_proj: the zoom used in case of parallel projection (bigger number corresponds to smaller image)
    type: int

    :param size: the width and height of the rendered image
    :type size: tuple
    """

    # mapper
    if isinstance(poly_data, vtk.vtkPolyData):
        mapper = vtk.vtkPolyDataMapper()
    else:
        mapper = vtk.vtkDataSetMapper()

    mapper.SetInputData(poly_data)

    if scalar_mode == 'point_data':
        mapper.SetScalarModeToUsePointData()
        scalars = poly_data.GetPointData().GetScalars()
        if scalars is not None:
            # TODO check this before committing, if debug works without,remove
            mapper.SetScalarRange(scalars.GetRange())
        else:
            mapper.SetScalarRange(poly_data.GetPointData().GetArray(0).GetRange())
    elif scalar_mode == 'cell_data':
        mapper.SetScalarModeToUseCellData()
        mapper.SetScalarRange(poly_data.GetCellData().GetScalars().GetRange())
    mapper.SetColorModeToMapScalars()

    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if representation == 'wireFrame':
        actor.GetProperty().SetRepresentationToWireframe()
    elif representation == 'surface':
        actor.GetProperty().SetRepresentationToSurface()
    elif representation == 'points':
        actor.GetProperty().SetRepresentationToPoints()

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.OffScreenRenderingOn()

    render_window.AddRenderer(renderer)
    renderer.AddActor(actor)
    renderer.SetBackground(0, 0, 0)  # Background color
    render_window.SetSize(size)

    renderer.ResetCamera()

    # generate plane images (segmentation maps)
    if view_plane:
        camera = vtk.vtkCamera()

        view_up_vec = np.cross(-1 * np.array(view_plane['normal']), [1, 1, 1])
        camera.SetViewUp(view_up_vec)
        camera.SetPosition(
            view_plane['origin'] - zoom_perspective_proj * np.array(
                view_plane['normal']))

        camera.SetFocalPoint(view_plane['origin'])

        if do_parallel_projection:
            camera.ParallelProjectionOn()
            camera.SetParallelScale(zoom_parallel_proj)

        renderer.SetActiveCamera(camera)

    render_window.Render()

    vtk_win_im = vtk.vtkWindowToImageFilter()
    vtk_win_im.SetInput(render_window)
    vtk_win_im.Update()

    vtk_image = vtk_win_im.GetOutput()

    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    arr = vtk_to_numpy(vtk_array).reshape(height, width, components)

    render_window.Finalize()

    del render_window
    del renderer

    flipped = np.flipud(arr)

    if view:
        if view.lower() == '3ch':
            flipped = np.fliplr(flipped)
    return flipped


def fill_contour_image(contour_image, verbose=False):
    grey = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)[1]

    # # not necessary, if contour is connected
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

    filled = np.zeros_like(thresh)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)

    contours = contours[0] if len(contours) == 2 else contours[1]

    if len(contours) == 0:
        assert np.sum(contour_image) == 0, "Contour should have been found!"
        return filled, False

    # fill contour
    the_contour = contours[0]
    filled = cv2.drawContours(filled, [the_contour], 0, 255, -1)

    # sanity chek if contour could be filled
    not_filled = (np.sum(filled) / np.sum(thresh) < 5)
    if verbose and not_filled:
        logger.info("Error, contour could not be filled!")

    return filled, not_filled


def randomize_view(plane, normal_max_change=0.1, origin_max_change=0.5):
    """ randomly sample different camera position"""

    assert 0 <= normal_max_change <= 1
    assert 0 <= origin_max_change <= 1

    max_normal = np.max(abs(np.asarray(plane['normal'])))
    max_origin = np.max(abs(np.asarray(plane['origin'])))

    # absolute change calculated based on max value and percentage of change
    normal_max_abs_change = max_normal * normal_max_change
    origin_max_abs_change = max_origin * origin_max_change

    rand_normal = np.random.uniform(-normal_max_abs_change, normal_max_abs_change, 3)
    rand_position = np.random.uniform(-origin_max_abs_change, origin_max_abs_change, 3)

    plane['normal'] += rand_normal
    plane['origin'] += rand_position


def generate_seg_maps(poly_data, heart_components=[], views=None,
                      save_path=Path('experiments'),
                      save=True, add_background_class=True, color_dict=None, no_fill=False,
                      size=None):
    """generate slices and corresponding segmentation maps

    Works only for the CONRAD model!

    :param poly_data: unscaled polydata for which maps should be generated
    :type poly_data: vtk polydata
    :param heart_components: the heart components
    :type heart_components: list
    :param views: dictionary of dictionary describing the different views
    :type views: dict
    :param save_path: path to file
    :type save_path: Path
    :param save: if the segmentation maps should be saved
    :type save: bool
    :param add_background_class: if a one-hot dimension of the background pixels should be added
    :type add_background_class: bool

    """

    def _normalize_view_vector(v):
        """normalize vector """

        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    # convert one-hot encoded image to rgb using the color dict assigning each component a color
    def onehot_to_rgb(onehot, color_dict):
        single_layer = np.argmax(onehot, axis=-1)
        output = np.zeros(onehot.shape[:2] + (3,))
        for k in color_dict.keys():
            output[single_layer == k] = list(reversed(color_dict[k]))  # save as BGR instead of RGB
        return np.uint8(output)

    p_zoom = constants.CONRAD_SEG_MAPS_PARALLEL_ZOOM

    if views is None:
        views = {'2ch': constants.CONRAD_2CH_PLANE,
                 '3ch': constants.CONRAD_3CH_PLANE,
                 '4ch': constants.CONRAD_4CH_PLANE,
                 '5ch': constants.CONRAD_5CH_PLANE,
                 'psax': constants.CONRAD_PSAX_PLANE,
                 'plax': constants.CONRAD_PLAX_PLANE}

    assert isinstance(views, dict)

    for k, v in views.items():
        views[k]['origin'] = np.squeeze(views[k]['origin'])
        views[k]['normal'] = np.squeeze(views[k]['normal'])

    components = sorted(list(set(vtk_to_numpy(
        poly_data.GetPointData().GetArray('components')).flat)))

    hc_ids = sorted(
        [constants.CONRAD_COMPONENTS_TO_IDS[hc] for hc in heart_components])

    all_hc_ids = sorted(
        [constants.CONRAD_COMPONENTS_TO_IDS[hc] for hc in constants.CONRAD_COMPONENTS])

    if len(hc_ids) == 0:  # if no heart component specified, use all components
        hc_ids = all_hc_ids

    # assert components == hc_ids
    seg_maps = {}
    if size is None:
        size = constants.CONRAD_SEG_MAPS_SIZE

    for view, plane in views.items():

        plane['normal'] = _normalize_view_vector(plane['normal'])
        cut = make_slices(plane_dicts=[plane], polys=[poly_data], save=False)

        images = []
        for c_idx in all_hc_ids:  # for all components
            if c_idx in hc_ids:
                poly = cut[0][0]
                thresholdfilter = vtk.vtkThreshold()
                thresholdfilter.SetInputData(poly)
                thresholdfilter.ThresholdBetween(-0.5 + c_idx, c_idx + 0.5)

                thresholdfilter.Update()
                part = thresholdfilter.GetOutput()

                image = render_poly(part,
                                    view=view,
                                    view_plane=plane,
                                    zoom_parallel_proj=p_zoom,
                                    size=size)

                filled, not_filled = fill_contour_image(image)

                if no_fill:  # if not-filled components requested
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]
                    images.append(image)
                elif not_filled:  # if component countour not filled, just append zeros instead of the unfilled countour
                    zeros = np.zeros(shape=size)
                    images.append(zeros)
                else:
                    images.append(filled)
            else:
                zeros = np.zeros(shape=size)
                images.append(zeros)

        seg_map = np.stack(images, axis=-1)

        # clean overlap
        for idx, comp_idx in enumerate(hc_ids):
            # left ventricle completely overlaps with myocardium, do not remove
            if comp_idx == constants.CONRAD_COMPONENTS_TO_IDS['leftVentricle']:
                continue
            others = [hc_ids.index(c) for c in hc_ids if c != comp_idx]
            for other_idx in others:
                seg_map[(seg_map[..., idx] == 255) &
                        (seg_map[..., other_idx] == 255), idx] = 0

        # make one-hot
        if np.max(seg_map) > 1:
            seg_map = seg_map > (np.max(seg_map) / 2)

        if add_background_class:
            background = np.sum(seg_map, axis=-1) == 0
            background = background[..., np.newaxis]
            seg_map = np.append(seg_map, background, axis=-1)

        if color_dict is None:
            color_dict = constants.CONRAD_SEG_MAP_COLORS
        seg_map_rgb = onehot_to_rgb(seg_map, color_dict=color_dict)

        hc_id_as_pixel = np.argmax(seg_map,
                                   axis=-1)  # array of pixels where each pixel value is the id of the corresponding component
        per_comp_pixel_count = {}
        for comp_idx in all_hc_ids:
            comp = constants.CONRAD_IDS_TO_COMPONENTS[comp_idx]
            occurrences = np.count_nonzero(hc_id_as_pixel == comp_idx)
            per_comp_pixel_count[comp] = occurrences

        seg_maps[view] = {'map': seg_map, 'map_rgb': seg_map_rgb,
                          'pixel_counts': per_comp_pixel_count, 'plane': plane, 'zoom': p_zoom}

    if save:
        with lzma.open(save_path, "wb") as f:
            pickle.dump(seg_maps, f)

    return seg_maps


def get_lv_seg_map_ef(feats, ref_poly, dataset_scale, dataset_min, parallel_process=False,
                      process_num=0, return_dict=None):
    views = {'4ch': CONRAD_VIEWS['4ch']}  # views to generate for
    heart_components = ["leftVentricle"]
    pixel_threshold = 0  # if nb pixel of left ventricle are smaller or equal to this threshold in a certain frame, then skip this frame

    # denormalize feats
    feats = (feats * dataset_scale) + dataset_min

    # compute seg_maps and 2D efs from it
    lv_areas = []  # nb of pixels of the left ventricle for each mesh
    for mesh_feat in feats:
        poly = overwrite_vtkpoly(ref_poly, points=mesh_feat)
        seg_maps = generate_seg_maps(poly, heart_components, views=views, save=False)
        lv_pixel_count = seg_maps['4ch']['pixel_counts']['leftVentricle']
        # if pixel count of left ventricle too low, skip this frame
        # to avoid that min_value and max_value are too low thus avoiding
        # extrem ef values (0 and 1)
        if lv_pixel_count > pixel_threshold:
            lv_areas.append(lv_pixel_count)

    # default ef value
    ef_2d = 0.0
    if len(lv_areas) > 0:
        max_area = max(lv_areas)
        min_area = min(lv_areas)
        ef_2d = ((max_area - min_area) / max_area)

    if parallel_process:
        return_dict[process_num] = ef_2d
    else:
        return ef_2d


# --------------------------------------------------------------------------------------------------------

# ---------------------------------- UKHeart and CISTIB data functions -----------------------------------
# used for CISTIB data
def generate_cistib_shape(mode=0, std_num=3, shape='Full', **saving_params):
    """Generate shape from PCA data of the second dataset

    Note that the `mode` is in range [0, 49] inclusive. This is in contrast
    with the original dataset format where modes are defined in [1,
    50]. This function is implemented to keep consistency with
    `generate_uk_shape` where indexing starts from 0.
    """

    input_dir = saving_params['input_dir']
    mean_path = input_dir / f"PCAmodel_{shape}_phaseAll_90pct_proc05it.vtp"
    var_path = (input_dir / f"PCAmodel"
                            f"_{shape}_phaseAll_90pct_proc05it_Lambdas.txt")

    polydata = load_polydata(mean_path)
    n_points = polydata.GetNumberOfPoints()

    pc_i = polydata.GetPointData().GetArray(f'mode_{mode + 1:02d}')
    pc_i = np.reshape(vtk_to_numpy(pc_i), (n_points, 3))

    # TODO: sqrt or not?
    std = np.sqrt(float(open(var_path, 'r').readlines()[mode]))

    newpolydata = generate_shape_from_mode(polydata, pc_i, std, std_num=std_num)
    saving_params['name'] = f"PCAmodel_{shape}_mode{mode}_{std_num}std"
    out_file = write_vtk_xml(newpolydata, **saving_params)
    return newpolydata, out_file


# used for UKHeart Data
def generate_uk_shape(mode=99, std_num=3, shape='LV', **saving_params):
    # Read the mean shape
    input_dir = saving_params['input_dir']
    mean_path = input_dir / f"{shape}_ED_mean.vtk"
    polydata = load_polydata(mean_path)
    n_points = polydata.GetNumberOfPoints()

    # Read the principal component and variance
    pc_path = input_dir / f"{shape}_ED_pc_100_modes.csv.gz"
    var_path = input_dir / f"{shape}_ED_var_100_modes.csv.gz"
    pc = np.genfromtxt(str(pc_path), delimiter=',')
    variance = np.genfromtxt(str(var_path), delimiter=',')

    pc_i = pc[:, mode]
    pc_i = np.reshape(pc_i, (n_points, 3))
    std = np.sqrt(variance[mode])

    newpolydata = generate_shape_from_mode(polydata, pc_i, std, std_num=std_num)
    saving_params['name'] = f"{shape}_ED_mode{mode + 1}_{std_num}std"
    out_file = write_vtk_xml(polydata, **saving_params)
    return newpolydata, out_file


# used for UKHeart Data and CISTIB Data
def get_subgraph(nxgraph, sub_nodes, relabel=True):
    """Get subgraph of `nxgraph` where all graph attributes are adjusted
    accordingly"""

    map_to_sub = {x: i for i, x in enumerate(sub_nodes)}
    sub_graph = nxgraph.subgraph(sub_nodes)

    polys = nxgraph.graph['Polys']
    sub_poly_idx = []
    sub_g_attrs = dict.fromkeys(nxgraph.graph.keys())
    sub_g_attrs['Polys'] = []
    for p_i, poly in enumerate(polys):
        if np.all(np.isin(poly[1:], sub_nodes)):
            # Append the poly data
            if relabel:
                mapped_poly = [poly[0]] + [map_to_sub[p] for p in poly[1:]]
                sub_g_attrs['Polys'].append(mapped_poly)
            else:
                sub_g_attrs['Polys'].append(poly)
            sub_poly_idx.append(p_i)
    sub_g_attrs['Polys'] = np.array(sub_g_attrs['Polys'])

    # Append other graph attributes
    for key in nxgraph.graph.keys():
        if key == "Polys":
            continue
        attr = nxgraph.graph[key]
        sub_g_attrs[key] = attr[sub_poly_idx]

    if relabel:
        sub_graph = nx.relabel_nodes(sub_graph, map_to_sub)
    for key, value in sub_g_attrs.items():
        sub_graph.graph[key] = value
    return sub_graph


# used for UKHeart Data and CISTIB Data
def poly_to_nxgraph(polydata, **save_params):
    """ Make a networkx graph from vtk polygon
    A polydata has several information such as Points, Polys, PointData and
    CellData. This function keeps all the information related to each vertex
    and loses other information related to each cell. The kept information
    are Points and PointData. The information related to Polys are lost
    except adjacent vertices. In a later version one can maybe assign each
    vertex to a face number to keep the face information as well.

    :param polydata: vtk polydata
    :param save_params: dictionary containing parameters required for
    saving. Very similar to parameters in `write_vtk_xml`:
        ``save``
            bool indicating whether to save or not
        ``name``
            The name of the file in which data will be saved. This is
            appended to the global `output_dir` to make the absolute path.
    :return: networkx graph
    """
    a_mat, x_mat = vtkpoly_to_adj_feats(polydata)

    n_attr = polydata.GetPointData().GetNumberOfArrays()
    v_attr_names = []
    v_attrs = []
    for attr_i in range(n_attr):
        attr = polydata.GetPointData().GetArray(attr_i)
        v_attr_names.append(attr.GetName())
        v_attrs.append(vtk_to_numpy(attr))
    v_attr_names.append('coordinates')
    v_attrs.append(x_mat)

    n_attr = polydata.GetCellData().GetNumberOfArrays()
    n_cell = polydata.GetNumberOfCells()
    g_attr_names = []
    g_attrs = []
    for attr_i in range(n_attr):
        attr = polydata.GetCellData().GetArray(attr_i)
        dim = int(attr.GetSize() / n_cell)
        g_attr_names.append(attr.GetName())
        g_attrs.append(vtk_to_numpy(attr).reshape(n_cell, dim))

    polys = vtk_to_numpy(polydata.GetPolys().GetData())
    polys = polys.reshape(polydata.GetNumberOfPolys(), polys[0] + 1)
    g_attr_names.append('Polys')
    g_attrs.append(polys)

    graph = adj_to_nxgraph(a_mat,
                           v_attrs=v_attrs, v_attr_names=v_attr_names,
                           g_attrs=g_attrs, g_attr_names=g_attr_names)

    if 'save' in save_params and save_params['save']:
        output_dir = save_params['output_dir']
        out_file = output_dir / f"graph_{save_params['name']}.gpickle"
        nx.write_gpickle(graph, out_file)
    return graph


# TODO: attributes better be dict than two different lists!
# used for UKHeart Data and CISTIB Data
def adj_to_nxgraph(adj_mat, v_attrs=None, v_attr_names=None,
                   g_attrs=None, g_attr_names=None,
                   e_attrs=None, e_attr_names=None,
                   **save_params):
    """ Make an networkx graph from adjacency matrix

    In addition to making a graph from the adjacency matrix, `attrs` are set as
    the node attributes of the graph. `attr_name` must be the corresponding
    names of the attributes in `attrs`. These are optional.
    The graph is saved as pickle if `saved_params` has the right parameters
    as similar to `write_vtk_xml`.

    :param adj_mat: 0-1 Matrix (n*n) of adjacency connections.
    :type adj_mat: numpy.ndarray or any other object accepted by `nx.DiGraph`
    :param v_attrs: (optional) list of m attributes. Each attribute (element of
    this list) should be another list of n attributes each for the n vertices
    of the graph.
    type: v_attrs: (numpy.ndarray, list) of (numpy.ndarray, list)
    :param v_attr_names: (optional) list of m strings where each string
    corresponds to one element of `v_attrs`. This parameter is ignored if
    `v_attrs` is omitted.
    type: v_attr_names: numpy.ndarray or list
    :param g_attrs: (optional) list of k graph attributes.
    type: g_attrs: (numpy.ndarray, list) of anything
    :param g_attr_names: (optional) list of k strings where each string
    corresponds to one element of `g_attrs`. This parameter is ignored if
    `g_attrs` is omitted.
    type: g_attr_names: numpy.ndarray or list
    :param e_attrs: (optional) list of j attributes. Each attribute (element of
    this list) should be another list of e attributes each for the e edges
    of the graph.
    type: e_attrs: (numpy.ndarray, list) of (numpy.ndarray, list)
    :param e_attr_names: (optional) list of j strings where each string
    corresponds to one element of `e_attrs`. This parameter is ignored if
    `e_attrs` is omitted.
    type: e_attr_names: numpy.ndarray or list
    :param save_params: dictionary containing parameters required for
    saving. Very similar to parameters in `write_vtk_xml`:
        ``save``
            bool indicating whether to save or not
        ``name``
            The name of the file in which data will be saved. This is
            appended to the global `output_dir` to make the absolute path.
    :return: networkx graph built of the adjacency matrix
    """
    nxgraph = nx.Graph(adj_mat)

    if v_attrs:
        assert isinstance(v_attrs, list)
        assert isinstance(v_attr_names, list)
        assert len(v_attrs) == len(v_attr_names)
        for i, attr in enumerate(v_attrs):
            assert len(attr) == nxgraph.number_of_nodes()
            attr_dict = {i: attr[i] for i in range(0, len(attr))}
            nx.set_node_attributes(nxgraph, attr_dict, v_attr_names[i])

    if e_attrs:
        assert isinstance(e_attrs, list)
        assert isinstance(e_attr_names, list)
        assert len(e_attrs) == len(e_attr_names)
        for i, attr in enumerate(e_attrs):
            assert len(attr) == nxgraph.number_of_nodes()
            attr_dict = {i: attr[i] for i in range(0, len(attr))}
            nx.set_edge_attributes(nxgraph, attr_dict, e_attr_names[i])

    if g_attrs:
        assert isinstance(g_attrs, list)
        assert isinstance(g_attr_names, list)
        assert len(g_attrs) == len(g_attr_names)
        for i, attr in enumerate(g_attrs):
            nxgraph.graph[g_attr_names[i]] = attr

    if 'save' in save_params and save_params['save']:
        output_dir = save_params['output_dir']
        out_file = output_dir / f"graph_{save_params['name']}.gpickle"
        nx.write_gpickle(nxgraph, out_file)

    return nxgraph


# TODO: Use the overwrite_poly to avoid duplicate code
# used for UKHeart Data and CISTIB Data
def nxgraph_to_vtkpoly(nxgraph,
                       xyz_name='coordinates', poly_name='Polys',
                       poly_scalar='part', point_scalar='epiendo',
                       normal_name=None,
                       **saving_params):
    """ Make a vtk polygon from networkx graph
    To be able to make a vtk polygon from a netowrkx graph, coordinates of
    each vertex must be provided. It is assumed that coordinates are given
    as vertex attributes of the graph where the attribute name is `xyz_name`.

    Polys will also be set if provided as graph attributes. Otherwise only
    Lines based on edge information will be constructed.

    All other node and graph attributes will also be set as PointData and
    CellData arrays.

    In addition, to recover the computed point normals, they can also be
    given as vertex attributes with name 'normal_name'.

    :param nxgraph: The networkx graph that contains coordinate of each
    vertex as their attribute
    :type nxgraph: nx.Graph
    :param xyz_name: (optional) a string indicating the name of the vertex
    attribute that contains vertex coordinates
    :type xyz_name: str
    :param poly_name: (optional) a string indicating the name of the graph
    attribute where poly data are stored
    :type poly_name: str
    :param poly_scalar: (optional) a string indicating the name of the vertex
    attribute that contains PointData scalars
    :type poly_scalar: str
    :param point_scalar: (optional) a string indicating the name of the vertex
    attribute that contains PolyData scalars
    :type point_scalar: str
    :param normal_name: a string indicating the name of the attribute of the
    vertex that contains its normal vector
    :type normal_name: str
    :param saving_params: parameters required by `write_vtk_xml`
    :return: vtk polydata created by the given graph
    """
    # Make polydata
    polydata = vtk.vtkPolyData()

    # Points
    node_coords = list(nx.get_node_attributes(nxgraph, xyz_name).values())
    points = vtk.vtkPoints()
    for node in node_coords:
        points.InsertNextPoint(node)
    polydata.SetPoints(points)

    # Add Polys if available
    if poly_name in nxgraph.graph.keys():
        poly_list = nxgraph.graph[poly_name]
        p_dim = poly_list.shape[1] - 1
        polys = vtk.vtkCellArray()
        polys.Allocate(len(poly_list), p_dim)
        for poly in poly_list:
            polys.InsertNextCell(p_dim)
            for p in poly[1:]:
                polys.InsertCellPoint(p)
        polydata.SetPolys(polys)

    # Add lines if no Poly info available
    if poly_name not in nxgraph.graph.keys():
        edges = np.array(nxgraph.edges)
        line = vtk.vtkCellArray()
        line.Allocate(len(edges), 2)
        for edge in edges:
            line.InsertNextCell(2)
            line.InsertCellPoint(edge[0])
            line.InsertCellPoint(edge[1])
        polydata.SetLines(line)

    # PointData
    for key in nxgraph.nodes[0].keys():
        if key == xyz_name:  # Already added as point coordinates
            continue

        attr = numpy_to_vtk(
            list(nx.get_node_attributes(nxgraph, key).values()))
        attr.SetName(key)
        polydata.GetPointData().AddArray(attr)
    polydata.GetPointData().SetActiveScalars(point_scalar)

    # CellData
    for key in nxgraph.graph.keys():
        if key == poly_name:  # Already added as polys
            continue

        attr = numpy_to_vtk(nxgraph.graph[key].flatten().astype(int))
        attr.SetName(key)
        polydata.GetCellData().AddArray(attr)
    polydata.GetCellData().SetActiveScalars(poly_scalar)

    if normal_name:
        normals = list(nx.get_node_attributes(nxgraph, normal_name).values())
        polydata.GetPointData().SetNormals(numpy_to_vtk(normals))

    write_vtk_xml(polydata, **saving_params)
    return polydata

# --------------------------------------------------------------------------------------------------------
