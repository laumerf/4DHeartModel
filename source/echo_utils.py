import random
import tensorflow as tf
import cv2 as cv
from IPython.display import Video, display
import numpy as np
import os
import pandas as pd
import skvideo.io
import uuid


def load_data(include_times=False, histogram_equalisation=False):
    data_info = pd.read_csv('./data/data_info.csv', index_col='globalID')
    data_info['diagnosis'] = data_info['id'].apply(lambda s: s[-1]).astype(
        'string')

    # load videos
    frames_by_view = {}
    times_by_view = {}

    for view in ['2CH', '4CH']:

        frames_by_view[view] = {}
        times_by_view[view] = {}
        for index, row in data_info.iterrows():
            number = row['id'][:-1]
            diagnosis = 'AMI' if row['diagnosis'] == 'A' else 'TTS'
            folder = './data/AMI_TTS/' + diagnosis + '/' + str(number)
            if histogram_equalisation:
                cache_folder = './cache/AMI_TTS_histeq/' + diagnosis + '/' + str(
                    number)
            else:
                cache_folder = './cache/AMI_TTS/' + diagnosis + '/' + str(
                    number)

            # load data from cache or directly from video files

            if os.path.exists(cache_folder + '/' + view):
                # load from cache
                times = np.loadtxt(cache_folder + '/' + view + '/times',
                                   delimiter='\n')

                frames = []
                for i, _ in enumerate(times):
                    img = cv.imread(
                        cache_folder + '/' + view + '/' + str(i) + '.bmp')
                    frames.append(img[:, :, 0])

            else:
                # load from dataset and cache images
                frames = skvideo.io.vread(folder + '/' + view + '.avi')
                frames = [frame[:, :, 0] for frame in frames]

                # histogram equalisation
                if histogram_equalisation:
                    frames = np.reshape(frames, [-1, 112])
                    frames = cv.equalizeHist(frames)
                    frames = np.reshape(frames, [-1, 112, 112])

                # times
                meta_data = skvideo.io.ffprobe(folder + '/' + view + '.avi')
                time_base = eval(meta_data['video']['@time_base'])
                duration_ts = int(meta_data['video']['@duration_ts'])
                times = [i * time_base for i in range(duration_ts)]

                # to cache
                os.makedirs(cache_folder + '/' + view + '/')
                for i, img in enumerate(frames):
                    cv.imwrite(
                        cache_folder + '/' + view + '/' + str(i) + '.bmp', img)
                np.savetxt(cache_folder + '/' + view + '/times', times,
                           delimiter='\n')

            frames_by_view[view][index] = frames
            times_by_view[view][index] = times

    if include_times == True:
        return data_info, frames_by_view['2CH'], frames_by_view['4CH'], \
               times_by_view['2CH'], times_by_view['4CH']

    return data_info, frames_by_view['2CH'], frames_by_view['4CH']


def load_data_ami_tts_112(histogram_equalisation=False, view='4CH'):
    """
    AMI/TTS data, A4C, 112x112
    Returns a dictionary with a mapping from subject ids to .npz-file paths which contain a frames and a times array.
    """

    if histogram_equalisation:
        video_cache_folder = './cache/AMI_TTS_112_histeq/' + view
    else:
        video_cache_folder = './cache/AMI_TTS_112/' + view
    for diagnosis in ['AMI', 'TTS']:
        if not os.path.exists(video_cache_folder + '/' + diagnosis):
            os.makedirs(video_cache_folder + '/' + diagnosis)

    data_info = pd.read_csv('./data/data_info.csv', index_col='globalID')
    data_info['diagnosis'] = data_info['id'].apply(lambda s: s[-1]).astype(
        'string')

    files = dict()
    for index, row in data_info.iterrows():
        number = row['id'][:-1]
        diagnosis = 'AMI' if row['diagnosis'] == 'A' else 'TTS'
        filepath = './data/AMI_TTS_112/' + view + '/' + diagnosis + '/' + str(
            number) + '.npy'
        filepath_original = './data/AMI_TTS/' + diagnosis + '/' + str(
            number) + '/' + view + '.avi'
        filepath_cached = video_cache_folder + '/' + diagnosis + '/' + str(
            number) + '.npz'

        # cache frames and times, if not existing
        if not os.path.exists(filepath_cached):

            # load from dataset
            frames = np.load(filepath)

            # histogram equalisation
            if histogram_equalisation:
                frames = np.reshape(frames, [-1, 112])
                frames = cv.equalizeHist(frames)
                frames = np.reshape(frames, [-1, 112, 112])

            # times, loading time base from original clip
            meta_data = skvideo.io.ffprobe(filepath_original)
            time_base = eval(meta_data['video']['@time_base'])
            times = [i * time_base for i in range(frames.shape[0])]

            # cache data
            np.savez(filepath_cached, frames=frames, times=times)

        files[index] = filepath_cached

    return data_info, files


def load_data_ami_tts_v2(cohort, histogram_equalisation=False,
                         standardisation=False, gaussian_smoothing=False,
                         view='4CH', shape=(112, 112)):
    """
    AMI/TTS data (v2)
    Returns a dictionary with a mapping from subject ids to .npz-file paths which contain a frames and a times array.
    """

    video_cache_folder = './cache/AMI_TTS_' + str(shape[0]) + '_' + str(
        shape[1]) + '_v2'
    if histogram_equalisation:
        video_cache_folder += '_histeq'
    if standardisation:
        video_cache_folder += '_std'
    if gaussian_smoothing:
        video_cache_folder += '_smooth'
    video_cache_folder += '/cohort' + str(cohort) + '/' + view

    for diagnosis in ['AMI', 'TTS']:
        path = video_cache_folder + '/' + diagnosis
        if not os.path.exists(path):
            os.makedirs(path)

    video_info = pd.read_csv('./data/AMI_TTS_v2/video_info.csv',
                             index_col=['cohort', 'name', 'disease',
                                        'chamber'])
    patient_data = pd.read_csv('./data/AMI_TTS_v2/patient_data.csv',
                               index_col='globalId')
    patient_data['diagnosis'] = patient_data['TTS'].map(
        lambda b: 'T' if b == 1 else 'A')

    files = dict()
    for index, row in video_info.iterrows():

        cohort_i, name_i, disease_i, chamber_i = index
        original_name = video_info.loc[index, 'original_name']

        if cohort_i != cohort or chamber_i != view:
            continue

        global_id = str(cohort) + '_' + str(name_i) + '_' + disease_i
        filepath_original = './data/AMI_TTS_v2/cohort' + str(
            cohort) + '/' + view + '/' + disease_i + '/' + str(
            original_name) + '.npy.npz'
        filepath_cached = video_cache_folder + '/' + disease_i + '/' + str(
            name_i) + '.npz'

        # cache frames and times, if not existing
        if not os.path.exists(filepath_cached):

            # load from dataset
            frames = np.load(filepath_original)['arr_0']

            # gaussian smoothing
            if gaussian_smoothing:
                frames = [cv.GaussianBlur(x, (35, 35), 7) for x in frames]

            # resize
            frames = np.array([cv.resize(frame, shape) for frame in frames])

            # histogram equalisation
            if histogram_equalisation:
                frames = np.reshape(frames, [-1, shape[1]])
                frames = cv.equalizeHist(frames)
                frames = np.reshape(frames, [-1, shape[0], shape[1]])

            # standardise
            if standardisation:
                frames = frames.astype('float')
                frames -= np.mean(frames)
                frames /= np.std(frames)

            # times, loading time base from original clip
            framerate = video_info.loc[index, 'fps']
            frame_duration = 1 / framerate
            times = [i * frame_duration for i in range(frames.shape[0])]

            # cache data
            np.savez(filepath_cached, frames=frames, times=times)

        files[global_id] = filepath_cached

    return patient_data, files


def load_data_ami_tts_v3(cohort, view='4CH', filepaths_only=True):
    """
    AMI/TTS data (v3)
    Returns a dictionary with a mapping from subject ids to .pbz2-filepaths.
    """

    video_cache_folder = './cache/AMI_TTS_v3/cohort' + str(cohort) + '/' + view

    for diagnosis in ['AMI', 'TTS']:
        path = video_cache_folder + '/' + diagnosis
        if not os.path.exists(path):
            os.makedirs(path)

    video_info = pd.read_csv('./data/AMI_TTS_v2/video_info.csv',
                             index_col=['cohort', 'name', 'disease',
                                        'chamber'])
    patient_data = pd.read_csv('./data/AMI_TTS_v2/patient_data.csv',
                               index_col='globalId')
    patient_data['diagnosis'] = patient_data['TTS'].map(
        lambda b: 'T' if b == 1 else 'A')

    files = dict()
    for index, row in video_info.iterrows():

        cohort_i, name_i, disease_i, chamber_i = index
        original_name = video_info.loc[index, 'original_name']

        if cohort_i != cohort or chamber_i != view:
            continue

        global_id = str(cohort) + '_' + str(name_i) + '_' + disease_i
        filepath_original = './data/AMI_TTS_v3/cohort' + str(
            cohort) + '/manual_crop/' + disease_i + '/' + str(
            original_name) + '/' + chamber_i + '.avi'
        filepath_cached = video_cache_folder + '/' + disease_i + '/' + str(
            name_i) + '.npz'

        # cache frames and frame_duration, if not existing
        if not os.path.exists(filepath_cached):

            # load frames from video file
            frames = []
            cap = cv.VideoCapture(filepath_original)
            while cap.grab():
                frames.append(cap.retrieve()[1][:, :, 0])
            cap.release()
            frames = np.array(frames)

            # load frame duration
            framerate = video_info.loc[index, 'fps']
            frame_duration = 1.0 / framerate

            # cache data
            np.savez(filepath_cached, frames=frames,
                     frame_duration=frame_duration)

        if filepaths_only:
            files[global_id] = filepath_cached
        else:
            data = np.load(filepath_cached)
            frame_duration = data['frame_duration']
            frames = data['frames']
            files[global_id] = {
                'frames': frames,
                'frame_duration': frame_duration
            }

    return patient_data, files


def load_data_echonet(path="./data/EchoNet-Dynamic/",
                      number_of_videos=None,
                      histogram_equalisation=False,
                      videos=None):
    """
    EchoNet-Dynamic data.
    Returns a dictionary with a mapping from subject ids to .npz-file paths which contain a frames and a times array.
    """
    print("Preparing/Loading ECHO videos")

    if histogram_equalisation:
        video_cache_folder = './cache/EchoNet-Dynamic/Videos_histeq'
    else:
        video_cache_folder = './cache/EchoNet-Dynamic/Videos'
    if not os.path.exists(video_cache_folder):
        os.makedirs(video_cache_folder)

    data_info = pd.read_csv(os.path.join(path, 'FileList.csv'))

    data_info['globalID'] = data_info['FileName'].apply(
        lambda s: s[:-4]).astype('str')

    data_info.set_index('globalID', inplace=True)

    files = dict()
    counter = 0

    if videos is not None:
        data_info = data_info.loc[data_info['FileName'].isin(videos)]
        number_of_videos = np.inf

    for index, row in data_info.iterrows():

        if counter == number_of_videos:
            break

        filepath = os.path.join(path, 'Videos', index + '.avi')
        filepath_cached = video_cache_folder + '/' + index + '.npz'

        # cache frames and times, if not existing
        if not os.path.exists(filepath_cached):

            # load from dataset
            frames = skvideo.io.vread(filepath)
            frames = [frame[:, :, 0] for frame in frames]

            # histogram equalisation
            if histogram_equalisation:
                frames = np.reshape(frames, [-1, 112])
                frames = cv.equalizeHist(frames)
                frames = np.reshape(frames, [-1, 112, 112])

            # times
            time_base = 1 / data_info.loc[index]['FPS']
            times = [i * time_base for i in range(len(frames))]

            # cache data
            np.savez(filepath_cached, frames=frames, times=times)

        files[index] = filepath_cached

        counter += 1

    return data_info, files


def show_clip(frames, rate=10, width=300, height=300):
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')

    id = uuid.uuid4().hex
    filepath = './tmp/' + id + '.mp4'
    skvideo.io.vwrite(filepath, frames, inputdict={'-r': str(rate)},
                      outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p',
                                  '-r': str(rate)})
    display(Video(data=filepath, width=width, height=height))


def show_clip_comparison(frames1, frames2, rate=10, width=600, height=300):
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')

    id = uuid.uuid4().hex
    filepath = './tmp/' + id + '.mp4'

    frames = np.concatenate([frames1, frames2], axis=2)
    skvideo.io.vwrite(filepath, frames, inputdict={'-r': str(rate)},
                      outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p',
                                  '-r': str(rate)})
    display(Video(data=filepath, width=width, height=height))


def save_video(frames, file_path, fps=30, display_video=False):

    width = frames.shape[1]
    height = frames.shape[2]

    # normalize to 0 255
    try:
        frames = cv.normalize(frames, frames,  0, 255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
        skvideo.io.vwrite(file_path, frames, inputdict={'-r': str(fps)},
                          outputdict={'-vcodec': 'libx264', '-pix_fmt': 'yuv420p',
                                      '-r': str(fps)}, backend='ffmpeg', verbosity=1)

        if display_video:
            display(Video(data=file_path, width=width, height=height))

    except BrokenPipeError as bpe:
        print(f"Video could not be saved. Broken pipe error: {bpe}")


def clip_from_file(filepath):
    data = np.load(filepath)
    times = data['times']
    frames = data['frames']
    frames = (frames / 255).astype('float32')
    return times, frames


def get_val_dataset(files, val_batch_size=128):
    n_val_subjects = len(files)
    print(f"Number of test videos: {n_val_subjects}")

    # as large as possible
    def _val_dataset_generator():
        for file in files:
            data = np.load(file)
            frames = data['frames']

            frames = np.array(
                [cv.resize(x, (112, 112)) for x in frames]).astype(
                'float32')
            # standardize data
            frames -= np.mean(frames)
            frames /= np.std(frames)
            for f in frames:
                yield f[..., np.newaxis]

    dataset_output_types = tf.float32
    dataset_output_shapes = tf.TensorShape([112, 112, 1])
    val_dataset = tf.data.Dataset.from_generator(_val_dataset_generator,
                                                 dataset_output_types,
                                                 dataset_output_shapes)

    val_dataset = val_dataset.batch(val_batch_size)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return val_dataset


def get_train_dataset(files, batch_size, n_shuffle=5000):
    n_train_subjects = len(files)
    print(f"Number of echo train videos: {n_train_subjects}")

    # Dataset generator
    def _train_dataset_generator():
        while True:

            videos = list(range(n_train_subjects))
            random.shuffle(videos)

            for ix in videos:
                filepath = files[ix]
                data = np.load(filepath)

                frames = data['frames']

                frames = np.array(
                    [cv.resize(x, (112, 112)) for x in frames]).astype(
                    'float32')

                # normalize per video (maybe to load file function)
                frames -= np.mean(frames)
                frames /= np.std(frames)

                # requires large enough shuffle buffer!
                for f in frames:
                    yield f[..., np.newaxis]

    dataset_output_types = tf.float32
    dataset_output_shapes = tf.TensorShape([112, 112, 1])

    train_dataset = tf.data.Dataset.from_generator(_train_dataset_generator,
                                                   dataset_output_types,
                                                   dataset_output_shapes)

    train_dataset = train_dataset.shuffle(n_shuffle)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset


def get_video_as_dataset(file):

    n_frames = len(np.load(file)['frames'])

    def _val_dataset_generator():
        data = np.load(file)
        frames = data['frames']

        frames = np.array(
            [cv.resize(x, (112, 112)) for x in frames]).astype(
            'float32')
        # standardize data
        frames -= np.mean(frames)
        frames /= np.std(frames)
        for f in frames:
            yield f[..., np.newaxis]

    dataset_output_types = tf.float32
    dataset_output_shapes = tf.TensorShape([112, 112, 1])
    video_dataset = tf.data.Dataset.from_generator(_val_dataset_generator,
                                                   dataset_output_types,
                                                   dataset_output_shapes)
    video_dataset = video_dataset.batch(n_frames)

    return video_dataset
