import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Disable tensorflow info and warning messages

import argparse
import yaml
import logging
from tensorflow import keras
import matplotlib.pyplot as plt
from xvfbwrapper import Xvfb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle

from source.utils_dhb import *
from source.models.echo_ae import EchoAutoencoderModel
from source.models.video_mesh_ae import DHBMeshAutoencoderModel
from source.models.mesh_ef_predictor import Mesh_EF_Predictor
from source.models.echo_ef_predictor import Echo_EF_Predictor
from source.models.ef_converter import EF_converter
from source.models.cycle_gan import CycleGan
from source.models.echo_to_mesh import Echo_to_Mesh_Model
from source.constants import ROOT_LOGGER_STR, LOGGER_RESULT_FILE

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

TIMESTR = time.strftime("%Y%m%d-%H%M%S")

LEOMED_RUN = "--leomed" in sys.argv
RUN_EXPS = "--run_exps" in sys.argv


def run_dhb_echo(args):
    # for reproducible results
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    experiment_path = args.results_dir / "EAE" / TIMESTR
    experiment_path.mkdir(parents=True, exist_ok=True)

    logfile = experiment_path / LOGGER_RESULT_FILE
    utils.setup_logger(logfile, create_stdlog=True)
    logger.info("Running DHB ECHO Autoencoder")
    logger.info(f"Saving results in {experiment_path}")

    data_dir = args.echo_dir

    # Load EchoNet-Dynamic data
    data_info, all_files = load_data_echonet(logger, data_dir)  # get list of cached files and data array (1 row of data per file)

    # Only use train and val splits
    echonet_train_ids = data_info[data_info.Split == 'TRAIN'].index.values  # get train file names
    echonet_val_ids = data_info[data_info.Split == 'VAL'].index.values  # get validation file names
    ids = list(echonet_train_ids) + list(echonet_val_ids)  # merge both as we will use a different train/validation split using KFold

    # ---------------------------------------------------------------------------------------------------------------------

    # 5-fold CV training -> 5 models
    files = np.array([all_files[id] for id in ids])
    dataset_size = len(files)
    logger.info(f"Dataset (train + val) size: {dataset_size}")
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=41)
    logger.info("Starting Training...")
    logger.info("Training {} times".format(folds))

    for i, (train_index, val_index) in enumerate(kf.split(files)):  # for each of the k-fold splits

        # fit model to data
        t1 = time.time()

        logger.info("\nTrain {}:\n".format(i))

        train_files = files[train_index]  # get train files
        val_files = files[val_index]  # get validation files

        trained_model_path = experiment_path / 'trained_models' / f'echonet_dynamic_{i}'  # path to save the i'th trained model

        model = EchocardioModel(logger=logger, latent_space_dim=128, batch_size=32, hidden_dim=128, log_dir=trained_model_path)  # init model
        model.fit(train_files, val_files, trained_model_path)  # fit data to model, use provided train files for train and validation files for val

        model.save_weights(trained_model_path)  # save best model weights in model path

        # evaluate model
        logger.info("Load sample echos with varied ejection fractions...")
        all_filenames = list(data_info.index)
        efs = list(data_info["EF"])
        n_samples = 10  # n_samples for experiments
        # pick dataset to contain varied EFs (as varied as possible) from the echos dataset
        dataset_indices = utils.spaced_efs_indices(efs, n_samples)
        echo_dataset_filenames = [all_filenames[i] for i in dataset_indices]

        # Print infos on dataset
        n_samples = len(echo_dataset_filenames)
        logger.info(f"ECHO: {n_samples} samples for echo experiments")

        # get the paths from validation set
        dataset_filepaths = [all_files[f] for f in echo_dataset_filenames]

        # get dataset (not encoded)
        echo_dataset = get_dataset(dataset_filepaths, batch_size=8, subsample=False)

        model.reconstruct_echo_vid(echo_dataset, echo_dataset_filenames, "rec_dataset", 0, save_original=True)
        logger.info("Finished echo reconstructions...")

        keras.backend.clear_session()
        del model
        time.sleep(10)

        # time.sleep(10)
        t2 = time.time()
        h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
        logger.info("Finished EAE {} in {}:{}:{}".format(i, h, m, s))

        # remove if multiple models should be trained
        break


def run_dhb_shape_ae(args, shape_configs):
    # ------------------------------------------------- Data Generator Init -----------------------------------------------
    # for reproducible results
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    continue_training = (args.continue_exp is not None)

    if continue_training:
        experiment_path = args.results_dir / "MAE" / args.continue_exp
        if not experiment_path.exists():
            logger.info(f"The following experiment folder was not found: {experiment_path}")
            exit(-1)
        with (experiment_path / "configs.yml").open(mode='r') as yamlfile:
            shape_configs = yaml.safe_load(yamlfile)
        if not (experiment_path / "trained_models").exists():
            logger.info(f"No saved model in the experiment path {experiment_path}")
            exit(-1)
    else:
        experiment_path = args.results_dir / "MAE" / TIMESTR
        experiment_path.mkdir(parents=True, exist_ok=True)

    logfile = experiment_path / LOGGER_RESULT_FILE
    utils.setup_logger(logfile, create_stdlog=True)
    logger.info("Running DHB Mesh Autoencoder")
    logger.info(f"Saving results in {experiment_path}")
    last_train_params = None
    if continue_training:
        with (experiment_path / "trained_models" / "last_params.yml").open(mode='r') as yamlfile:
            last_train_params = yaml.safe_load(yamlfile)
        logger.info(f"Loading model and continuing training at epoch {last_train_params['epoch']}...\n")

    # get class (cls) for data generation
    data_cls = utils.get_data_handler_cls(shape_configs['data']['data_name'])
    data_handler = data_cls(data_dir=args.data_dir, results_dir=args.results_dir, **shape_configs['data'])
    # ---------------------------------------------------------------------------------------------------------------------

    # Save a copy of configs in the log dir
    with (experiment_path / 'configs.yml').open(mode='w') as yaml_file:
        to_save = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in shape_configs.items()}
        yaml.dump(to_save, yaml_file)

    n_train_samples = shape_configs['training']['n_train_samples']
    n_val_samples = shape_configs['training']['n_val_samples']
    n_test_samples = shape_configs['training']['n_test_samples']
    logger.info(f"{n_train_samples} train samples, {n_val_samples} val samples and {n_test_samples} test samples")

    # create an instance of the model, pass relevant args: data_handler, log_dir, training_params and model params (from yml file)
    model = DHBMeshAutoencoderModel(data_handler=data_handler,
                                    log_dir=experiment_path,
                                    training_params=shape_configs['training'],
                                    model_params=shape_configs['model'])

    # get train dataset of size "n_train_samples" as a tf dataset
    train_dataset = data_handler.get_dataset_from_disk("train", n_train_samples)
    train_plotting_dataset = data_handler.get_dataset_from_disk("train", min(n_train_samples, 1000))  # at most 1000 samples of the train dataset to plot on
    # train_dataset_unrepeated = data_handler.get_train_dataset(n_train_samples, batch_size=8, repeat=False)
    # get validation dataset of size "n_val_samples" as a tf dataset
    val_dataset = data_handler.get_dataset_from_disk("val", n_val_samples)
    # get test dataset
    test_dataset = data_handler.get_dataset_from_disk("test", n_test_samples)
    test_samples_to_visualize = 8
    test_dataset_viz = data_handler.get_dataset_from_disk_spaced_efs("test", test_samples_to_visualize)

    # fit model to data
    t1 = time.time()
    model.fit(train_dataset, train_plotting_dataset, val_dataset, test_dataset, test_dataset_viz, last_train_params=last_train_params)
    # model.latent_space_visualization()

    # delete model
    keras.backend.clear_session()
    del model

    # time.sleep(10)
    t2 = time.time()
    h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
    logger.info("DHB MAE done in {}:{}:{}".format(h, m, s))


def run_shape_ae(args, shape_configs):
    t1 = time.time()
    np.random.seed(args.seed)

    # Set paths
    experiment_path = args.results_dir / shape_configs['data'][
        'data_name'] / TIMESTR
    experiment_path.mkdir(parents=True)

    logfile = experiment_path / LOGGER_RESULT_FILE
    utils.setup_logger(logfile, create_stdlog=True)
    logger.info("Running Mesh Autoencoder")
    logger.info(f"Saving results in {experiment_path}")

    # commit_id = Repo(Path().absolute()).head.commit
    # logger.debug(f"Running code on git commit {commit_id}")

    logger.info("Start data pipeline...")

    # get cls i,e class called "CONRADData" (aka shape_configs['data']['data_name']) to handle CONRAD Data
    # the class is found in source package in data.py
    data_cls = utils.get_data_handler_cls(shape_configs['data']['data_name'])
    # create an instance of CONRADData using the () operator and passing args
    # **dictionary matched each arg to the key-value pair of the same key name
    data_handler = data_cls(data_dir=args.data_dir,
                            results_dir=args.results_dir,
                            **shape_configs['data'])
    t2 = time.time()
    h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
    logger.info("Data pipeline done in {}:{}:{}".format(h, m, s))

    # Save a copy of configs in the log dir
    with (experiment_path / 'configs.yml').open(mode='w') as yaml_file:
        to_save = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in shape_configs.items()}
        yaml.dump(to_save, yaml_file)

    # create an instance of the model, pass relevant args: data_handler, log_dir, training_params (from yml file) and model params
    model = MeshAutoencoderModel(data_handler=data_handler,
                                 log_dir=experiment_path,
                                 training_params=shape_configs['training'],
                                 model_params=shape_configs['model'])

    # fit model to data
    t1 = time.time()
    model.fit()
    model.latent_space_visualization()

    # delete model
    keras.backend.clear_session()
    del model

    time.sleep(10)
    t2 = time.time()
    h, m, s = utils.get_HH_MM_SS_from_sec(t2 - t1)
    logger.info("MAE done in {}:{}:{}".format(h, m, s))


def run_echo_ae(args, echo_configs):

    logger.info("Running Echo Autoencoder")

    # Set paths
    experiment_path = args.results_dir / echo_configs['data']['data_name'] / TIMESTR
    experiment_path.mkdir(parents=True)

    # Save a copy of configs in the log dir
    with (experiment_path / 'configs.yml').open(mode='w') as yaml_file:
        to_save = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in echo_configs.items()}
        yaml.dump(to_save, yaml_file)

    model = EchoAutoencoderModel(model_params=echo_configs['model'],
                                 data_params=echo_configs['data'],
                                 training_params=echo_configs['training'],
                                 log_dir=experiment_path)

    model.fit()
    keras.backend.clear_session()
    del model
    time.sleep(10)
    logger.info("EAE DONE!")


def run_echo_to_mesh(args, echo_to_mesh_configs):
    # for reproducible results
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    experiment_path = args.results_dir / "ECHO_TO_MESH" / TIMESTR
    experiment_path.mkdir(parents=True)

    # Save a copy of configs in the log dir
    with (experiment_path / 'configs.yml').open(mode='w') as yaml_file:
        to_save = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in echo_to_mesh_configs.items()}
        yaml.dump(to_save, yaml_file)

    logfile = experiment_path / LOGGER_RESULT_FILE
    utils.setup_logger(logfile, create_stdlog=True)
    logger.info("Running Echo to Mesh Generator")
    logger.info(f"Saving results in {experiment_path}")

    # ---------------------------------------------------------------- ECHOS --------------------------------------------------------
    # Load DHB EAE
    echo_model = echo_to_mesh_configs['loaded_models']['echo']
    echo_model_exp = echo_model['experiment']
    echo_model_nb = echo_model["model_nb"]
    echo_model_path = args.results_dir / "EAE" / echo_model_exp / "trained_models" / f"echonet_dynamic_{echo_model_nb}"
    echo_ae = load_echonet_dynamic_model(echo_model_path, logger)
    logger.info(f"EAE loaded from {echo_model_path}")
    # --------------------------------------------------------------- MESHES --------------------------------------------------------
    # Load DHB MAE config
    mesh_model_exp = echo_to_mesh_configs['loaded_models']['mesh']['experiment']
    mesh_config_path = args.results_dir / "MAE" / mesh_model_exp / "configs.yml"
    with (mesh_config_path).open(mode='r') as yamlfile:
        mesh_configs = yaml.safe_load(yamlfile)
    # get class (cls) for data generation
    data_cls = utils.get_data_handler_cls(mesh_configs['data']['data_name'])
    data_handler = data_cls(data_dir=args.data_dir, results_dir=args.results_dir, **mesh_configs['data'])

    # create an instance of the mesh model
    mesh_ae = DHBMeshAutoencoderModel(data_handler=data_handler,
                                      log_dir=None,
                                      training_params=mesh_configs['training'],
                                      model_params=mesh_configs['model'],
                                      save_metrics=False,  # skip metrics
                                      freeze=True)  # do not make the upsampling and downsampling matrices trainable

    # load mesh ae best model
    mesh_model_path = args.results_dir / "MAE" / mesh_model_exp / "trained_models" / "CoMA_DHB_best"
    mesh_ae.load_weights(mesh_model_path).expect_partial()
    logger.info(f"MAE model loaded from {mesh_model_path}")

    # Models and dataset params
    mesh_model = {"mesh_ae": mesh_ae, "mesh_model_exp": mesh_model_exp}
    echo_model = {"echo_ae": echo_ae, "echo_model_exp": echo_model_exp, "echo_model_nb": echo_model_nb}
    n_train_samples = mesh_configs['training']['n_train_samples']
    n_val_samples = mesh_configs['training']['n_val_samples']
    n_test_samples = mesh_configs['training']['n_test_samples']
    batch_size = echo_to_mesh_configs['training']['batch_size']
    # Load datasets
    test_dataset = data_handler.get_dataset_mesh_slice_pairs_encoded_from_disk(mesh_model, echo_model, "test", n_test_samples)
    test_dataset = utils.tf_dataset_from_tensor_slices(test_dataset, batch_size=batch_size)
    batch_size = echo_to_mesh_configs['training']['batch_size']
    train_dataset = data_handler.get_dataset_mesh_slice_pairs_encoded_from_disk(mesh_model, echo_model, "train", n_train_samples)
    train_dataset = utils.tf_dataset_from_tensor_slices(train_dataset, batch_size=batch_size, repeat=True)
    val_dataset = data_handler.get_dataset_mesh_slice_pairs_encoded_from_disk(mesh_model, echo_model, "val", n_val_samples)
    val_dataset = utils.tf_dataset_from_tensor_slices(val_dataset, batch_size=batch_size)

    echo_to_mesh_model = Echo_to_Mesh_Model(echo_data_dir=args.echo_dir,
                                            log_dir=experiment_path,
                                            mesh_ae=mesh_ae,
                                            training_params=echo_to_mesh_configs['training'],
                                            model_params=echo_to_mesh_configs['model'])

    # Fit Model
    logger.info("Fitting echo to mesh model.")
    echo_to_mesh_model.fit(train_dataset, val_dataset, test_dataset)
    logger.info("DONE!")


def run_cycle_gan(args, cycle_configs, run_exps=False, loaded_model_dir=None):
    # for reproducible results
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if run_exps:
        experiment_path = args.results_dir / "GenModel" / (TIMESTR + "_experiments")
    else:
        experiment_path = args.results_dir / "GenModel" / TIMESTR
    experiment_path.mkdir(parents=True)

    logfile = experiment_path / LOGGER_RESULT_FILE
    utils.setup_logger(logfile, create_stdlog=True)
    if run_exps:
        logger.info("Running Cycle GAN exps")
    else:
        logger.info("Running Cycle GAN")

    logger.info(f"Load GenModel model {loaded_model_dir}")
    logger.info(f"Saving results in {experiment_path}")

    # --------------------------------------------------------------- ECHOS ---------------------------------------------------------
    # Load EAE
    echo_model = cycle_configs['loaded_models']['echo']
    echo_model_exp = echo_model['experiment']
    echo_model_nb = echo_model["model_nb"]

    if echo_model_exp is None:
        echo_model_exp = sorted(os.listdir(args.results_dir / "EAE"))[-1]
        cycle_configs['loaded_models']['echo']['experiment'] = echo_model_exp
        echo_model_nb = 0
        cycle_configs['loaded_models']['echo']['model_nb'] = echo_model_nb

    echo_model_path = args.results_dir / "EAE" / echo_model_exp / "trained_models" / f"echonet_dynamic_{echo_model_nb}"
    echo_ae = load_echonet_dynamic_model(echo_model_path, logger)
    logger.info(f"EAE loaded from {echo_model_path}")
    # Load EAE data
    echo_data_dir = args.echo_dir
    data_info, files = load_data_echonet(logger, echo_data_dir)
    data_info['globalID'] = data_info['FileName'].apply(lambda s: s[:-4]).astype('string')
    bad_quality_echos = utils.bad_quality_echos
    data_info = data_info[~data_info["globalID"].isin(bad_quality_echos)]
    # filter EFs
    min_echo_ef = 0
    max_echo_ef = 100
    logger.info(f"Using echos with efs in [{min_echo_ef}, {max_echo_ef}]")
    data_info_filtered = data_info[(data_info['EF'] >= min_echo_ef) & (data_info['EF'] <= max_echo_ef)]
    filenames_filtered = list(data_info_filtered.index)
    efs = list(data_info_filtered["EF"])

    # get EDF and ESF for echos (skipping echos with nb of traced segments != 42)
    echo_EDF_ESF = get_echo_edf_esf(echo_data_dir, logger)

    all_echo_edfs = {}
    for _, row in echo_EDF_ESF.iterrows():
        all_echo_edfs[row["FileName"]] = row["EDF"]

    # hard coded values, use param in config file later
    n = len(filenames_filtered)  # nb data samples
    n_train_samples = int(0.95 * n)  # 95% of files for train
    n_val_samples = (n - n_train_samples)  # the remaining samples for validation
    n_test_samples = 0  # no test files

    # Select train, val and test sets (with val set containing varied EFs (as varied as possible) from the echos dataset)
    val_dataset_indices = utils.spaced_efs_indices(efs, n_val_samples)
    val_filenames = [filenames_filtered[i] for i in val_dataset_indices]
    val_filenames = [filename for filename in val_filenames if filename in all_echo_edfs]  # filter those without echo edfs
    val_efs = [data_info_filtered.loc[f]["EF"] for f in val_filenames]

    train_filenames = [filenames_filtered[i] for i in range(n) if i not in val_dataset_indices]
    train_filenames = [filename for filename in train_filenames if filename in all_echo_edfs]  # filter those without echo edfs
    train_efs = [data_info_filtered.loc[f]["EF"] for f in train_filenames]

    # remove val_filenames whose ef not in the train_dataset efs range
    min_train_ef, max_train_ef = min(train_efs), max(train_efs)
    val_filenames = [tup[1] for tup in list(zip(val_efs, val_filenames)) if (tup[0] >= min_train_ef and tup[0] <= max_train_ef)]
    val_efs = [data_info_filtered.loc[f]["EF"] for f in val_filenames]

    test_filenames = []

    n_train_viz = int(0.1 * len(train_filenames))
    train_viz_dataset_indices = utils.spaced_efs_indices(train_efs, n_train_viz)
    train_filenames_viz = [train_filenames[i] for i in train_viz_dataset_indices]

    n_train_render = 10
    train_render_dataset_indices = utils.spaced_efs_indices(train_efs, n_train_render)
    train_filenames_render = [train_filenames[i] for i in train_render_dataset_indices]
    train_efs_render = [data_info_filtered.loc[f]["EF"] for f in train_filenames_render]
    train_efs_filenames_render_sorted = sorted(list(zip(train_efs_render, train_filenames_render)),
                                               key=lambda tup: tup[0])  # create tuples (ef, filename) and sort by ef
    train_filenames_render = [tup[1] for tup in train_efs_filenames_render_sorted]  # extract sorted train_render filenames

    n_val_render = 10
    val_render_dataset_indices = utils.spaced_efs_indices(val_efs, n_val_render)
    val_filenames_render = [val_filenames[i] for i in val_render_dataset_indices]
    val_efs_render = [data_info_filtered.loc[f]["EF"] for f in val_filenames_render]
    val_efs_filenames_render_sorted = sorted(list(zip(val_efs_render, val_filenames_render)),
                                             key=lambda tup: tup[0])  # create tuples (ef, filename) and sort by ef
    val_filenames_render = [tup[1] for tup in val_efs_filenames_render_sorted]  # extract sorted val_render filenames

    n_train_samples = len(train_filenames)
    n_val_samples = len(val_filenames)
    n_test_samples = len(test_filenames)
    n_total = n_train_samples + n_val_samples + n_test_samples
    logger.info(f"ECHO: {n_train_samples} train samples, {n_val_samples} val samples, {n_test_samples} test samples")
    logger.info(f"Total echo files: {n_total} samples")

    # EFs hists data
    plot_dir = experiment_path / "plots" / "echo_efs_plots"
    bins = bins = list(range(5, 101, 1))
    efs_full = efs
    efs_train = list(data_info_filtered[data_info_filtered["globalID"].isin(train_filenames)]["EF"])
    efs_val = list(data_info_filtered[data_info_filtered["globalID"].isin(val_filenames)]["EF"])
    # EFs hists plotting
    utils.hist_plot(plot_dir, "EchoEFHist_full.png", efs_full, bins, 'Histogram of Echo EFs (Full dataset)')
    utils.hist_plot(plot_dir, "EchoEFHist_train.png", efs_train, bins, 'Histogram of Echo EFs (Train dataset)')
    utils.hist_plot(plot_dir, "EchoEFHist_val.png", efs_val, bins, 'Histogram of Echo EFs (Val dataset)')

    # get all filepaths of filtered data
    all_files = [files[f] for f in list(data_info_filtered['globalID'])]
    all_filenames = [Path(f).stem for f in all_files]

    # Encode the whole echo dataset, in same order as filepaths
    encoded_echo_output_dir = echo_data_dir / "EchoNet-Dynamic" / "encoded_echos" / f"{echo_model_exp}" / f"echonet_dynamic_{echo_model_nb}"
    echo_latents, echo_efs = get_encoded_dataset(all_files, echo_ae, logger, encoded_echo_output_dir)

    echo_filename_data_enc = {filename: (latent, ef) for filename, latent, ef in zip(all_filenames, echo_latents, echo_efs)}
    echo_train_data_enc = [echo_filename_data_enc[f] for f in train_filenames]
    echo_val_data_enc = [echo_filename_data_enc[f] for f in val_filenames]
    echo_train_latents, echo_train_efs = np.array([data[0] for data in echo_train_data_enc]), np.array([data[1] for data in echo_train_data_enc])
    echo_val_latents, echo_val_efs = np.array([data[0] for data in echo_val_data_enc]), np.array([data[1] for data in echo_val_data_enc])

    echo_batch_size = cycle_configs['data_echo']['batch_size']
    echo_encoded_train_dataset = echo_train_latents, echo_train_efs
    echo_encoded_val_dataset = echo_val_latents, echo_val_efs
    echo_encoded_train_dataset = utils.tf_dataset_from_tensor_slices(echo_encoded_train_dataset, echo_batch_size, repeat=True)
    echo_encoded_val_dataset = utils.tf_dataset_from_tensor_slices(echo_encoded_val_dataset, echo_batch_size, repeat=False)

    echo_filenames = {"train": train_filenames,
                      "train_viz": train_filenames_viz,
                      "train_render": train_filenames_render,
                      "val_render": val_filenames_render,
                      "val": val_filenames,
                      "test": None}

    echo_filepaths = {"train": [files[f] for f in train_filenames],
                      "train_viz": [files[f] for f in train_filenames_viz],
                      "train_render": [files[f] for f in train_filenames_render],
                      "val_render": [files[f] for f in val_filenames_render],
                      "val": [files[f] for f in val_filenames],
                      "test": None}

    echo_datasets = {"train": get_dataset(echo_filepaths["train"], batch_size=32, repeat=True, subsample=False),
                     "train_viz": get_dataset(echo_filepaths["train_viz"], batch_size=4, subsample=False),
                     "train_render": get_dataset(echo_filepaths["train_render"], batch_size=1, subsample=False),
                     "val_render": get_dataset(echo_filepaths["val_render"], batch_size=1, subsample=False),
                     "val": get_dataset(echo_filepaths["val"], batch_size=4, subsample=False),
                     "test": None}

    echo_datasets_edfs = {"train": [all_echo_edfs[f] for f in echo_filenames["train"]],
                          "train_viz": [all_echo_edfs[f] for f in echo_filenames["train_viz"]],
                          "train_render": [all_echo_edfs[f] for f in echo_filenames["train_render"]],
                          "val_render": [all_echo_edfs[f] for f in echo_filenames["val_render"]],
                          "val": [all_echo_edfs[f] for f in echo_filenames["val"]],
                          "test": None}

    echo_datasets_enc = {"train": echo_encoded_train_dataset,
                         "train_viz": None,
                         "train_render": None,
                         "val_render": None,
                         "val": echo_encoded_val_dataset,
                         "test": None}

    # --------------------------------------------------------------- MESHES --------------------------------------------------------
    # Load MAE config
    mesh_model_exp = cycle_configs['loaded_models']['mesh']['experiment']
    if mesh_model_exp is None:
        mesh_model_exp = sorted(os.listdir(args.results_dir / "MAE"))[-1]
        cycle_configs['loaded_models']['mesh']['experiment'] = mesh_model_exp

    mesh_config_path = args.results_dir / "MAE" / mesh_model_exp / "configs.yml"
    with mesh_config_path.open(mode='r') as yamlfile:
        mesh_configs = yaml.safe_load(yamlfile)
    # get class (cls) for data generation
    data_cls = utils.get_data_handler_cls(mesh_configs['data']['data_name'])
    data_handler = data_cls(data_dir=args.data_dir, results_dir=args.results_dir, **mesh_configs['data'])

    n_train_samples = mesh_configs['training']['n_train_samples']
    n_val_samples = mesh_configs['training']['n_val_samples']
    n_test_samples = mesh_configs['training']['n_test_samples']
    logger.info(f"MESH: {n_train_samples} train samples, {n_val_samples} val samples and {n_test_samples} test samples")

    # create an instance of the mesh model
    mesh_ae = DHBMeshAutoencoderModel(data_handler=data_handler,
                                      log_dir=None,
                                      training_params=mesh_configs['training'],
                                      model_params=mesh_configs['model'],
                                      save_metrics=False,  # skip metrics
                                      freeze=True)  # do not make the upsampling and downsampling matrices trainable

    # load mesh ae best model
    mesh_model_path = args.results_dir / "MAE" / mesh_model_exp / "trained_models" / "CoMA_DHB_best"
    mesh_ae.load_weights(mesh_model_path).expect_partial()
    logger.info(f"MAE model loaded from {mesh_model_path}")

    # get encoded mesh train and val dataset
    mesh_batch_size = cycle_configs['data_mesh']['batch_size']
    mesh_encoded_train_dataset = data_handler.get_encoded_ef_data(mesh_model_exp, mesh_ae, "train", n_train_samples)
    mesh_encoded_val_dataset = data_handler.get_encoded_ef_data(mesh_model_exp, mesh_ae, "val", n_val_samples)
    mesh_datasets_enc = {}
    mesh_datasets_enc["train"] = utils.tf_dataset_from_tensor_slices(mesh_encoded_train_dataset, mesh_batch_size, repeat=True)
    mesh_datasets_enc["val"] = utils.tf_dataset_from_tensor_slices(mesh_encoded_val_dataset, mesh_batch_size)

    # ----------------------------------------------------------- MESH EF PRED -------------------------------------------------------
    # Load Mesh EF Pred
    mesh_ef_pred_model_exp = cycle_configs['loaded_models']['mesh_ef_pred']['experiment']
    if mesh_ef_pred_model_exp is None:
        # take most recent model if none defined
        mesh_ef_pred_model_exp = sorted(os.listdir(args.results_dir / "MESH_EF"))[-1]
        cycle_configs['loaded_models']['mesh_ef_pred']['experiment'] = mesh_ef_pred_model_exp

    mesh_ef_config_path = args.results_dir / "MESH_EF" / mesh_ef_pred_model_exp / "configs.yml"
    with (mesh_ef_config_path).open(mode='r') as yamlfile:
        mesh_ef_configs = yaml.safe_load(yamlfile)
    mesh_ef_pred = Mesh_EF_Predictor(
        log_dir=None,
        training_params=mesh_ef_configs['training'],
        model_params=mesh_ef_configs['model'],
        save_metrics=False
    )
    # init mesh EF model
    zeros = np.zeros(shape=(1, 18))
    mesh_ef_pred(zeros)
    # load mesh EF pred model
    mesh_ef_model_path = args.results_dir / "MESH_EF" / mesh_ef_pred_model_exp / "trained_models" / "EF_pred_best"
    mesh_ef_pred.load_weights(mesh_ef_model_path).expect_partial()
    logger.info(f"Mesh ef pred model loaded from {mesh_ef_model_path}")

    # Save a copy of configs in the log dir
    with (experiment_path / 'configs.yml').open(mode='w') as yaml_file:
        to_save = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in cycle_configs.items()}
        yaml.dump(to_save, yaml_file)

    # w = mesh_ef_pred.trainable_weights
    # logger.info(f"Mesh EF pred weights: {w}")
    # exit(0)

    # ------------------------------------------------------------ CYCLE GAN --------------------------------------------------------
    # Create the CycleGan model
    if run_exps:
        cycle_gan = CycleGan(echo_ae=echo_ae,
                             mesh_ae=mesh_ae,
                             mesh_ef_pred=mesh_ef_pred,
                             log_dir=experiment_path,
                             echo_data_dir=echo_data_dir,
                             training_params=cycle_configs['training'],
                             model_params=cycle_configs['model'],
                             save_metrics=False)
        logger.info("Running Cycle GAN exps")
        cycle_gan_model_path = args.results_dir / "GenModel" / loaded_model_dir / "trained_models" / "cycleGAN_best"
        cycle_gan.load_weights(cycle_gan_model_path).expect_partial()
        logger.info(f"Cycle GAN model loaded from {cycle_gan_model_path}")
        # Run exps
        exps_list = ["Render", "Echo_rec", "Overlay", "EF_train", "EF_val", "IoU"]
        cycle_gan.run_experiments(exps_list, echo_datasets, echo_filenames, echo_datasets_edfs, 0, False, True)
    else:
        cycle_gan = CycleGan(echo_ae=echo_ae,
                             mesh_ae=mesh_ae,
                             mesh_ef_pred=mesh_ef_pred,
                             log_dir=experiment_path,
                             echo_data_dir=echo_data_dir,
                             training_params=cycle_configs['training'],
                             model_params=cycle_configs['model'])

        # Fit Model
        logger.info("Fitting generative model.")
        cycle_gan.fit(mesh_datasets_enc, echo_datasets_enc, echo_datasets, echo_filenames, echo_datasets_edfs)
        logger.info("DONE!")


def run_cycle_gan_10_fold(args, cycle_configs):
    # for reproducible results
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    experiment_path = args.results_dir / "GenModel" / (TIMESTR + "_10_fold")
    experiment_path.mkdir(parents=True)

    # Save a copy of configs in the log dir
    with (experiment_path / 'configs.yml').open(mode='w') as yaml_file:
        to_save = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in cycle_configs.items()}
        yaml.dump(to_save, yaml_file)

    logfile = experiment_path / LOGGER_RESULT_FILE
    utils.setup_logger(logfile, create_stdlog=True)
    logger.info("Running Cycle GAN 10 times")
    logger.info(f"Saving results in {experiment_path}")

    # Load DHB MAE config
    mesh_model_exp = cycle_configs['loaded_models']['mesh']['experiment']

    if mesh_model_exp is None:
        mesh_model_exp = sorted(os.listdir(args.results_dir / "MAE"))[-1]
        cycle_configs['loaded_models']['mesh']['experiment'] = mesh_model_exp

    mesh_config_path = args.results_dir / "MAE" / mesh_model_exp / "configs.yml"
    with mesh_config_path.open(mode='r') as yamlfile:
        mesh_configs = yaml.safe_load(yamlfile)

    # get class (cls) for data generation
    data_cls = utils.get_data_handler_cls(mesh_configs['data']['data_name'])
    data_handler = data_cls(data_dir=args.data_dir, results_dir=args.results_dir, **mesh_configs['data'])

    mesh_n_train_samples = mesh_configs['training']['n_train_samples']
    mesh_n_val_samples = mesh_configs['training']['n_val_samples']
    mesh_n_test_samples = mesh_configs['training']['n_test_samples']
    logger.info(f"MESH: {mesh_n_train_samples} train samples, {mesh_n_val_samples} val samples and {mesh_n_test_samples} test samples")

    # create an instance of the mesh model
    mesh_ae = DHBMeshAutoencoderModel(data_handler=data_handler,
                                      log_dir=None,
                                      training_params=mesh_configs['training'],
                                      model_params=mesh_configs['model'],
                                      save_metrics=False,  # skip metrics
                                      freeze=True)  # do not make up-/down-sampling matrices trainable

    # load mesh ae best model
    mesh_model_path = args.results_dir / "MAE" / mesh_model_exp / "trained_models" / "CoMA_DHB_best"
    mesh_ae.load_weights(mesh_model_path).expect_partial()
    logger.info(f"MAE model loaded from {mesh_model_path}")

    # get encoded mesh train dataset (batched and repeated) and val dataset (not batched, nor repeated)
    mesh_batch_size = cycle_configs['data_mesh']['batch_size']
    mesh_encoded_train_dataset = data_handler.get_encoded_ef_data(mesh_model_exp, mesh_ae, "train", mesh_n_train_samples)
    latents, efs = mesh_encoded_train_dataset
    logger.info(f"\nMesh latents max_vals: {np.amax(latents, axis=0)}")
    logger.info(f"Mesh latents min_vals: {np.amin(latents, axis=0)}\n")
    mesh_encoded_val_dataset = data_handler.get_encoded_ef_data(mesh_model_exp, mesh_ae, "val", mesh_n_val_samples)

    # Load Mesh EF Pred
    mesh_ef_pred_model_exp = cycle_configs['loaded_models']['mesh_ef_pred']['experiment']
    if mesh_ef_pred_model_exp is None:
        # take most recent model if none defined
        mesh_ef_pred_model_exp = sorted(os.listdir(args.results_dir / "MESH_EF"))[-1]
        cycle_configs['loaded_models']['mesh_ef_pred']['experiment'] = mesh_ef_pred_model_exp

    mesh_ef_config_path = args.results_dir / "MESH_EF" / mesh_ef_pred_model_exp / "configs.yml"
    with mesh_ef_config_path.open(mode='r') as yamlfile:
        mesh_ef_configs = yaml.safe_load(yamlfile)

    mesh_ef_pred = Mesh_EF_Predictor(
        log_dir=None,
        training_params=mesh_ef_configs['training'],
        model_params=mesh_ef_configs['model'],
        save_metrics=False)

    mesh_ef_model_path = args.results_dir / "MESH_EF" / mesh_ef_pred_model_exp / "trained_models" / "EF_pred_best"
    mesh_ef_pred.load_weights(mesh_ef_model_path).expect_partial()
    logger.info(f"Mesh ef pred model loaded from {mesh_ef_model_path}")

    # Load DHB AE
    echo_model = cycle_configs['loaded_models']['echo']
    echo_model_exp = echo_model['experiment']
    echo_model_nb = echo_model["model_nb"]

    if echo_model_exp is None:
        echo_model_exp = sorted(os.listdir(args.results_dir / "EAE"))[-1]
        cycle_configs['loaded_models']['echo']['experiment'] = echo_model_exp
        echo_model_nb = 0
        cycle_configs['loaded_models']['echo']['model_nb'] = echo_model_nb

    echo_model_path = args.results_dir / "EAE" / echo_model_exp / "trained_models" / f"echonet_dynamic_{echo_model_nb}"
    echo_ae = load_echonet_dynamic_model(echo_model_path, logger)
    logger.info(f"EAE loaded from {echo_model_path}")

    # Load DHB AE data
    echo_data_dir = args.echo_dir

    data_info, filepaths = load_data_echonet(logger, echo_data_dir)
    data_info['globalID'] = data_info['FileName'].apply(lambda s: s[:-4]).astype('string')

    # get EDF and ESF for echos (skipping echos with nb of traced segments != 42)
    echo_EDF_ESF = get_echo_edf_esf(echo_data_dir, logger)

    all_echo_edfs = {}
    for _, row in echo_EDF_ESF.iterrows():
        all_echo_edfs[row["FileName"]] = row["EDF"]

    # filter echo data info to only include echos with EDF and ESF frame nbs
    echo_EDF_ESF_files = list(echo_EDF_ESF["FileName"])
    data_info = data_info[data_info['globalID'].isin(echo_EDF_ESF_files)]

    min_echo_ef = 0
    max_echo_ef = 100
    logger.info(f"Using echos with efs in [{min_echo_ef}, {max_echo_ef}]")
    data_info = data_info[(data_info['EF'] >= min_echo_ef) & (data_info['EF'] <= max_echo_ef)]

    # get all filepaths of filtered data
    filepaths = [filepaths[f] for f in list(data_info['globalID'])]
    filepaths = shuffle(filepaths, random_state=args.seed)

    folds = 10
    kf = KFold(n_splits=folds)

    # Encode the whole echo dataset, in same order as filepaths
    encoded_echo_output_dir = echo_data_dir / "EchoNet-Dynamic" / "encoded_echos" / f"{echo_model_exp}" / f"echonet_dynamic_{echo_model_nb}"
    echo_latents, echo_efs = get_encoded_dataset(filepaths, echo_ae, logger, encoded_echo_output_dir)
    logger.info(f"\nEcho latents max_vals: {np.amax(echo_latents, axis=0)}")
    logger.info(f"Echo latents min_vals: {np.amin(echo_latents, axis=0)}\n")

    echo_batch_size = cycle_configs['data_echo']['batch_size']
    all_echo_test_filenames = []
    all_echo_test_model_nbs = []

    for i, (train_val_index, test_index) in enumerate(kf.split(filepaths)):  # for each of the k-fold splits
        logger.info(f"\nTraining procedure {i}:\n")

        train_index, val_index = train_test_split(train_val_index, test_size=0.1, random_state=i)
        echo_n_train_samples, echo_n_val_samples, echo_n_test_samples = len(train_index), len(val_index), len(test_index)

        logger.info(f"ECHO: {echo_n_train_samples} train samples, {echo_n_val_samples} val samples, {echo_n_test_samples} test samples")
        logger.info(f"Total echo files: {len(filepaths)} samples")

        # get encoded echo train and val datasets
        echo_encoded_train_dataset = echo_latents[train_index], echo_efs[train_index]
        echo_encoded_train_dataset = utils.tf_dataset_from_tensor_slices(echo_encoded_train_dataset, echo_batch_size, repeat=True)
        echo_encoded_val_dataset = echo_latents[val_index], echo_efs[val_index]
        echo_encoded_val_dataset = utils.tf_dataset_from_tensor_slices(echo_encoded_val_dataset, echo_batch_size, repeat=False)

        val_files = [filepaths[i] for i in val_index]
        train_files = [filepaths[i] for i in train_index]

        val_filenames = [Path(f).stem for f in [filepaths[i] for i in val_index]]
        train_filenames = [Path(f).stem for f in [filepaths[i] for i in train_index]]

        n_train_viz = int(0.1 * len(train_filenames))
        train_viz_dataset_indices = utils.spaced_efs_indices(echo_efs[train_index], n_train_viz)
        train_filenames_viz = [train_filenames[i] for i in train_viz_dataset_indices]

        # render data
        n_render = 10
        val_dataset_render_indices = utils.spaced_efs_indices(echo_efs[val_index], n_render)
        train_dataset_render_indices = utils.spaced_efs_indices(echo_efs[train_index], n_render)

        val_files_render = [val_files[i] for i in val_dataset_render_indices]
        train_files_render = [train_files[i] for i in train_dataset_render_indices]

        val_filenames_render = [val_filenames[i] for i in val_dataset_render_indices]
        train_filenames_render = [train_filenames[i] for i in train_dataset_render_indices]

        val_efs_rendered = [data_info.loc[f]["EF"] for f in val_filenames_render]
        train_efs_rendered = [data_info.loc[f]["EF"] for f in train_filenames_render]

        val_efs_filenames_render_sorted = sorted(list(zip(val_efs_rendered, val_filenames_render)), key=lambda tup: tup[0])
        train_efs_filenames_render_sorted = sorted(list(zip(train_efs_rendered, train_filenames_render)), key=lambda tup: tup[0])

        val_filenames_render = [tup[1] for tup in val_efs_filenames_render_sorted]  # extract sorted val_render filenames
        train_filenames_render = [tup[1] for tup in train_efs_filenames_render_sorted]  # extract sorted val_render filenames

        # gather echo data (filepaths, filenames, encoded_echo, original_echos, edfs, ...)
        echo_filepaths = {"train": [filepaths[i] for i in train_index],
                          "val": [filepaths[i] for i in val_index],
                          "test": [filepaths[i] for i in test_index],
                          "train_viz": [filepaths[f] for f in train_viz_dataset_indices],
                          "train_render": train_files_render,
                          "val_render": val_files_render,
                          }

        echo_filenames = {"train": [Path(f).stem for f in echo_filepaths["train"]],
                          "val": [Path(f).stem for f in echo_filepaths["val"]],
                          "test": [Path(f).stem for f in echo_filepaths["test"]],
                          "train_viz": train_filenames_viz,
                          "train_render": train_filenames_render,
                          "val_render": val_filenames_render,
                          }

        echo_datasets_enc = {"train": echo_encoded_train_dataset,
                             "val": echo_encoded_val_dataset,
                             "test": None,
                             "train_viz": None,

                             "train_render": None,
                             "val_render": None,
                             }

        echo_datasets = {"train": get_dataset(echo_filepaths["train"], batch_size=32, subsample=False),
                         "val": get_dataset(echo_filepaths["val"], batch_size=4, subsample=False),
                         "test": get_dataset(echo_filepaths["test"], batch_size=4, subsample=False),
                         "train_viz": get_dataset(echo_filepaths["train_viz"], batch_size=4, subsample=False),
                         "train_render": get_dataset(echo_filepaths["train_render"], batch_size=1, subsample=False),
                         "val_render": get_dataset(echo_filepaths["val_render"], batch_size=1, subsample=False),
                         }

        echo_datasets_edfs = {"train": [all_echo_edfs[f] for f in echo_filenames["train"]],
                              "val": [all_echo_edfs[f] for f in echo_filenames["val"]],
                              "test": [all_echo_edfs[f] for f in echo_filenames["test"]],
                              "train_viz": [all_echo_edfs[f] for f in echo_filenames["train_viz"]],
                              "train_render": [all_echo_edfs[f] for f in echo_filenames["train_render"]],
                              "val_render": [all_echo_edfs[f] for f in echo_filenames["val_render"]],
                              }

        # add test filenames to list
        all_echo_test_filenames.extend(echo_filenames["test"])
        model_nbs = [i] * len(echo_filenames["test"])
        all_echo_test_model_nbs.extend(model_nbs)

        # get mesh train and val data as tf datasets (shuffled, batched), repeat mesh_train dataset
        # limit mesh val dataset to have as many samples as echo_val_dataset
        mesh_datasets_enc = {}
        mesh_datasets_enc["train"] = utils.tf_dataset_from_tensor_slices(mesh_encoded_train_dataset, batch_size=mesh_batch_size,
                                                                         shuffle_buffer=mesh_n_train_samples, shuffle_seed=i, repeat=True)
        mesh_datasets_enc["val"] = utils.tf_dataset_from_tensor_slices(mesh_encoded_val_dataset, batch_size=mesh_batch_size, shuffle_buffer=mesh_n_val_samples,
                                                                       shuffle_seed=i, samples_taken=echo_n_val_samples)
        mesh_datasets_enc["test"] = None

        latents, _ = mesh_encoded_train_dataset

        # Create the CycleGan model
        log_dir = experiment_path / f"model_{i}"
        log_dir.mkdir(parents=True, exist_ok=True)
        cycle_gan = CycleGan(echo_ae=echo_ae,
                             mesh_ae=mesh_ae,
                             mesh_ef_pred=mesh_ef_pred,
                             echo_data_dir=echo_data_dir,
                             log_dir=log_dir,
                             training_params=cycle_configs['training'],
                             model_params=cycle_configs['model'])

        logger.info(f"Fitting Cycle GAN model {i}.")
        cycle_gan.fit(mesh_datasets_enc, echo_datasets_enc, echo_datasets, echo_filenames, echo_datasets_edfs)  # fit the Cycle GAN model and get best model
        cycle_gan.save_me()  # save trained model weights one last time

        keras.backend.clear_session()
        del cycle_gan
        time.sleep(10)

        logger.info(f"Finished {i+1}/{10}")

    echo_test_data = pd.DataFrame({"FileName": all_echo_test_filenames, "Model": all_echo_test_model_nbs})
    test_data_file = experiment_path / "TestEchosModels.csv"
    echo_test_data.to_csv(test_data_file, index=False)
    logger.info(f"Done {folds} trainings, models saved!")

    logger.info("DONE!")


def run_mesh_ef_pred(args, mesh_ef_configs):
    # for reproducible results
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    experiment_path = args.results_dir / "MESH_EF" / TIMESTR
    experiment_path.mkdir(parents=True)

    logfile = experiment_path / LOGGER_RESULT_FILE
    utils.setup_logger(logfile, create_stdlog=True)
    logger.info("Running MAE EF predictor")
    logger.info(f"Saving results in {experiment_path}")

    # Load MAE config
    mesh_model_exp = mesh_ef_configs['loaded_model']['experiment']

    if mesh_model_exp is None:
        mesh_model_exp = sorted(os.listdir(args.results_dir / "MAE"))[-1]
        mesh_ef_configs['loaded_model']['experiment'] = mesh_model_exp

    logger.info(f"Loading mesh model exp: {mesh_model_exp}")

    # Save a copy of configs in the log dir
    with (experiment_path / 'configs.yml').open(mode='w') as yaml_file:
        to_save = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in mesh_ef_configs.items()}
        yaml.dump(to_save, yaml_file)

    mesh_config_path = args.results_dir / "MAE" / mesh_model_exp / "configs.yml"
    with mesh_config_path.open(mode='r') as yamlfile:
        mesh_configs = yaml.safe_load(yamlfile)
    # get class (cls) for data generation
    data_cls = utils.get_data_handler_cls(mesh_configs['data']['data_name'])
    data_handler = data_cls(data_dir=args.data_dir, results_dir=args.results_dir, **mesh_configs['data'])

    n_train_samples = mesh_configs['training']['n_train_samples']
    n_val_samples = mesh_configs['training']['n_val_samples']
    n_test_samples = mesh_configs['training']['n_test_samples']
    logger.info(f"MESH: {n_train_samples} train samples, {n_val_samples} val samples and {n_test_samples} test samples")

    # create an instance of the mesh model
    mesh_ae_log_dir = args.results_dir / "MAE" / mesh_model_exp  # path to pass to the model to load best model using "load_me"
    mesh_ae = DHBMeshAutoencoderModel(data_handler=data_handler,
                                      log_dir=mesh_ae_log_dir,
                                      training_params=mesh_configs['training'],
                                      model_params=mesh_configs['model'],
                                      save_metrics=False,  # skip metrics
                                      freeze=True)  # do not make variables trainable

    mesh_ae.load_me(0, name="best")  # load best model

    # Create datasets
    batch_size_encoded = mesh_ef_configs['data_mesh']['batch_size']
    datasets = {}
    # get encoded data
    datasets["train"] = data_handler.get_encoded_ef_data(mesh_model_exp, mesh_ae, "train", n_train_samples, return_true_efs=False)
    datasets["val"] = data_handler.get_encoded_ef_data(mesh_model_exp, mesh_ae, "val", n_val_samples, return_true_efs=False)
    datasets["test"] = data_handler.get_encoded_ef_data(mesh_model_exp, mesh_ae, "test", n_test_samples, return_true_efs=False)

    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_test, y_test = datasets["test"]

    # EF values in [0, 1]
    y_train, y_val, y_test = y_train / 100.0, y_val / 100.0, y_test / 100.0

    # # normalize to range [0, 1]
    # logger.info("Normalizing to [0, 1]")
    # min_val = min([X_train.min(), X_val.min(), X_test.min()])
    # max_val = max([X_train.max(), X_val.max(), X_test.max()])
    # scale = max_val - min_val

    # X_train = (X_train - min_val) / scale
    # X_val = (X_val - min_val) / scale
    # X_test = (X_test - min_val) / scale

    # # scale to 0-mean and 1 std dev
    # logger.info("Normalizing to 0-mean, 1-std dev")
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)

    # # normalize to range [-1. 1]
    # logger.info("Normalizing to range [-1, 1]")
    # scaler = MinMaxScaler((-1, 1))
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)

    datasets["train"] = X_train, y_train
    datasets["val"] = X_val, y_val
    datasets["test"] = X_test, y_test

    # dataset for plotting efs on train set
    datasets["train_plotting"] = X_train, y_train

    # convert numpy data to tf dataset
    for name in datasets:
        tf_dataset = tf.data.Dataset.from_tensor_slices(datasets[name])
        tf_dataset = tf_dataset.batch(batch_size_encoded)
        if name == "train":  # repeat train dataset
            tf_dataset = tf_dataset.repeat()

        datasets[name] = tf_dataset

    # delete mesh ae model
    keras.backend.clear_session()
    del mesh_ae

    logger.info("Deleted mesh ae model. Done!")

    # Create the Mesh Predictor model
    mesh_ef_predictor = Mesh_EF_Predictor(
        log_dir=experiment_path,
        training_params=mesh_ef_configs['training'],
        model_params=mesh_ef_configs['model']
    )

    # Fit Model
    logger.info("Fitting mesh ef predictor model.")
    mesh_ef_predictor.fit(datasets["train"], datasets["train_plotting"], datasets["val"], datasets["test"])

    logger.info("DONE!")


def run_echo_ef_pred(args, echo_ef_configs):
    # for reproducible results
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    experiment_path = args.results_dir / "ECHO_EF" / TIMESTR
    experiment_path.mkdir(parents=True)

    logfile = experiment_path / LOGGER_RESULT_FILE
    utils.setup_logger(logfile, create_stdlog=True)
    logger.info("Running EAE EF predictor")
    logger.info(f"Saving results in {experiment_path}")

    # Load EAE
    echo_model = echo_ef_configs['loaded_model']
    echo_model_exp = echo_model['experiment']
    echo_model_nb = echo_model["model_nb"]

    if echo_model_exp is None:
        echo_model_exp = sorted(os.listdir(args.results_dir / "EAE"))[-1]
        echo_model_nb = 0
        echo_ef_configs['loaded_model']['experiment'] = echo_model_exp
        echo_ef_configs['loaded_model']["model_nb"] = echo_model_nb

    # Save a copy of configs in the log dir
    with (experiment_path / 'configs.yml').open(mode='w') as yaml_file:
        to_save = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in echo_ef_configs.items()}
        yaml.dump(to_save, yaml_file)

    echo_model_path = args.results_dir / "EAE" / echo_model_exp / "trained_models" / f"echonet_dynamic_{echo_model_nb}"
    logger.info(f"Loading echo model: {echo_model_path}")

    echo_ae = load_echonet_dynamic_model(echo_model_path, logger)

    # Load  echo data
    echo_data_dir = args.echo_dir
    data_info, files = load_data_echonet(logger, echo_data_dir)
    # train, val and test split
    train_files = list(data_info[data_info['Split'] == 'TRAIN'].index)
    val_files = list(data_info[data_info['Split'] == 'VAL'].index)
    test_files = list(data_info[data_info['Split'] == 'TEST'].index)
    # get paths
    train_files = [files[f] for f in train_files]
    val_files = [files[f] for f in val_files]
    test_files = [files[f] for f in test_files]

    logger.info(f"ECHO: {len(train_files)} train samples, {len(val_files)} val samples and {len(test_files)} test samples")

    # create datasets
    batch_size_encoded = echo_ef_configs['data_echo']['batch_size']
    datasets = {}

    # get encoded train dataset
    datasets['train'] = get_encoded_dataset(train_files, echo_ae, logger)
    datasets['val'] = get_encoded_dataset(val_files, echo_ae, logger)
    datasets['test'] = get_encoded_dataset(test_files, echo_ae, logger)

    # pre-process data
    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_test, y_test = datasets["test"]

    # EF values in [0, 1]
    y_train, y_val, y_test = y_train / 100.0, y_val / 100.0, y_test / 100.0

    # normalize data
    # min_val = min([X_train.min(), X_val.min(), X_test.min()])
    # max_val = max([X_train.max(), X_val.max(), X_test.max()])
    # scale = max_val - min_val

    # X_train = (X_train - min_val) / scale
    # X_val = (X_val - min_val) / scale
    # X_test = (X_test - min_val) / scale

    # scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    datasets["train"] = X_train, y_train
    datasets["val"] = X_val, y_val
    datasets["test"] = X_test, y_test

    # dataset for plotting efs on train set
    datasets["train_plotting"] = X_train, y_train

    # convert numpy data to tf dataset
    for name in datasets:
        tf_dataset = tf.data.Dataset.from_tensor_slices(datasets[name])
        tf_dataset = tf_dataset.batch(batch_size_encoded)
        if name == "train":  # repeat train dataset
            tf_dataset = tf_dataset.repeat()

        datasets[name] = tf_dataset

    # delete mesh ae model
    keras.backend.clear_session()
    del echo_ae

    logger.info("Deleted echo ae model. Done!")

    # Create the ECHO EF Predictor model
    echo_ef_predictor = Echo_EF_Predictor(
        log_dir=experiment_path,
        training_params=echo_ef_configs['training'],
        model_params=echo_ef_configs['model']
    )

    # Fit Model
    logger.info("Fitting echo ef predictor model.")
    echo_ef_predictor.fit(datasets["train"], datasets["train_plotting"], datasets["val"], datasets["test"])

    logger.info("DONE!")


def run_ef_conv(args, ef_conv_configs):
    # for reproducible results
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    experiment_path = args.results_dir / "2D_TO_3D_EF" / TIMESTR
    experiment_path.mkdir(parents=True)

    # Save a copy of configs in the log dir
    with (experiment_path / 'configs.yml').open(mode='w') as yaml_file:
        to_save = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in ef_conv_configs.items()}
        yaml.dump(to_save, yaml_file)

    logfile = experiment_path / LOGGER_RESULT_FILE
    utils.setup_logger(logfile, create_stdlog=True)
    logger.info("Running 2D to 3D EF converter")
    logger.info(f"Saving results in {experiment_path}")

    # Load EAE config
    mesh_model_exp = ef_conv_configs['data_ef']['experiment']
    mesh_config_path = args.results_dir / "MAE" / mesh_model_exp / "configs.yml"
    with (mesh_config_path).open(mode='r') as yamlfile:
        mesh_configs = yaml.safe_load(yamlfile)
    # get class (cls) for data generation
    data_cls = utils.get_data_handler_cls(mesh_configs['data']['data_name'])
    data_handler = data_cls(data_dir=args.data_dir, results_dir=args.results_dir, **mesh_configs['data'])

    n_train_samples = mesh_configs['training']['n_train_samples']
    n_val_samples = mesh_configs['training']['n_val_samples']
    n_test_samples = mesh_configs['training']['n_test_samples']
    logger.info(f"MESH DATA: {n_train_samples} train samples, {n_val_samples} val samples and {n_test_samples} test samples")

    # create an instance of the mesh model
    mesh_ae = DHBMeshAutoencoderModel(data_handler=data_handler,
                                      log_dir=None,
                                      training_params=mesh_configs['training'],
                                      model_params=mesh_configs['model'],
                                      save_metrics=False,  # skip metrics
                                      build_MAE=False)  # skip building the model components and only set the data_handler values (ref_poly, ...)

    # Create datasets
    batch_size_efs = ef_conv_configs['data_ef']['batch_size']
    datasets = {}
    # get 2d and 3d ef data
    datasets["train"] = data_handler.get_2d_3d_ef_data("train", n_train_samples)
    datasets["val"] = data_handler.get_2d_3d_ef_data("val", n_val_samples)
    datasets["test"] = data_handler.get_2d_3d_ef_data("test", n_test_samples)

    for set_name in datasets:
        logger.info(f"Plotting 2D vs 3D EF plot on {set_name} dataset")
        efs_2d, efs_3d = datasets[set_name]
        efs_2d = np.reshape(efs_2d, -1) * 100.0
        efs_3d = np.reshape(efs_3d, -1) * 100.0

        # 2D vs 3D ejection fraction plot
        ef_dir = experiment_path / "plots" / "ejection_fraction_datasets" / set_name
        ef_dir.mkdir(parents=True, exist_ok=True)
        plt.clf()
        filename = "2Dvs3DEjectionFractionPlot.png"
        plt.plot(efs_2d, efs_3d, 'bo')
        plt.title(f'2D vs 3D Ejection Fraction')
        min_2d_ef = min(efs_2d)
        max_2d_ef = max(efs_2d)
        min_3d_ef = min(efs_3d)
        max_3d_ef = max(efs_3d)
        plt.xlabel(f'2D Ejection Fraction\nmin val: {min_2d_ef:.5f}, max val: {max_2d_ef:.5f}')
        plt.ylabel(f'3D Ejection Fraction\nmin val: {min_3d_ef:.5f}, max val: {max_3d_ef:.5f}')
        plt.savefig(ef_dir / filename, bbox_inches='tight')
        logger.info("Done plotting.")

    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_test, y_test = datasets["test"]

    logger.info(f"X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    logger.info(f"X_val shape {X_val.shape}, y_val shape {y_val.shape}")
    logger.info(f"X_test shape {X_test.shape}, y_test shape {y_test.shape}")

    # normalize
    # min_val = min([X_train.min(), X_val.min(), X_test.min()])
    # max_val = max([X_train.max(), X_val.max(), X_test.max()])
    # scale = max_val - min_val

    # X_train = (X_train - min_val) / scale
    # X_val = (X_val - min_val) / scale
    # X_test = (X_test - min_val) / scale

    # scale
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)

    datasets["train"] = X_train, y_train
    datasets["val"] = X_val, y_val
    datasets["test"] = X_test, y_test

    # dataset for plotting efs on train set
    datasets["train_plotting"] = X_train, y_train

    # convert numpy data to tf dataset
    for name in datasets:
        tf_dataset = tf.data.Dataset.from_tensor_slices(datasets[name])
        tf_dataset = tf_dataset.batch(batch_size_efs)
        if name == "train":  # repeat train dataset
            tf_dataset = tf_dataset.repeat()

        datasets[name] = tf_dataset

    # delete mesh ae model
    keras.backend.clear_session()
    del mesh_ae

    # Create the Mesh Predictor model
    ef_conv = EF_converter(
        log_dir=experiment_path,
        training_params=ef_conv_configs['training'],
        model_params=ef_conv_configs['model']
    )

    # Fit Model
    logger.info("Fitting ef converter model.")
    ef_conv.fit(datasets["train"], datasets["train_plotting"], datasets["val"], datasets["test"])

    logger.info("DONE!")


def main():

    project_dir = Path(__file__).absolute().parent  # absolute path to this python file

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('config_dir',
                        type=lambda p: Path(p).absolute(),
                        help='Path to the folder containing the three configs')
    parser.add_argument('--leomed', action='store_true')
    parser.add_argument('--run_exps', action='store_true')
    parser.add_argument('--data_dir',
                        default=project_dir / 'heart_mesh' / 'shape_models',
                        type=lambda p: Path(p).absolute(),
                        help='specify the folder containing the shape model')
    parser.add_argument('--echo_dir',
                        default="/media/fabian/Extreme SSD/PhD/DATA/ECHO/",
                        type=lambda p: Path(p).absolute(),
                        help='specify the folder containing echo data.')
    parser.add_argument('--results_dir',
                        default=project_dir / 'experiments',
                        type=lambda p: Path(p).absolute(),
                        help='specify the folder where the results get saved')
    parser.add_argument('--mode', default='train', type=str,
                        help='train or test')
    parser.add_argument('--gpu_name', type=str, default='fabian',
                        help='specify part of socket name if run locally')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed (default: 2)')
    parser.add_argument('--viz', type=int, default=0,
                        help='visualize while test')
    parser.add_argument('--continue_exp',
                        type=str,
                        help='experiment folder to continue training')
    args = parser.parse_args()

    # only run DeepHeartBeat mesh AE
    if args.mode == 'dhb_mae':
        # load config file
        with (args.config_dir / "dhb.yml").open(mode='r') as yamlfile:
            dhb_shape_configs = yaml.safe_load(yamlfile)
        run_dhb_shape_ae(args, dhb_shape_configs)
    # only run shape ae
    elif args.mode == 'dhb_eae':
        run_dhb_echo(args)

    elif args.mode == 'mae':
        # load config file
        with (args.config_dir / "shape.yml").open(mode='r') as yamlfile:
            shape_configs = yaml.safe_load(yamlfile)
        run_shape_ae(args, shape_configs)
    # only run echo ae
    elif args.mode == 'eae':
        # load config file
        with (args.config_dir / "echo.yml").open(mode='r') as yamlfile:
            echo_configs = yaml.safe_load(yamlfile)
        run_echo_ae(args, echo_configs)
    # only run cycle gan
    elif args.mode == 'gm':
        # load config file
        with (args.config_dir / "cycle.yml").open(mode='r') as yamlfile:
            cycle_configs = yaml.safe_load(yamlfile)

        if RUN_EXPS:
            loaded_model_dir = cycle_configs['loaded_models']['gm']['experiment']
            if loaded_model_dir is None:
                loaded_model_dir = sorted(list(filter(lambda d: 'experiment' not in d, os.listdir(args.results_dir / "GenModel"))))[-1]
            cycle_config_file = args.results_dir / "GenModel" / loaded_model_dir / "configs.yml"
            with cycle_config_file.open(mode='r') as yamlfile:
                cycle_configs = yaml.safe_load(yamlfile)
        else:
            loaded_model_dir = None
        run_cycle_gan(args, cycle_configs, run_exps=RUN_EXPS, loaded_model_dir=loaded_model_dir)
    # only run cycle gan
    elif args.mode == 'gm_10folds':
        # load config file
        with (args.config_dir / "cycle.yml").open(mode='r') as yamlfile:
            cycle_configs = yaml.safe_load(yamlfile)
        run_cycle_gan_10_fold(args, cycle_configs)
    # only run mesh ef predictor
    elif args.mode == 'mesh_ef_pred':
        # load config file
        with (args.config_dir / "mesh_ef.yml").open(mode='r') as yamlfile:
            mesh_ef_configs = yaml.safe_load(yamlfile)
        run_mesh_ef_pred(args, mesh_ef_configs)
    # only run echo ef predictor
    elif args.mode == 'echo_ef_pred':
        # load config file
        with (args.config_dir / "echo_ef.yml").open(mode='r') as yamlfile:
            echo_ef_configs = yaml.safe_load(yamlfile)
        run_echo_ef_pred(args, echo_ef_configs)
    elif args.mode == 'echo_to_mesh':
        # load config file
        with (args.config_dir / "echo_to_mesh.yml").open(mode='r') as yamlfile:
            echo_to_mesh_configs = yaml.safe_load(yamlfile)
        run_echo_to_mesh(args, echo_to_mesh_configs)
    else:
        raise Exception(f"Flag:{args.mode} is not valid.")


if __name__ == "__main__":

    # open virtual display then launch main()
    if LEOMED_RUN:
        with Xvfb() as xvfb:
            main()
    else:
        main()
