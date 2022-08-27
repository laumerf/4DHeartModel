# project-wide constants:
ROOT_LOGGER_STR = "HeartShapeProject"
LOGGER_RESULT_FILE = "logs.txt"
LEOMED_PROJECT_DIR = "/cluster/dataset/shfn/laumerf/projects/4dheart"
LEOMED_DATA_DIR = "/cluster/dataset/shfn/laumerf/data"

# map each matrix to the list of Conv Layers that use it
# e,g adjacency matrix is used by GraphSageConv, AGNNConv, TAGConv ...
# also modified laplacian matrix is used by GraphConv and APPNP
# etc....
# Note that each layer only needs 1 matrix type to work, e,g APPNP is
# mapped only to "modified_laplacian" and not other matrices
CONV_INPUTS = {"modified_laplacian": ["GCNConv", "APPNPConv", "DiffusionConv"], # DiffusionConv.preprocess() = gcn_filter() - see code of spektral v 1.0.6 - thus DiffusionConv here
               "normalized_adjacency": ["GCSConv"],
               "normalized_rescaled_laplacian": ["ARMAConv"],
               "chebyshev_polynomials": ["ChebConv"],
               "adjacency": ["GraphSageConv", "GATConv", "GINConv",
                             "GatedGraphConv", "AGNNConv", "TAGConv",
                             "EdgeConv", "GeneralConv"]}

UK_FOLDER = "UK_Digital_Heart"
CISTIB_FOLDER = "CISTIB"
CISTIB_ALL_FOLDER = "{}_AllPhases"
CISTIB_EPIENDO = {"art": 0, "epi": 1, "endo": 2}

# CISTIB
CISTIB_4CH_PLANE = {'origin': [-0.096, 0.176, -0.316],
                    'normal': [0.290, -0.282, 0.9143]}
CISTIB_3CH_PLANE = {'origin': [0.0300, 0.274, -0.589],
                    'normal': [0.609, 0.321, 0.724]}
CISTIB_2CH_PLANE = {'origin': [0.164, 0.356, -0.340],
                    'normal': [0.764, 0.645, -0.008]}

# CONRAD
CONRAD_2CH_PLANE = {'origin': [2.21, 18.26, 2.705],  # 2CH
                    'normal': [-0.58, -0.802, 0.11]}

#### IMPORTANT ####
# the synthetic 3-chamber view is horizontally flipped with respect to the ECHO recording 3ch view!
# keep that in mind when calculating the segmentation loss between real and predicted maps.
#### IMPORTANT ####

CONRAD_3CH_PLANE = {'origin': [0.97, 19.78, 3.19],  # 3CH
                    'normal': [0.590, 0.319, 0.740]}

CONRAD_4CH_PLANE = {'origin': [6.56, 20.29, 7.75],  # 4CH
                    'normal': [0.10, -0.46, 0.88]}

# TODO: verify
CONRAD_5CH_PLANE = {'origin': [-0.6, 18.0, 4.5],  # 5CH
                    'normal': [0.42, -0.42, 0.80]}

CONRAD_PSAX_PLANE = {'origin': [3.40, -9.92, -0.24],
                     # PSAX (attempt, unclear)
                     'normal': [-0.80, 0.32, 0.50]}

CONRAD_PLAX_PLANE = {'origin': [2.97, 7.86, -2.86],  # PLAX
                     'normal': [-0.63, -0.63, -0.45]}

CONRAD_RV_PLANE = {'origin': [0.0, 0.0, 0.0],  # dummy random view!
                   'normal': [1.0, 0.0, 0.0]}

CONRAD_VIEWS = {'2ch': CONRAD_2CH_PLANE,
                '3ch': CONRAD_3CH_PLANE,
                '4ch': CONRAD_4CH_PLANE,
                '5ch': CONRAD_5CH_PLANE,
                'psax': CONRAD_PSAX_PLANE,
                'plax': CONRAD_PLAX_PLANE,
                'rv': CONRAD_RV_PLANE}


# RGB values to color each chamber in the segmentation map
white = [255, 255, 255]
yellow = [253, 231, 36]
light_green = [121, 209, 81]
dark_green = [34, 167, 132]
light_blue = [41, 120, 142]
dark_blue = [64, 67, 135]
violet = [68, 1, 84]
CONRAD_SEG_MAP_COLORS = {0 : white, # "aorta"
                         1 : dark_blue, # "leftAtrium"
                         2 : light_blue, # "leftVentricle"
                         3 : dark_green, # "myocardium"
                         4 : light_green, # "rightAtrium"
                         5 : yellow, # "rightVentricle"
                         6 : violet # background
                        }

CONRAD_HEART_COLORS = {
        "aorta": (71, 71, 219),
        "leftAtrium": (0, 102, 157),
        "leftVentricle": (0, 154, 51),
        "myocardium": (255, 223, 0),
        "rightAtrium": (166, 39, 0),
        "rightVentricle": (224, 77, 77),
        "background": (82, 87, 110)
    }

CONRAD_SEG_MAPS_SIZE = (500, 500)
CONRAD_SEG_MAPS_PARALLEL_ZOOM = 90  # larger makes images smaller

CONRAD_FOLDER = "CardiacModel"
CONRAD_VTK_FOLDER = "vtkPolys"
CONRAD_MEAN_FILE = "mean_phase_{}_{}.{}"
CONRAD_PARAM_FILE = "param_model.{}"
CONRAD_NUM_PHASE = 10
CONRAD_COMPONENTS = {"aorta", "leftAtrium", "leftVentricle", "myocardium",
                     "rightAtrium", "rightVentricle"}
CONRAD_COMPONENTS_TO_IDS = {"aorta": 0, "leftAtrium": 1, "leftVentricle": 2, "myocardium": 3,
                            "rightAtrium": 4, "rightVentricle": 5}
CONRAD_IDS_TO_COMPONENTS = {v: k for k, v in CONRAD_COMPONENTS_TO_IDS.items()}
CONRAD_DATA_PARAMS = ["nb_cycles", "time_per_cycle", "shapes_per_cycle", "cycle_shift", "time_shift", "frequency", "EF_Vol", "EF_Biplane"]
ECHO_DATA_PARAMS = ["EF","ESV","EDV","FrameHeight","FrameWidth","FPS","NumberOfFrames"]
