import numpy as np
import os
import sys
import logging
import datetime

DataPath = {
    "Main": "/media/hyomin/HDD6/DATA/lapfu",

    "Anchor": "protocol_info/shead_protocol.npz",
    # "Anchor_smpl": "protocol_info/shead_protocol_smpl.npz",

    "model_path_male": "human_models/smplx/SMPLX_MALE.npz",
    "model_path_female": "human_models/smplx/SMPLX_FEMALE.npz",

    # "model_path_male_smpl": "human_models/smpl/SMPL_MALE.pkl",
    # "model_path_female_smpl": "human_models/smpl/SMPL_FEMALE.pkl",
    
    "motion":
        {"AIST++": "/Kiwi/Data1/Dataset/Human_Motions/AIST++/*.pkl"
        }
}
VISUALIZE = True
DEBUG = True

def make_dir_structure(subject_name):
    new_dirs = []
    new_dirs.append(os.path.join(DataPath["Main"], "logs", subject_name))
    new_dirs.append(os.path.join(DataPath["Main"], "subjects", subject_name, "net"))
    new_dirs.append(os.path.join(DataPath["Main"], "subjects", subject_name, "train"))

    for d in new_dirs:
        os.makedirs(d, exist_ok=True)

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
if DEBUG:
    rootLogger.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

def set_log_file(filename):
    global rootLogger
    now = datetime.datetime.now()
    filename += '_' + now.strftime('%Y%m%d_%H%M%S') + '.log'
    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)


### Method Parameters

rest_rot = np.zeros((1, 72), dtype=np.float32)
rest_rot[0, 5] = np.pi / 6
rest_rot[0, 8] = -np.pi / 6
## star-pose
# rest_rot[0, 16 * 3 + 2] = -np.pi / 9
# rest_rot[0, 17 * 3 + 2] = np.pi / 9
# rest_rot[0, 18 * 3 + 1] = -np.pi / 6
# rest_rot[0, 19 * 3 + 1] = np.pi / 6

# memory issue
infer_frame_max = 5
lap_infer_frame_max = 5

# learning
lap_learning_rate = 3e-4
smpl_learning_rate = 1e-2
residual_learning_rate = 1e-4

num_worker_threads = 0

shape_batch_size_lap = 5000
shape_batch_size_residual = 5

# epochs_lap = 1000
# epochs_residual = 3000
epochs_lap = 50
epochs_residual = 100

shuffle = True
is_cuda = True