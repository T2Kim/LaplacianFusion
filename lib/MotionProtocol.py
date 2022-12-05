from config import DataPath
import glob
import pickle
import numpy as np
import os

def AIST2LapF(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    aaa = np.arange(0, len(data['smpl_poses']), 5)

    res = {}
    res["pose_params"] = data['smpl_poses'][aaa]
    res["global_trans"] = data['smpl_trans'][aaa] / data['smpl_scaling']
    return res


class MotionProtocol():
    def __init__(self, ds_name = "LapF", filename = ""):
        assert ds_name in ["LapF", "AIST++", "AIST++name", "CAPE"], 'Unknown motion dataset type'
        self.ds_name = ds_name
        
        if ds_name == "LapF":
            assert filename != ""
            self.smpl_fit = {}
            packed = np.load(filename)
            pose = np.concatenate((packed["global_orient"], packed["body_pose"]),axis=-1)
            self.smpl_fit["pose_params"] = pose
            self.smpl_fit["global_trans"] = np.squeeze(packed["transl"])

        if ds_name == "AIST++":
            self.filenames = glob.glob(DataPath["motion"]["AIST++"])
            self._invert_param = AIST2LapF


        if ds_name == "AIST++name":
            self.foldername = DataPath["motion"]["AIST++"][:-6]
            self.filenames = glob.glob(DataPath["motion"]["AIST++"])
            self._invert_param = AIST2LapF

    def get_params(self, idx = 0, motionname = None):
        if self.ds_name == "AIST++":
            return self._invert_param(self.filenames[idx])


        if self.ds_name == "AIST++name":
            idx = self.filenames.index(os.path.join(self.foldername, motionname + ".pkl"))
            return self._invert_param(self.filenames[idx])

        if self.ds_name == "LapF":
            return self.smpl_fit
        
        return None

