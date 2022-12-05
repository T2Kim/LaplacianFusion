import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import open3d as o3d

from models import load_model
# from lib.renderer import *
import copy

from matplotlib import cm

def pcd_sparse_tensor_fast(coords, voxel_size=0.01):
    # Create a batch, this process is done in a data loader during training in parallel.
    a, b, inv_idx = pcd_voxel_fast(coords, voxel_size)
    batch = [[a, b]]
    coordinates_, featrues_ = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)    
    # Normalize features and create a sparse tensor
    return coordinates, (features - 0.5).float(), inv_idx

def pcd_voxel_fast(coords, voxel_size):
    feats = np.zeros_like(coords, dtype=np.float32)
    quantized_coords = np.floor(coords / voxel_size)
    inds, inv_ids = ME.utils.sparse_quantize(quantized_coords, return_index=True, return_inverse=True)
    return quantized_coords[inds], feats[inds], inv_ids


class Labeler:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        # Define a model and load the weights   
        NetClass = load_model(config.model)
        self.model = NetClass(3, config.marker, config).to(self.device)
        model_dict = torch.load(config.weights)
        self.model.load_state_dict(model_dict['state_dict'])
        self.model.eval()
        self.voxel_size = config.voxel_size
        self.marker = config.marker
        self._gen_color_map()
        self.cmap_name = "default"
        self.image_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    def getLabel_points_fast2(self, pts, y_filp = True, additional_aff = np.eye(3), get_color = True):
        if y_filp:
            additional_aff = np.dot(additional_aff, np.array([[1.,0,0],[0,-1,0],[0,0,-1]]))
        sample_points = np.dot(np.asarray(pts), additional_aff)

        pred, up_coord, inv_idx = self._inference_fast(sample_points)

        pts_pred = pred[inv_idx]

        pts_color = np.ones((len(pts), 3), dtype=np.float32) * 0.5
        if get_color:
            upcoord_color = self._gen_soft_color(pred)
            pts_color = upcoord_color[inv_idx]

        mesh_p_bag = {}
        mesh_p_bag["pred"] = pts_pred
        mesh_p_bag["color"] = pts_color
        mesh_p_bag["coord"] = pts
        mesh_p_bag["up_coord"] = np.dot(up_coord, additional_aff) * self.voxel_size
        mesh_p_bag["up_color"] = upcoord_color

        mesh_p_bag["max_coords_mask"] = np.ones((self.config.marker, 4))
        up_coord_pred_max_idx = np.argsort(pred, axis=0)[-1, :]
        arr_w = list(range(self.config.marker))        
        up_coord_pred_max_coords = mesh_p_bag["up_coord"][up_coord_pred_max_idx, :]
        up_coord_pred_max_w = pred[up_coord_pred_max_idx, arr_w]
        # pred_max_coords = pred_max_coords[np.where(pred_max_w > 0.3), :].reshape((-1,3))

        mesh_p_bag["max_coords_mask"][:, :3] = up_coord_pred_max_coords
        for i, w in enumerate(up_coord_pred_max_w):
            if(w < 0.6):
                mesh_p_bag["max_coords_mask"][i, 3] = 0

        return mesh_p_bag, inv_idx, pred


    def _inference_fast(self, coords):
        with torch.no_grad():
            coordinates, features, inv_idx = pcd_sparse_tensor_fast(coords, voxel_size=self.voxel_size)
            sinput = ME.SparseTensor(features, coords=coordinates).to(self.device)
            soutput = self.model(sinput)
        pred = F.softmax(soutput.F[:, :self.marker], dim=1)
        pred = pred.cpu().numpy()
        cell = soutput.C.numpy()[:, 1:]
        return pred, cell, inv_idx

    def _gen_soft_color(self, upsampled_pred):
        v_color = np.zeros((len(upsampled_pred), 3), dtype=np.float32)

        for i in range(upsampled_pred.shape[1]):
            v_color += np.dot((upsampled_pred[:, i])[:, np.newaxis],
                            np.array(self.COLOR_MAPS[self.cmap_name][i], dtype=np.float32).reshape(1, 3))

        return v_color / 255

    def _gen_color_map(self):
        self.COLOR_MAPS = {}

        ## default
        cc_map = cm.get_cmap('Paired')
        COLOR_MAP = np.zeros((111, 3))
        part = [[],[],[],[],[],[]]
        part[0]= np.array([12,6,7,8,9,10,11,0,1,2,3,4,5, 13,14,15,16])[::-1]                              # head
        part[1]=np.concatenate((np.arange(17,25),np.arange(51,75))) # body

        part[2]= np.arange(25,38) # right arm
        part[3]= np.arange(38,51) # left arm 

        part[4]= np.arange(75,93) # right leg
        part[5]= np.arange(93,111) # left leg

        n_part = 6
        for i, p in enumerate(part):
            n_seg = len(p)
            for j, m in enumerate(p):
                COLOR_MAP[m, :] = np.array([k * 255 for k in list(cc_map(i/n_part + (1/n_part) * (j/n_seg))[:3])])

        self.COLOR_MAPS["default"] = copy.deepcopy(COLOR_MAP)

        ## legacy
        COLOR_MAP = []

        palette = [[0,0,1], [0,1,0], [1,0,0], [0,1,1], [1,0,1], [1,1,0],\
            [1,0,0.5], [0,1,0.5], [1,0.5,0], [0.5, 0,1],[0,0.5,1], [0.5,1,0],   [0.7,0.7,0.7], \
            [0.7,0.3,0], [0.3, 0,0.7], [0,0.3,0.7], [0.3,0.7,0], [0.7,0,0.3], [0,0.7,0.3]]
        scale = 5
        decimator = 3
        for l in range(120):
            s = l % 2 + 1
            r = l % len(palette)
            COLOR_MAP.append((100 + s * 50) * np.array(palette[r]))

        COLOR_MAP.append([0, 0, 0])
        COLOR_MAP = np.array(COLOR_MAP)
        self.COLOR_MAPS["legacy"] = copy.deepcopy(COLOR_MAP)

