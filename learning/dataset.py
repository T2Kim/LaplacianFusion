import numpy as np
from torch.utils.data import Dataset
import torch

from lib.utils import load_params, get_posemap_custom
from lib.smplx.lbs import batch_rodrigues
from lib.custom_lbs import convert_global_R
from pytorch3d.transforms import matrix_to_quaternion

class MappedLapDataset(Dataset):
    def __init__(self, body_model, delta_filename, pose_filename, anchor, use_RGBD, use_noise=True, device ='cuda', dtype=torch.float32):
        self.device = device
        raw_data = np.load(delta_filename)

        params = load_params(pose_filename, device, dtype)
        (input_body_pose, input_betas_1, input_global_orient,
            input_transl, input_scale_1, 
            input_left_hand_pose, input_right_hand_pose) = params


        self.custom_lbs_weight = body_model.lbs_weights[anchor["smplx2shead"]]
        self.body_skin_weight_mask = self.custom_lbs_weight[:, :22]

        self.template_v = torch.from_numpy(anchor["shead_coord"]).float().to(device)
        self.template_f = torch.from_numpy(anchor["shead_tri"]).long().to(device)

        self.bary_weights = torch.from_numpy(raw_data[:, 2:5][:, :, np.newaxis]).float().to(device)
        self.bary_coords = torch.sum(self.template_v[self.template_f][raw_data[:, 0].astype(np.int64), :, :] * self.bary_weights, dim=1)
        self.f_idx = torch.from_numpy(raw_data[:, 0]).long().to(device)
        self.pose_code_idx = torch.from_numpy(raw_data[:, 1]).long().to(device)
        self.delta = torch.from_numpy(raw_data[:, 5:8]).float().to(device)

        self.pose_R = convert_global_R(body_model, input_body_pose, input_global_orient)
        self.num_joints = self.custom_lbs_weight.shape[1]

        self.pose_map = get_posemap_custom().float().to(device)

        mask_p_gpu = raw_data[:, 8] !=0
        # self.z_weight = torch.ones(len(raw_data[:, 8]))
        # self.z_weight[mask_p_gpu] = torch.exp(torch.ones_like(self.z_weight)[mask_p_gpu] / torch.abs(torch.from_numpy(raw_data[:, 8]).float()[mask_p_gpu]))
        # self.z_weight = self.z_weight.to(device)

        self.z_weight = torch.tensor(np.exp(-2 * np.abs(raw_data[:, 8])), dtype=dtype, device=device)
        if not use_RGBD:
            self.z_weight = torch.ones_like(self.z_weight)
            
        self.use_noise = use_noise
        
        input_body_pose_feat = input_body_pose[:, :63]
        self.pose_code = matrix_to_quaternion(batch_rodrigues(input_body_pose_feat.reshape(-1, 3))).reshape(input_body_pose.shape[0], 21, 4)
        
        self.n_points = len(raw_data)

        self.count = 0


    def __len__(self):
        return self.n_points

    def get_item(self, index):
        batch_size = len(index)
        noise_pose = self.pose_code[self.pose_code_idx[index]]
        # if self.use_noise and self.count < 8000:
        #     noise = 0.1 - 0.05 * torch.rand(noise_pose.shape[0], noise_pose.shape[1]).to(self.device)
        #     noise_axis = 0.1 - 0.5 * torch.rand(noise_pose.shape[0], noise_pose.shape[1], 3).to(self.device)
        #     noise_pose[:, :, 3] += noise
        #     noise_pose[:, :, :3] += noise_axis
        # self.count += 1
        if self.use_noise:
            noise = 0.02 - 0.01 * torch.rand(noise_pose.shape[0], noise_pose.shape[1]).to(self.device)
            noise_axis = 0.02 - 0.01 * torch.rand(noise_pose.shape[0], noise_pose.shape[1], 3).to(self.device)
            noise_pose[:, :, 3] += noise
            noise_pose[:, :, :3] += noise_axis

        tmp_bary_skin = torch.sum(self.custom_lbs_weight[self.template_f[self.f_idx[index]]] * self.bary_weights[index], dim=1)
        tmp_bary_skin_mask = torch.where(tmp_bary_skin[:, :22] > 0.3, 1., 0.)

        pose_map_point = torch.where(torch.einsum('ij, jk -> ik', tmp_bary_skin_mask, self.pose_map) > 0, 1., 0.) # n_points x 21

        # (B x 1 x (J+1)) x (B x (J+1) x 9)
        tmp_golbal_R = torch.matmul(tmp_bary_skin.unsqueeze(1), \
            self.pose_R[self.pose_code_idx[index]].view(batch_size, self.num_joints, 9)).view(batch_size, 1, 3, 3)
        
        tmp_pose_code_gpu = pose_map_point.unsqueeze(-1) * noise_pose
        return self.bary_coords[index], tmp_pose_code_gpu.reshape(-1, 84), self.delta[index], self.z_weight[index], tmp_golbal_R




