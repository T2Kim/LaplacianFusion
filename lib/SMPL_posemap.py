import sys
sys.path.append("../")
sys.path.append("./")

import numpy as np
import math

import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

kinematic_tree = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# SCANimate
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
def get_posemap(map_type, n_joints, parents, n_traverse=1, normalize=False, no_head = False):
    pose_map = torch.zeros(n_joints,n_joints-1)
    if map_type == 'parent':
        for i in range(n_joints-1):
            pose_map[i+1,i] = 1.0
    elif map_type == 'children':
        for i in range(n_joints-1):
            parent = parents[i+1]
            for j in range(n_traverse):
                pose_map[parent, i] += 1.0
                if parent == 0:
                    break
                parent = parents[parent]
        if normalize:
            pose_map /= pose_map.sum(0,keepdim=True)+1e-16
    elif map_type == 'both':
        for i in range(n_joints-1):
            pose_map[i+1,i] += 1.0
            parent = parents[i+1]
            for j in range(n_traverse):
                pose_map[parent, i] += 1.0
                if parent == 0:
                    break
                parent = parents[parent]
        if normalize:
            pose_map /= pose_map.sum(0,keepdim=True)+1e-16
    else:
        raise NotImplementedError('unsupported pose map type [%s]' % map_type)
    if no_head:
        pose_map[15, :] = 0
    return pose_map


def rodrigues_vec_to_rotation_mat(rodrigues_vec):
    theta = np.linalg.norm(rodrigues_vec)
    if theta < sys.float_info.epsilon:              
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rodrigues_vec / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array([
            [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
        ])
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat 

def create_joint_frame(rodrigues_vec):
    rot_mats = []
    for rot in rodrigues_vec:
        rot_mats.append(rodrigues_vec_to_rotation_mat(rot))
    
    for i, k in enumerate(kinematic_tree):
        rot_mats[i + 1] = rot_mats[k] @ rot_mats[i + 1]
    rot_mats = np.array(rot_mats)
    return rot_mats

def create_joint_T_GloRot(glo_rot, rodrigues_vec, translation):
    rot_mats = []
    trans_vecs = []
    for rot in rodrigues_vec:
        rot_mats.append(rodrigues_vec_to_rotation_mat(rot))
        # rot_mats.append(np.eye(3))

    rot_mats[0] = glo_rot @ rot_mats[0]
    
    for i, k in enumerate(kinematic_tree):
        rot_mats[i + 1] = rot_mats[k] @ rot_mats[i + 1]
    
    trans_vecs.append(translation[0])

    for i, k in enumerate(kinematic_tree):
        trans_vecs.append(trans_vecs[k] + (rot_mats[k] @ translation[i + 1].T))


    T_mats = []
    for rot, t in zip(rot_mats, trans_vecs):
        tmp_T = np.eye(4)
        tmp_T[:3, :3] = rot
        tmp_T[:3, 3] = t
        T_mats.append(tmp_T)
    
    T_mats = np.array(T_mats)
    return T_mats

def create_joint_T(rodrigues_vec, translation):
    rot_mats = []
    trans_vecs = []
    for rot in rodrigues_vec:
        rot_mats.append(rodrigues_vec_to_rotation_mat(rot))
        # rot_mats.append(np.eye(3))
    
    for i, k in enumerate(kinematic_tree):
        rot_mats[i + 1] = rot_mats[k] @ rot_mats[i + 1]
    
    trans_vecs.append(translation[0])

    for i, k in enumerate(kinematic_tree):
        trans_vecs.append(trans_vecs[k] + (rot_mats[k] @ translation[i + 1].T))


    T_mats = []
    for rot, t in zip(rot_mats, trans_vecs):
        tmp_T = np.eye(4)
        tmp_T[:3, :3] = rot
        tmp_T[:3, 3] = t
        T_mats.append(tmp_T)
    
    T_mats = np.array(T_mats)
    return T_mats




def core2all_torch(pose):
    device = pose.device
    pose_all = torch.zeros((len(pose), 72)).to(device)
    pose_all[:, :21] = pose[:, :21]
    pose_all[:, 27:30] = pose[:, 21:24]
    pose_all[:, 36:60] = pose[:, 24:]
    # pose_all[:, 39:60] = pose[:, 27:]
    return pose_all

def core2all1(pose):
    pose_all = np.zeros(72)
    pose_all[:21] = pose[:21]
    pose_all[27:30] = pose[21:24]
    pose_all[36:60] = pose[24:]
    return pose_all

def core2all(pose):
    pose_all = np.zeros((len(pose), 72))
    pose_all[:, :21] = pose[:, :21]
    pose_all[:, 27:30] = pose[:, 21:24]
    pose_all[:, 36:60] = pose[:, 24:]
    # pose_all[:, 39:60] = pose[:, 27:]
    return pose_all

class MLP_skinner(nn.Module):
    def __init__(self):
        super(MLP_skinner, self).__init__()
        self.fc = nn.Linear(111, 24)

    def forward(self, dvm):
        return self.fc(dvm)
