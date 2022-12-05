import sys
sys.path.append("../")
sys.path.append("./")
import os
import glob
import numpy as np

import argparse
import scipy as sp
from scipy.spatial.transform import Rotation as R

import open3d as o3d
from sklearn.neighbors import KDTree
import torch

from tqdm import tqdm

import config as cfg
import lib.smplx as smplx
from lib.utils import load_params

parser = argparse.ArgumentParser()
parser.add_argument("--target_subj", default='hyomin_example')
parser.add_argument("--target_gender", default='male')
parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
parser.add_argument('--flathand', default=False)


args = parser.parse_args()

if __name__ == '__main__':
    cfg.make_dir_structure(args.target_subj)
    target_dir = os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj)
    
    if cfg.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float32

    pcd_dir = os.path.join(target_dir, "train/pcd")
    warp_dir = os.path.join(target_dir, "train/coarse")
    delta_dir = os.path.join(target_dir, "train/delta")
    filename_mapped_all = os.path.join(target_dir, "train/mapped_delta.npy")

    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*")))
    warp_files = sorted(glob.glob(os.path.join(warp_dir, "*")))
    delta_files = sorted(glob.glob(os.path.join(delta_dir, "*")))

    anchor = np.load(os.path.join(cfg.DataPath["Main"], cfg.DataPath["Anchor"]))

    recon_frames = len(pcd_files)
    pcd_list = []
    max_p_len = 0

    model_path = os.path.join(cfg.DataPath["Main"], (cfg.DataPath["model_path_male"] if args.target_gender=='male' else cfg.DataPath["model_path_female"] ))
    model_params = dict(model_path=model_path,
                        model_type='smplx',
                        #joint_mapper=joint_mapper,
                        create_global_orient=False,
                        create_body_pose=False,
                        create_betas=False,
                        create_left_hand_pose=False,
                        create_right_hand_pose=False,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        flat_hand_mean=True,
                        use_pca=False,
                        dtype=dtype,
                        )
    body_model = smplx.create(gender=args.target_gender, **model_params)
    body_model.to(device)
    expression=torch.zeros([1, 10], dtype=dtype, device=device)
    jaw_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    leye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    reye_pose=torch.zeros([1, 3], dtype=dtype, device=device)

    params = load_params(os.path.join(target_dir, "train/smplx_fit.npz"), device, dtype)
    (input_body_pose, input_betas_1, input_global_orient,
        input_transl, input_scale_1, 
        input_left_hand_pose, input_right_hand_pose) = params

    custom_lbs_weight = body_model.lbs_weights[anchor["smplx2shead"]].cpu().numpy()

    # triangle index, frame index, barycentric coordinates, rotated laplacian coordinate
    results = np.empty((0, 9))

    frame_count = -1
    for pcdf, warpf, deltaf in tqdm(zip(pcd_files, warp_files, delta_files), total=recon_frames, desc="Laplacian projection"):
        frame_count += 1
        basename = os.path.basename(pcdf)

        tmp_pcd = o3d.io.read_point_cloud(pcdf)
        tmp_coarse_mesh = o3d.io.read_triangle_mesh(warpf)
        v = np.asarray(tmp_coarse_mesh.vertices)
        f = np.asarray(tmp_coarse_mesh.triangles)
        delta_all = np.load(deltaf)
        delta_all = delta_all.astype(np.float32)

        f_pos = np.sum(v[f], axis=1) / 3

        p0 = v[f[:, 0]]
        p1 = v[f[:, 1]]
        p2 = v[f[:, 2]]
        p01 = v[f[:, 1]] - v[f[:, 0]]
        p02 = v[f[:, 2]] - v[f[:, 0]]
        normals = np.cross(p01, p02)

        v_sub = np.asarray(tmp_pcd.points)

        tree = KDTree(f_pos)
        nearest_dist, nearest_ind = tree.query(v_sub, k=9)

        in_tri = np.zeros(len(v_sub), dtype=bool)
        sel_tri_idx = -np.ones(len(v_sub), dtype=np.int64)
        bary_coords = np.zeros((len(v_sub), 3))

        for i in range(9):
            near_idx = nearest_ind[:, i]
            P = v_sub
            P0 = p0[near_idx]
            P01 = p01[near_idx]
            P02 = p02[near_idx]
            N = normals[near_idx]
            w = P - P0

            NN = np.sum(N * N, axis = 1)
            NN = np.where(NN == 0, 100, NN)

            gamma = np.sum(np.cross(P01, w) * N, axis=1) / NN
            beta = np.sum(np.cross(w, P02) * N, axis=1) / NN
            alpha = np.ones_like(gamma) - gamma - beta

            gamma_in = np.logical_and(gamma >= 0, gamma <= 1)
            beta_in = np.logical_and(beta >= 0, beta <= 1)
            alpha_in = np.logical_and(alpha >= 0, alpha <= 1)

            tmp_in_tri = np.logical_and(np.logical_and(gamma_in, beta_in), alpha_in)

            new_assign = np.logical_and(np.logical_not(in_tri), tmp_in_tri)
            sel_tri_idx = np.where(new_assign, near_idx, sel_tri_idx)

            bary_coords[:, 0] = np.where(new_assign, alpha, bary_coords[:, 0])
            bary_coords[:, 1] = np.where(new_assign, beta, bary_coords[:, 1])
            bary_coords[:, 2] = np.where(new_assign, gamma, bary_coords[:, 2])
            
            in_tri = np.logical_or(in_tri, tmp_in_tri)

        p_indices = np.arange(len(v_sub))[in_tri]
        sub_delta = delta_all[in_tri]
        valid_sel_tri = sel_tri_idx[in_tri][:, np.newaxis]

        # unwarp delta
        tmp_skin = torch.tensor(np.sum(custom_lbs_weight[anchor["shead_tri"][sel_tri_idx[in_tri]]] * bary_coords[p_indices, :, np.newaxis], axis = 1), dtype=dtype, device=device)

        # delta_new = body_model.LBS_deform(tmp_skin, torch.tensor(sub_delta[np.newaxis, :, :], dtype=dtype, device=device),
        #                         only_rotation = True, inverse = True,
        #                         body_pose=input_body_pose[frame_count].unsqueeze(0),
        #                         betas=input_betas_1,
        #                         global_orient=input_global_orient[frame_count].unsqueeze(0),
        #                         left_hand_pose=input_left_hand_pose[frame_count].unsqueeze(0),
        #                         right_hand_pose=input_right_hand_pose[frame_count].unsqueeze(0),
        #                         expression=expression,
        #                         jaw_pose=jaw_pose,
        #                         leye_pose=leye_pose,
        #                         reye_pose=reye_pose,
        #                         return_verts=True)[0].cpu().numpy()

        delta_new = sub_delta
                
        results = np.concatenate((results, np.concatenate((valid_sel_tri, np.ones_like(valid_sel_tri) * frame_count, bary_coords[p_indices], delta_new, v_sub[p_indices, 2][:, np.newaxis]), axis=1)), axis=0)

    np.save(filename_mapped_all, results)


