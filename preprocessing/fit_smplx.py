import sys
sys.path.append("../")
sys.path.append("./")

import argparse
import glob
import os
import json
from time import time
from tqdm import tqdm

import torch

import numpy as np
import open3d as o3d

import lib.smplx as smplx

from lib.O3D_NB_Vis import o3d_nb_vis
from lib.human_fitting.fitting import *
from lib.human_fitting.trg_kp_load import get_keypts, smplx_dvm_sparse

import config as cfg

parser = argparse.ArgumentParser()


parser.add_argument("--target_subj", default='hyomin_example')
parser.add_argument("--target_gender", default='male')
parser.add_argument('--flathand', default=False)
parser.add_argument("--useRGB", default=True)


args = parser.parse_args()


def save_params(file_name, input_params):
    (input_body_pose, input_betas, input_global_orient,
     input_transl, input_scale, 
     input_left_hand_pose, input_right_hand_pose) = input_params

    np.savez(file_name,
             J = None,
             body_pose=input_body_pose.detach().cpu().numpy(),
             left_hand_pose=input_left_hand_pose.detach().cpu().numpy(),
             right_hand_pose=input_right_hand_pose.detach().cpu().numpy(),
             betas=input_betas.detach().cpu().numpy(),
             global_orient=input_global_orient.detach().cpu().numpy(),
             transl=input_transl.detach().cpu().numpy(),
             scale=input_scale.detach().cpu().numpy())


def main(model, target_dir, device, dtype, use_RGBD = True, input_params=None):
    cfg.rootLogger.debug("Read inputs")

    pcd_dir = os.path.join(target_dir, "train/pcd")
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.ply")))
    op_kp_dir = os.path.join(target_dir, "train/keypoints")
    op_kp_files = sorted(glob.glob(os.path.join(op_kp_dir, "*.json")))

    recon_frames = len(pcd_files)

    priors = {}
    priors["use_RGBD"] = use_RGBD
    if use_RGBD:
        intrinsic = json.load(open(os.path.join(target_dir, "train/intrinsics.json"), 'r'))['color']
        priors["cv2gl"] = torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=dtype, device=device)
        
        priors["intrinsic"] = intrinsic

        body_keypts_list = []
        face_keypts_list = []
        hand_left_keypts_list = []
        hand_right_keypts_list = []
        for path in op_kp_files:
            body_keypts_list.append(get_keypts(path, 'pose_keypoints_2d', intrinsic)[np.newaxis, :, :])
            face_keypts_list.append(get_keypts(path, 'face_keypoints_2d', intrinsic)[np.newaxis, :, :])
            hand_left_keypts_list.append(get_keypts(path, 'hand_left_keypoints_2d', intrinsic)[np.newaxis, :, :])
            hand_right_keypts_list.append(get_keypts(path, 'hand_right_keypoints_2d', intrinsic)[np.newaxis, :, :])
        body_keypts = torch.tensor(np.vstack(body_keypts_list), dtype=dtype, device=device)
        face_keypts = torch.tensor(np.vstack(face_keypts_list), dtype=dtype, device=device)
        hand_left_keypts = torch.tensor(np.vstack(hand_left_keypts_list), dtype=dtype, device=device)
        hand_right_keypts = torch.tensor(np.vstack(hand_right_keypts_list), dtype=dtype, device=device)
        hand_keypts = torch.tensor(np.concatenate((np.vstack(hand_left_keypts_list), np.vstack(hand_right_keypts_list)), axis=1), dtype=dtype, device=device)
        face_valid = torch.tensor(1 - np.max(np.vstack(face_keypts_list)[:, :, 2], axis=1), dtype=dtype, device=device).unsqueeze(-1)

        priors["body_keypts"] = body_keypts
        priors["face_keypts"] = face_keypts
        priors["face_valid"] = face_valid
        priors["hand_left_keypts"] = hand_left_keypts
        priors["hand_right_keypts"] = hand_right_keypts
        priors["hand_keypts"] = hand_keypts

    priors["smplx_dvm_sparse"] = smplx_dvm_sparse.to(device)
    priors["tar_dvm_sparse"] = torch.tensor(np.load(os.path.join(target_dir, "train/dvm_sparse_markers.npy")), dtype=dtype, device=device)


    pcd_list_o3d = []
    coarse_pcd_list = []
    max_p_len = 0
    for pcd_name in pcd_files:
        tmp_pcd = o3d.io.read_point_cloud(pcd_name)
        pcd_list_o3d.append(tmp_pcd)
        tmp_uni_down_pcd = tmp_pcd.voxel_down_sample(voxel_size=0.03)
        new_p = np.asarray(tmp_uni_down_pcd.points)
        coarse_pcd_list.append(new_p)
        if max_p_len < len(new_p):
            max_p_len = len(new_p)

    tar_p_gpu = torch.zeros((recon_frames, max_p_len, 3)).to(device)
    tar_p_len_gpu = torch.zeros(recon_frames).to(device)
    mask_p_gpu = torch.zeros((recon_frames, max_p_len)).to(device)
    for i in range(recon_frames):
        tar_p_gpu[i, :len(coarse_pcd_list[i])] = torch.from_numpy(coarse_pcd_list[i]).to(device)
        tar_p_len_gpu[i] = len(coarse_pcd_list[i])
        mask_p_gpu[i, :len(coarse_pcd_list[i])] = 1

    priors["pcd_torch"] = tar_p_gpu
    priors["pcd_len_torch"] = tar_p_len_gpu.long()
    priors["mask_torch"] = mask_p_gpu
    priors["pcd_list_o3d"] = pcd_list_o3d

    # Params
    if input_params is None:
        input_body_pose = torch.zeros([recon_frames, 63],
                                      dtype=dtype,
                                      device=device,
                                      requires_grad=True)
        input_left_hand_pose = torch.zeros([recon_frames, 45],
                                      dtype=dtype,
                                      device=device,
                                      requires_grad=True)
        input_right_hand_pose = torch.zeros([recon_frames, 45],
                                      dtype=dtype,
                                      device=device,
                                      requires_grad=True)
        input_betas = torch.zeros([1, model.num_betas],
                                  dtype=dtype,
                                  device=device,
                                  requires_grad=True)
        input_global_orient = torch.zeros([recon_frames, 3],
                                          dtype=dtype,
                                          device=device,
                                          requires_grad=True)
        input_transl = torch.zeros([recon_frames, 1, 3],
                                dtype=dtype,
                                device=device,
                                requires_grad=True)
        input_scale = torch.ones([1, 1, 1], dtype=dtype, device=device, requires_grad=True)

        if use_RGBD:
            with torch.no_grad():
                input_transl[:, :, 2] -= 1

        input_params = [
            input_body_pose,
            input_betas,
            input_global_orient,
            input_transl,
            input_scale,
            input_left_hand_pose,
            input_right_hand_pose
        ]
    else:
        input_betas = input_params[1]

    closure_max_iters = 10
    ftol = 1e-6
    gtol = 1e-6
    s_time = time()

    priors["temp_reg_weight"] = 0.1
    priors["glo_temp_reg_weight"] = 0.0
    priors["pose_reg_weight"] = 0.01
    priors["P2P_weight"] = 0
    priors["OP_weight"] = 0
    priors["DVM_weight"] = 1

    cfg.rootLogger.debug('step0: start rigid fitting ...')
    step = 'orient+transl'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, priors, step, cfg.VISUALIZE)
    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)
    cfg.rootLogger.debug('step0: finsihed ...\n')

    closure_max_iters = 10

    priors["temp_reg_weight"] = 10
    priors["glo_temp_reg_weight"] = 0.001

    cfg.rootLogger.debug('step_global_pose: start fitting global pose ...')
    step = 'global_pose'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, priors, step, cfg.VISUALIZE)
    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)
    cfg.rootLogger.debug('step_global_pose1: finished ...\n')

    closure_max_iters = 10
    priors["temp_reg_weight"] = 1
    priors["glo_temp_reg_weight"] = 0.005
    priors["pose_reg_weight"] = 0.01
    priors["DVM_weight"] = 1
    priors["OP_weight"] = 0
    priors["P2P_weight"] = 10

    cfg.rootLogger.debug('step_global_pose: start fitting global pose ...')
    step = 'global_pose_OP'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, priors, step, cfg.VISUALIZE)
    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)
    cfg.rootLogger.debug('step_global_pose3: finished ...\n')


    # closure_max_iters = 30
    # priors["temp_reg_weight"] = 0
    # priors["glo_temp_reg_weight"] = 0.0
    # priors["pose_reg_weight"] = 0.0
    closure_max_iters = 10
    priors["temp_reg_weight"] = 1
    priors["glo_temp_reg_weight"] = 0.01
    # priors["temp_reg_weight"] = 0.1
    # priors["glo_temp_reg_weight"] = 0.01
    priors["pose_reg_weight"] = 0.01
    priors["OP_weight"] = 0
    priors["DVM_weight"] = 0
    priors["P2P_weight"] = 1
    cfg.rootLogger.debug('step_global_pose: start fitting global pose ...')
    step = 'global_pose'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, priors, step, cfg.VISUALIZE)
    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)
    cfg.rootLogger.debug('step_global_pose3: finished ...\n')

    e_time = time()
    cfg.rootLogger.info(f'Fitting time: {e_time - s_time:3f}s')

    if cfg.VISUALIZE:
        print("========================= complete ===========================")
        print("=========================   view   ===========================")

        with torch.no_grad():
            (input_body_pose, input_betas_1, input_global_orient,
            input_transl, input_scale_1, 
            input_left_hand_pose, input_right_hand_pose) = input_params
            input_betas_tmp = torch.ones((recon_frames, input_betas_1.shape[-1]), dtype = input_betas_1.dtype, device=input_betas_1.device)
            input_betas = input_betas_tmp * input_betas_1
            input_scale_tmp = torch.ones((recon_frames, 1, 1), dtype = input_scale_1.dtype, device=input_scale_1.device)
            input_scale = input_scale_tmp * input_scale_1

            body_model_output = model(body_pose=input_body_pose,
                                        betas=input_betas,
                                        global_orient=input_global_orient,
                                        left_hand_pose=input_left_hand_pose,
                                        right_hand_pose=input_right_hand_pose,
                                        return_verts=True)
            verts = (body_model_output.vertices * input_scale + input_transl).cpu().numpy()

        o3d_nb_vis({"Mesh0" : {"vertices":verts, "triangles": model.faces},
                    "O3D_PCD0" : {"pcd":pcd_list_o3d}
                    })

    save_params(os.path.join(target_dir, "train/smplx_fit.npz"), input_params)


    print("=========================   save   ===========================")
    # smpl_dir = os.path.join(target_dir, "train/smplx")

    # os.makedirs(smpl_dir, exist_ok=True)

    # tmp_mesh = o3d.geometry.TriangleMesh()
    # for i, pcdf in tqdm(enumerate(pcd_files), total=len(pcd_files), desc="Save body mesh"):
    #     basename = os.path.basename(pcdf)
    #     filename_smpl = os.path.join(smpl_dir, os.path.splitext(basename)[0] + ".ply")

    #     tmp_mesh.vertices = o3d.utility.Vector3dVector(verts[i])
    #     tmp_mesh.triangles = o3d.utility.Vector3iVector(model.faces)
    #     tmp_mesh.compute_vertex_normals()
    #     o3d.io.write_triangle_mesh(filename_smpl, tmp_mesh)


    return input_params


if __name__=='__main__':
    cfg.make_dir_structure(args.target_subj)
    cfg.set_log_file(os.path.join(cfg.DataPath["Main"], "logs", args.target_subj, os.path.splitext(os.path.basename(__file__))[0]))
    cfg.rootLogger.info("Start fitting smplx")

    if cfg.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float32

    target_dir = os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj)


    pcd_dir = os.path.join(target_dir, "train/pcd")
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.ply")))

    gender = args.target_gender

    model_path = os.path.join(cfg.DataPath["Main"], (cfg.DataPath["model_path_male"] if gender=='male' else cfg.DataPath["model_path_female"] ))
    model_params = dict(model_path=model_path,
                        model_type='smplx',
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
                        batch_size = len(pcd_files), # number of frames
                        use_pca=False,
                        dtype=dtype,
                        )
    
    model = smplx.create(gender=gender, **model_params)
    model.to(device)

    input_params = main(model, target_dir, device, dtype, args.useRGB)

