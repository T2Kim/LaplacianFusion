
import sys
sys.path.append("../")
sys.path.append("./")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "1"


import numpy as np
import open3d as o3d
import glob
import argparse

from tqdm import tqdm

import torch
import lib.smplx as smplx
from lib.smplx.utils import load_params, save_params, get_posemap_custom
from lib.smplx.lbs import batch_rodrigues
from pytorch3d.transforms import matrix_to_quaternion
from lib.model.mlp import *
import config as cfg
import igl

parser = argparse.ArgumentParser()

parser.add_argument("--target_subj", default='NeuroGIF_ShrugsPants')
parser.add_argument("--target_gender", default='female')
parser.add_argument('--RGBD', default=False, help='Is Point Cloud?')
parser.add_argument('--flathand', default=False)
parser.add_argument('--epoch', type=int, default=3000)

args = parser.parse_args()

if __name__ == '__main__':
    cfg.make_dir_structure(args.target_subj)
    cfg.rootLogger.info("Inference pose blend base mesh")
    target_dir = os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj)

    if cfg.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float32

    params = load_params(os.path.join(target_dir, "train/smplx_fit.npz"), device, dtype)
    (input_body_pose, input_betas_1, input_global_orient,
        input_transl, input_scale_1, 
        input_left_hand_pose, input_right_hand_pose) = params

    recon_frames = len(input_body_pose)
    
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

    pcd_unwarp_dir = os.path.join(target_dir, "train/pcd")
    pcds = sorted(glob.glob(os.path.join(pcd_unwarp_dir, "*.ply")))
    
    
    expression=torch.zeros([1, 10], dtype=dtype, device=device)
    jaw_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    leye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    reye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    anchor = np.load(os.path.join(cfg.DataPath["Main"], cfg.DataPath["Anchor"]))
    body_model_output = body_model(body_pose=torch.zeros_like(input_body_pose[0]).unsqueeze(0),
                                betas=input_betas_1,
                                global_orient=torch.zeros_like(input_global_orient[0]).unsqueeze(0),
                                left_hand_pose=torch.zeros_like(input_left_hand_pose[0]).unsqueeze(0),
                                right_hand_pose=torch.zeros_like(input_right_hand_pose[0]).unsqueeze(0),
                                expression=expression,
                                jaw_pose=jaw_pose,
                                leye_pose=leye_pose,
                                reye_pose=reye_pose,
                                return_verts=True)
    template_v = body_model_output.vertices[0, anchor["smplx2shead"]].cpu().numpy()
    template_f = anchor["shead_tri"]
    
    b_mesh = o3d.geometry.TriangleMesh()
    b_mesh.vertices = o3d.utility.Vector3dVector(template_v)
    b_mesh.triangles = o3d.utility.Vector3iVector(template_f)

    S_4, template_f = igl.loop_subdivision_matrix(len(template_v), template_f)
    template_v = S_4 @ template_v
    S_44, template_f = igl.loop_subdivision_matrix(S_4.shape[0], template_f)
    template_v = S_44 @ template_v
    # S_444, template_f = igl.loop_subdivision_matrix(S_44.shape[0], template_f)
    # template_v = S_444 @ template_v
    template_v = template_v.astype(np.float32)
    shead_S_x16 = S_44 @ S_4

    model_residual = MLP_Coarse_res(3, 84).to(device)
    model_residual.load_state_dict(torch.load(os.path.join(target_dir, "net/abl1_residual_model" + "_e" + str(args.epoch) + ".pts")))

    n_v = len(anchor["noahnd_x16_coord"])
    n_v_hand = len(anchor["shead_x16_coord"])

    custom_lbs_weight = torch.from_numpy(shead_S_x16 @ body_model.lbs_weights[anchor["smplx2shead"]].cpu().numpy()).float().to(device)

    manifold_coord_gpu = torch.from_numpy(anchor["noahnd_x16_coord"]).float().to(device).unsqueeze(0)
    body_skin_weight = custom_lbs_weight.clone()[anchor["shead2nohand_x16"]][:, :22]
    body_skin_weight = torch.where(body_skin_weight > 0.3, 1., 0.)
    pose_map = get_posemap_custom().float().to(device) # 22 x 21
    pose_map_vert = torch.where(torch.einsum('ij, jk -> ik', body_skin_weight, pose_map) > 0, 1., 0.) # n_verts x 21

    
    pose_code_gpu = matrix_to_quaternion(batch_rodrigues(input_body_pose.reshape(-1, 3))).reshape(recon_frames, 21, 4)

    pose_code_gpu = pose_code_gpu.unsqueeze(1).repeat(1, n_v, 1, 1)
    pose_code_gpu = pose_map_vert.unsqueeze(0).unsqueeze(-1) * pose_code_gpu
    pose_code_gpu = pose_code_gpu.reshape(recon_frames, -1, 84)

    pos_enc_input = manifold_coord_gpu.repeat(recon_frames, 1, 1).reshape(-1, 3)
    pose_code_input = pose_code_gpu.reshape(-1, 84)

    # Inference
    v_new_rest_vec = np.zeros((recon_frames, n_v_hand, 3))
    v_new_warp_vec = np.zeros((recon_frames, n_v_hand, 3))

    for i in tqdm(range((recon_frames // cfg.infer_frame_max) + 1), desc="Inference offset"):
        data_idx_s = i * cfg.infer_frame_max
        data_idx_e = min((i + 1) * cfg.infer_frame_max, recon_frames)

        with torch.no_grad():
            residual_pred = model_residual(pos_enc_input[data_idx_s * n_v: data_idx_e * n_v], pose_code_input[data_idx_s * n_v: data_idx_e * n_v])
        v_new_rest_vec[data_idx_s:data_idx_e] = template_v[np.newaxis, :, :]
        v_new_rest_vec[data_idx_s:data_idx_e, anchor["shead2nohand_x16"]] += residual_pred.reshape(data_idx_e - data_idx_s, n_v, 3).cpu().numpy()

    expression=torch.zeros([1, 10], dtype=dtype, device=device)
    jaw_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    leye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    reye_pose=torch.zeros([1, 3], dtype=dtype, device=device)

    hand_pose=torch.zeros([1, 45], dtype=dtype, device=device)

    deformed_verts = body_model.LBS_deform(custom_lbs_weight, torch.tensor(v_new_rest_vec, dtype=dtype, device=device),
                            body_pose=input_body_pose,
                            betas=input_betas_1,
                            global_orient=input_global_orient,
                            left_hand_pose=input_left_hand_pose,
                            right_hand_pose=input_right_hand_pose,
                            expression=expression.repeat(recon_frames, 1),
                            jaw_pose=jaw_pose.repeat(recon_frames, 1),
                            leye_pose=leye_pose.repeat(recon_frames, 1),
                            reye_pose=reye_pose.repeat(recon_frames, 1),
                            return_verts=True)
    v_new_warp_vec = (deformed_verts + input_transl).cpu().numpy()

    # Visualization
    if cfg.VISUALIZE:
        cfg.rootLogger.info("View")
        from lib.O3D_NB_Vis import o3d_nb_vis

        pcd_list = []

        for pcd in pcds:
            tmp_pcd = o3d.io.read_point_cloud(pcd)
            pcd_list.append(tmp_pcd)

        v_new_rest_vec_view = v_new_rest_vec + input_transl[0].squeeze().cpu().numpy()
        v_new_rest_vec_view[:, :, 0] += 1.5
        o3d_nb_vis({"Mesh0" : {"vertices":v_new_rest_vec_view, "triangles": template_f},
                    "Mesh1" : {"vertices":v_new_warp_vec, "triangles": template_f},
                    "O3D_PCD0" : {"pcd":pcd_list}
                    })

    print("=========================   save   ===========================")
    pcd_dir = os.path.join(target_dir, "train/pcd")
    warp_dir = os.path.join(target_dir, "train/abl1")

    os.makedirs(warp_dir, exist_ok=True)

    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*")))
    tmp_mesh = o3d.geometry.TriangleMesh()
    for i, pcdf in tqdm(enumerate(pcd_files), total=len(pcd_files), desc="Save pose blend base mesh"):
        basename = os.path.basename(pcdf)
        filename_warp = os.path.join(warp_dir, os.path.splitext(basename)[0] + ".ply")

        tmp_mesh.vertices = o3d.utility.Vector3dVector(v_new_warp_vec[i])
        tmp_mesh.triangles = o3d.utility.Vector3iVector(template_f)
        tmp_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(filename_warp, tmp_mesh)
