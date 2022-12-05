import sys
sys.path.append("../")
sys.path.append("./")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import scipy as sp
import argparse
import glob
import open3d as o3d
import igl
from sklearn.neighbors import KDTree

from time import time
from tqdm import tqdm

from pytorch3d.transforms import matrix_to_quaternion

from lib.LaplCal import get_uniform_laplacian_theta, LapEditor
from lib.MotionProtocol import MotionProtocol
import lib.smplx as smplx
from lib.smplx.utils import load_params, get_posemap_custom2, get_posemap_custom
from lib.smplx.lbs import batch_rodrigues
from lib.model.mlp import *
import config as cfg

test_set_CAPE = [
"longlong_babysit_trial2", "longlong_ballet2_trial2", "longlong_basketball_trial2", "longlong_box_trial2", "longlong_catchpick_trial2", "longlong_climb_trial2", "longlong_club_trial2", "longlong_drinkeat_trial2", "longlong_frisbee_trial2", "longlong_golf_trial2", "longlong_handball_trial2", "longlong_hands_up_trial2", "longlong_hockey_trial2", "longlong_housework_trial2", "longlong_lean_trial2", "longlong_music_trial2", "longlong_row_trial2", "longlong_run_trial2", "longlong_shoulders_trial2", "longlong_ski_trial2", "longlong_swim_trial2", "longlong_twist_tilt_trial2", "longlong_volleyball_trial2", "longlong_walk_trial2"
]


parser = argparse.ArgumentParser()

# parser.add_argument("--target_subj", default='anna')
# parser.add_argument("--target_gender", default='female')
# parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
# parser.add_argument('--flathand', default=True)
# parser.add_argument('--epoch_offset', default=100)
# parser.add_argument('--epoch_lap', default=30)

parser.add_argument("--target_subj", default='CT_male3_out')
parser.add_argument("--target_gender", default='male')
parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
parser.add_argument('--flathand', default=False)
parser.add_argument('--epoch_offset', default=40)
parser.add_argument('--epoch_lap', default=30)

# parser.add_argument("--target_subj", default='new_carla')
# parser.add_argument("--target_gender", default='female')
# parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
# parser.add_argument('--flathand', default=False)
# parser.add_argument('--epoch_offset', default=600)
# parser.add_argument('--epoch_lap', default=50)

# parser.add_argument("--target_subj", default='rp_carla_posed_004')
# parser.add_argument("--target_gender", default='female')
# parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
# parser.add_argument('--flathand', default=False)
# parser.add_argument('--epoch_offset', default=300)
# parser.add_argument('--epoch_lap', default=100)

# parser.add_argument("--target_subj", default='hyomin_raise_hand')
# parser.add_argument("--target_gender", default='male')
# parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
# parser.add_argument('--flathand', default=False)
# parser.add_argument('--epoch_offset', default=100)
# parser.add_argument('--epoch_lap', default=30)

args = parser.parse_args()

density = 4

if __name__ == '__main__':
    cfg.rootLogger.info("Start learning pose dependent details")
    target_dir = os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj)

    if cfg.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float32
    density_postfix = "x" + str(density)

    #region AIST
    # target_motion = {"name": "AIST++name", "idx": 10}
    # target_motion_name = "gLO_sBM_cAll_d14_mLO0_ch10"
    # recon_dir = os.path.join(target_dir, target_motion_name)

    # print("Read poses")
    # motion_reader = MotionProtocol(target_motion["name"])
    # smpl_motion = motion_reader.get_params(motionname="gLO_sBM_cAll_d14_mLO0_ch10")
    #endregion

    #region LAPF
    # target_motion = {"name": "LapF", "filename": os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj, "train/smplx_fit.npz")}
    # target_motion = {"name": "LapF", "filename": os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj, "train/smplx_fit_smooth.npz")}
    # target_motion = {"name": "LapF", "filename": os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj, "train/smplx_fit_x3.npz")}
    # target_motion_name = "x4"
    
    target_motion = {"name": "LapF", "filename": os.path.join("/media/hyomin/NVME/LAPFusion/code/dataset/03375_test", "train/smplx_fit.npz")}
    target_motion_name = "03375"
    
    recon_dir = os.path.join(target_dir, "motions", target_motion_name)

    cfg.rootLogger.info("Read poses")
    motion_reader = MotionProtocol(target_motion["name"], target_motion["filename"])
    smpl_motion = motion_reader.get_params()
    #endregion

    #region POP_test
    # target_motion = {"name": "LapF", "filename": os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj, "train/smplx_fit.npz")}
    # target_motion = {"name": "LapF", "filename": os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj, "train/smplx_fit_test.npz")}
    # target_motion_name = "POP_test"
    # recon_dir = os.path.join(target_dir, "motions", target_motion_name)

    # cfg.rootLogger.info("Read poses")
    # motion_reader = MotionProtocol(target_motion["name"], target_motion["filename"])
    # smpl_motion = motion_reader.get_params()
    # #endregion

    #FIXME: move to MotionProtocol class
    #region CAPE
    # # CAPE
    # target_motion_name = test_set_CAPE[0]
    # recon_dir = os.path.join(target_dir, target_motion_name)
    # raw_files = sorted(glob.glob(os.path.join("/NVME/cross_code/cape_release/sequences/03375", test_set_CAPE[0], "*.npz")))
    
    # smpl_motion = {}
    # smpl_motion["pose_params"] = []
    # smpl_motion["global_trans"] = []
    # smpl_motion["filenames"] = []
    # for k, filename_raw in enumerate(raw_files):
    #     if (k + 5) % 10 != 0:
    #         continue
    #     aaa = np.load(filename_raw)
    #     smpl_motion["pose_params"].append(aaa["pose"])
    #     # trans__ = aaa["transl"] - (np.eye(3) + rodrigues_vec_to_rotation_mat(aaa["pose"][:3])) @  [0.00231714, -0.08717195, -0.01025292]
    #     # trans__ =  (rodrigues_vec_to_rotation_mat(aaa["pose"][:3])) @ (aaa["transl"] -[-0.0020802 , -0.22698215,  0.02317401])
    #     trans__ =  aaa["transl"] - (np.eye(3) + rodrigues_vec_to_rotation_mat(aaa["pose"][:3])) @ np.array([0.00231714, -0.08717195,  -0.01025292])
    #     # -0.00217368, -0.24078918,  0.02858379
    #     smpl_motion["global_trans"].append(trans__)
    #     basename = os.path.basename(filename_raw)
    #     smpl_motion["filenames"].append(basename)


    # smpl_motion["pose_params"] = np.asarray(smpl_motion["pose_params"])
    # smpl_motion["global_trans"] =np.asarray(smpl_motion["global_trans"])
    #endregion

    recon_frames = len(smpl_motion["pose_params"])

    model_residual = MLP_Coarse_res(3, 84).to(device)
    model_residual.load_state_dict(torch.load(os.path.join(target_dir, "net/residual_model" + "_e" + str(args.epoch_offset) + ".pts")))
    
    model_lap = MLP_Detail(3, 84).to(device)
    model_lap.load_state_dict(torch.load(os.path.join(target_dir, "net/lap_model" + "_e" + str(args.epoch_lap) + ".pts")))

    model_residual.eval()
    model_lap.eval()

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

    b_mesh = o3d.io.read_triangle_mesh(os.path.join(target_dir, "train/base_mesh.ply"))
    template_v = np.asarray(b_mesh.vertices)
    template_f = np.asarray(b_mesh.triangles)
    template_v = template_v.astype(np.float32)

    params = load_params(os.path.join(target_dir, "train/smplx_fit.npz"), device, dtype)
    (input_body_pose__, input_betas_1, input_global_orient__,
        input_transl__, input_scale_1, 
        input_left_hand_pose__, input_right_hand_pose__) = params
    input_body_pose = torch.tensor(smpl_motion["pose_params"][:, 3:66], dtype=dtype, device=device)
    input_global_orient = torch.tensor(smpl_motion["pose_params"][:, :3], dtype=dtype, device=device)
    input_transl = torch.tensor(smpl_motion["global_trans"], dtype=dtype, device=device).unsqueeze(1)
    expression=torch.zeros([1, 10], dtype=dtype, device=device)
    jaw_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    leye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    reye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    hand_pose=torch.zeros([1, 45], dtype=dtype, device=device)


    anchor = np.load(os.path.join(cfg.DataPath["Main"], cfg.DataPath["Anchor"]))

    n_v = len(anchor["noahnd_coord"])
    n_v_hand = len(anchor["shead_coord"])

    manifold_coord_gpu = torch.from_numpy(anchor["noahnd_coord"]).float().to(device).unsqueeze(0)
    body_skin_weight = body_model.lbs_weights[anchor["smplx2nohand"]][:, :22] # n_verts x 22
    body_skin_weight = torch.where(body_skin_weight > 0.3, 1., 0.)
    pose_map = get_posemap_custom().float().to(device)
    pose_map_vert = torch.where(torch.einsum('ij, jk -> ik', body_skin_weight, pose_map) > 0, 1., 0.) # n_verts x 21

    custom_lbs_weight = body_model.lbs_weights[anchor["smplx2shead"]]

    pose_code_gpu = matrix_to_quaternion(batch_rodrigues(input_body_pose.reshape(-1, 3))).reshape(recon_frames, 21, 4)
    pose_code_gpu = pose_code_gpu.unsqueeze(1).repeat(1, n_v, 1, 1)
    pose_code_gpu = pose_map_vert.unsqueeze(0).unsqueeze(-1) * pose_code_gpu
    pose_code_gpu = pose_code_gpu.reshape(recon_frames, -1, 84)

    pos_enc_input = manifold_coord_gpu.repeat(recon_frames, 1, 1).reshape(-1, 3)
    pose_code_input = pose_code_gpu.reshape(-1, 84)

    # Inference Pose Blend Base Mesh & Save
    v_new_rest_vec = np.zeros((recon_frames, n_v_hand, 3))
    v_new_warp_vec = np.zeros((recon_frames, n_v_hand, 3))

    for i in tqdm(range((recon_frames // cfg.infer_frame_max) + 1), desc="Inference offset"):
        data_idx_s = i * cfg.infer_frame_max
        data_idx_e = min((i + 1) * cfg.infer_frame_max, recon_frames)

        with torch.no_grad():
            residual_pred = model_residual(pos_enc_input[data_idx_s * n_v: data_idx_e * n_v], pose_code_input[data_idx_s * n_v: data_idx_e * n_v])
        v_new_rest_vec[data_idx_s:data_idx_e] = template_v[np.newaxis, :, :]
        v_new_rest_vec[data_idx_s:data_idx_e, anchor["shead2nohand"]] += residual_pred.reshape(data_idx_e - data_idx_s, n_v, 3).cpu().numpy()

    deformed_verts = body_model.LBS_deform(custom_lbs_weight, torch.tensor(v_new_rest_vec, dtype=dtype, device=device),
                            body_pose=input_body_pose,
                            betas=input_betas_1,
                            global_orient=input_global_orient,
                            left_hand_pose=hand_pose.repeat(recon_frames, 1),
                            right_hand_pose=hand_pose.repeat(recon_frames, 1),
                            expression=expression.repeat(recon_frames, 1),
                            jaw_pose=jaw_pose.repeat(recon_frames, 1),
                            leye_pose=leye_pose.repeat(recon_frames, 1),
                            reye_pose=reye_pose.repeat(recon_frames, 1),
                            return_verts=True)
    v_new_warp_vec = (deformed_verts + input_transl).cpu().numpy()

    if True:
        print("=========================  coarse save   ===========================")
        warp_dir = os.path.join(recon_dir, "coarse")
        cano_dir = os.path.join(recon_dir, "coarse_cano")

        os.makedirs(warp_dir, exist_ok=True)
        os.makedirs(cano_dir, exist_ok=True)

        tmp_mesh = o3d.geometry.TriangleMesh()
        for i in range(recon_frames):
            basename = target_motion_name + "_" + str(i).zfill(6)
            print(basename)
            filename_warp = os.path.join(warp_dir, os.path.splitext(basename)[0] + ".ply")
            filename_cano = os.path.join(cano_dir, os.path.splitext(basename)[0] + ".ply")

            tmp_mesh.vertices = o3d.utility.Vector3dVector(v_new_warp_vec[i])
            tmp_mesh.triangles = o3d.utility.Vector3iVector(template_f)
            tmp_mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(filename_warp, tmp_mesh)

            tmp_mesh.vertices = o3d.utility.Vector3dVector(v_new_rest_vec[i])
            tmp_mesh.triangles = o3d.utility.Vector3iVector(template_f)
            tmp_mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(filename_cano, tmp_mesh)

    # Inference Detailed Mesh & View & Save
    n_v = len(anchor["noahnd_coord"])

    S = sp.sparse.load_npz(os.path.join(cfg.DataPath["Main"], "protocol_info/nohand_" + density_postfix + ".npz" ))
    n_dense_v = S.shape[0]

    manifold_coord_gpu = torch.from_numpy(S @ anchor["noahnd_coord"]).float().to(device)
    custom_lbs_weight = torch.tensor(S @ body_model.lbs_weights[anchor["smplx2nohand"]].cpu().numpy(), dtype=dtype, device=device)

    pose_map = get_posemap_custom().float().to(device)

    coarse_verts = v_new_warp_vec[:, anchor["shead2nohand"]]

    cfg.rootLogger.info("Inference delta")
    pose_code = matrix_to_quaternion(batch_rodrigues(input_body_pose.reshape(-1, 3))).reshape(recon_frames, 21, 4)
    all_infer_count = n_dense_v * recon_frames
    mapped_delta = np.zeros((recon_frames, S.shape[0], 3))
    n_infer_once = cfg.lap_infer_frame_max * (n_dense_v + 1)

    body_skin_weight = custom_lbs_weight[:, :22].clone() # n_verts x 22
    body_skin_weight = torch.where(body_skin_weight > 0.3, 1., 0.)

    for i in tqdm(range((all_infer_count // n_infer_once) + 1), desc="inference delta"):
        data_idx_s = i * n_infer_once
        data_idx_e = min((i + 1) * n_infer_once, all_infer_count)

        tmp_index = np.arange(data_idx_s, data_idx_e)
        tmp_index_frame = tmp_index // n_dense_v
        tmp_index_vert = tmp_index % n_dense_v

        with torch.no_grad():
            pose_map_sub_verts = torch.where(torch.einsum('ij, jk -> ik', body_skin_weight[tmp_index_vert], pose_map) > 0, 1., 0.) # n_sub_verts x 21
            mapped_delta[tmp_index_frame, tmp_index_vert, :] \
                    = model_lap(manifold_coord_gpu[tmp_index_vert], (pose_map_sub_verts.unsqueeze(-1) * pose_code[tmp_index_frame]).reshape(-1, 84)).cpu().numpy()

    cfg.rootLogger.info("Rotate delta")

    expression=torch.zeros([1, 10], dtype=dtype, device=device)
    jaw_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    leye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    reye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    mapped_delta = body_model.LBS_deform(custom_lbs_weight, torch.tensor(mapped_delta, dtype=dtype, device=device),
                            only_rotation = True, inverse = False,
                            body_pose=input_body_pose,
                            betas=input_betas_1,
                            global_orient=input_global_orient,
                            left_hand_pose=hand_pose.repeat(recon_frames, 1),
                            right_hand_pose=hand_pose.repeat(recon_frames, 1),
                            expression=expression.repeat(recon_frames, 1),
                            jaw_pose=jaw_pose.repeat(recon_frames, 1),
                            leye_pose=leye_pose.repeat(recon_frames, 1),
                            reye_pose=reye_pose.repeat(recon_frames, 1),
                            return_verts=True).cpu().numpy()


    #region Laplacian reconstruction
    cfg.rootLogger.info("Gen Laplacian editor")

    #region Pick constraint vertices
    mmesh = o3d.geometry.TriangleMesh()
    mmesh.vertices = o3d.utility.Vector3dVector(anchor["noahnd_coord"])
    mmesh.triangles = o3d.utility.Vector3iVector(anchor["nohand_tri"])
    ppcd = mmesh.sample_points_poisson_disk(3000)
    # ppcd = mmesh.sample_points_poisson_disk(density * 150)
    ttree = KDTree(anchor["shead_coord"])
    _, nn_idx = ttree.query(np.asarray(ppcd.points))
    picked_points = nn_idx[:, 0]
    hand_foot_v_idx = np.array([5213, 4788, 4914, 5009, 5126, 2283, 1856, 1982, 2094, 2195, 5586, 5589, 5675, 5599, 5596, 5756, 2697, 2780, 2694, 2792, 2797, 2867, 2918, 2882, 2849, 5738, 5775, 5813])
    picked_points = np.concatenate((picked_points, hand_foot_v_idx), axis = 0)
    #endregion

    with torch.no_grad():
        body_model_output = body_model(body_pose=input_body_pose,
                                        betas=input_betas_1,
                                        global_orient=input_global_orient,
                                        left_hand_pose=hand_pose.repeat(recon_frames, 1),
                                        right_hand_pose=hand_pose.repeat(recon_frames, 1),
                                        expression=expression.repeat(recon_frames, 1),
                                        jaw_pose=jaw_pose.repeat(recon_frames, 1),
                                        leye_pose=leye_pose.repeat(recon_frames, 1),
                                        reye_pose=reye_pose.repeat(recon_frames, 1),
                                        return_verts=True)

    smplx_verts_ = (body_model_output.vertices + input_transl).cpu().numpy()
    smplx_S = sp.sparse.load_npz(os.path.join(cfg.DataPath["Main"], "protocol_info/smplx_" + density_postfix + ".npz" ))
    smplx_verts = np.zeros((recon_frames, smplx_S.shape[0], 3))

    for i in range(recon_frames):
        smplx_verts[i] = smplx_S @ smplx_verts_[i]

    shead_S = sp.sparse.load_npz(os.path.join(cfg.DataPath["Main"], "protocol_info/shead_" + density_postfix + ".npz" ))

    coarse_vertex_w_hand = smplx_verts[:, anchor["smplx2shead"], :]
    v_sub_dense = shead_S @ coarse_vertex_w_hand[0]

    s_time_editor = time()
    L_dense = get_uniform_laplacian_theta(v_sub_dense, anchor["shead_" + density_postfix + "_tri"])
    lap_editor = LapEditor(L_dense, v_sub_dense, picked_points)
    f = anchor["nohand_tri"]
    e_time_editor = time()
    cfg.rootLogger.debug("Gen time: " + str(e_time_editor - s_time_editor))

    cfg.rootLogger.info("Laplacian reconstruction")
    recon_vertex_t = np.zeros((recon_frames, shead_S.shape[0], 3))
    for i in tqdm(range(recon_frames), desc="Laplacian reconstruction"):
        M_coarse = igl.massmatrix(coarse_verts[i], f, igl.MASSMATRIX_TYPE_VORONOI)
        m_new = (S @ np.squeeze(np.asarray(M_coarse.sum(1)))) / density

        delta_smpl = L_dense @ shead_S @ coarse_vertex_w_hand[i]

        tmp_delta_new = mapped_delta[i] * m_new[:, np.newaxis]

        delta_new = delta_smpl
        delta_new[anchor["shead2nohand_" + density_postfix]] = tmp_delta_new

        tmp_coarse_verts = anchor["shead_coord"]
        tmp_coarse_verts[anchor["shead2nohand"]] = coarse_verts[i]
        tmp_coarse_verts[hand_foot_v_idx] = coarse_vertex_w_hand[i][hand_foot_v_idx]

        lap_editor.update_anchor(tmp_coarse_verts, picked_points)

        v_new = lap_editor.recon(delta_new)
        recon_vertex_t[i] = v_new

    #endregion


    # Visualization
    if cfg.VISUALIZE:
        cfg.rootLogger.info("View")
        from lib.O3D_NB_Vis import o3d_nb_vis

        o3d_nb_vis({"Mesh0" : {"vertices":recon_vertex_t, "triangles": anchor["shead_" + density_postfix + "_tri"]}})


    # Save
    os.makedirs(recon_dir, exist_ok=True)
    for t in tqdm(range(recon_frames), desc="Save"):
        filename_recon = os.path.join(recon_dir , str(t).zfill(6) + ".ply")
        tmp_mesh = o3d.geometry.TriangleMesh()
        tmp_mesh.vertices = o3d.utility.Vector3dVector(recon_vertex_t[t])
        tmp_mesh.triangles = o3d.utility.Vector3iVector(anchor["shead_" + density_postfix + "_tri"])
        tmp_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(filename_recon, tmp_mesh)
