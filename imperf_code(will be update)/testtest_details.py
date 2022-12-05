import sys
sys.path.append("../")
sys.path.append("./")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import scipy as sp
import argparse
import glob
import random
import open3d as o3d
import igl
from sklearn.neighbors import KDTree

from time import time
from tqdm import tqdm

from pytorch3d.transforms import matrix_to_quaternion

from lib.LaplCal import get_uniform_laplacian_theta, LapEditor
import config as cfg
from lib.model.mlp import *
from lib.smplx.utils import load_params, get_posemap_custom
from lib.smplx.lbs import batch_rodrigues
import lib.smplx as smplx

parser = argparse.ArgumentParser()

# skin = np.load("/media/hyomin/NVME/cross_code/MetaAvatar-release/body_models/misc/skinning_weights_all.npz")
# aaa = np.load("/media/hyomin/NVME/cross_code/MetaAvatar-release/data/CAPE_sampling-rate-1/03375/longlong_ATUsquat_trial1/longlong_ATUsquat_trial1.000009.npz")
# bbb = np.load("/media/hyomin/NVME/cross_code/POP/data/cape/packed/03375_longlong/test/longlong_ATUsquat_trial1.000009.npz")

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(aaa["a_pose_mesh_points"] - aaa["trans"])
# pcd2 = o3d.geometry.PointCloud()
# # dad = np.concatenate((bbb["body_verts"], np.ones(len(bbb["body_verts"]))[:, np.newaxis]), axis=-1)
# # dad = np.einsum("bij, bj -> bi", bbb["vtransf"], dad)[:, :3]
# # dad = np.einsum("bji, bj -> bi", bbb["vtransf"][:, :3, :3], bbb["body_verts"] - bbb["vtransf"][:, :3, 3])
# # pcd2.points = o3d.utility.Vector3dVector(dad)

# pcd2.points = o3d.utility.Vector3dVector(bbb["body_verts"])

# pcd = o3d.io.read_point_cloud("/media/hyomin/NVME/cross_code/POP/results/saved_samples/POP_pretrained_CAPEdata_14outfits/test_seen/query_resolution256/03375_longlong/POP_pretrained_CAPEdata_14outfits_03375_longlong_ATUsquat_trial1.000009_pred.ply")

# o3d.visualization.draw_geometries([pcd, pcd2])


# parser.add_argument("--target_subj", default='carla')
# parser.add_argument("--target_gender", default='female')
# parser.add_argument('--RGBD', default=False, help='Is Point Cloud?')
# parser.add_argument('--flathand', default=True)
# parser.add_argument('--epoch', default=100)

# parser.add_argument("--target_subj", default='new_carla')
# parser.add_argument("--target_gender", default='female')
# parser.add_argument('--RGBD', default=False, help='Is Point Cloud?')
# parser.add_argument('--flathand', default=True)
# parser.add_argument('--epoch', default=20)

# parser.add_argument("--target_subj", default='beatric')
# parser.add_argument("--target_gender", default='female')
# parser.add_argument('--RGBD', default=False, help='Is Point Cloud?')
# parser.add_argument('--flathand', default=True)
# parser.add_argument('--epoch', default=100)

parser.add_argument("--target_subj", default='hyomin_raise_hand')
parser.add_argument("--target_gender", default='male')
parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
parser.add_argument('--flathand', default=False)
parser.add_argument('--epoch', default=10)

# parser.add_argument("--target_subj", default='anna')
# parser.add_argument("--target_gender", default='female')
# parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
# parser.add_argument('--flathand', default=True)
# parser.add_argument('--epoch', default=30)

# parser.add_argument("--target_subj", default='hyomin_dance')
# parser.add_argument("--target_gender", default='male')
# parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
# parser.add_argument('--flathand', default=False)
# parser.add_argument('--epoch', default=30)

args = parser.parse_args()

density = 16

if __name__ == '__main__':
    cfg.make_dir_structure(args.target_subj)
    # cfg.set_log_file(os.path.join(cfg.DataPath["Main"], "logs", args.target_subj, os.path.splitext(os.path.basename(__file__))[0]))
    cfg.rootLogger.info("Inference pose dependent details")
    target_dir = os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj)

    if cfg.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float32
    density_postfix = "x" + str(density)

    recon_dir = os.path.join(target_dir, "train/recon")
    os.makedirs(recon_dir, exist_ok=True)

    coarse_files = sorted(glob.glob(os.path.join(target_dir, "train/coarse/*.ply")))
    coarse_cano_files = sorted(glob.glob(os.path.join(target_dir, "train/coarse_cano/*.ply")))
    recon_frames = len(coarse_files)

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

    cfg.rootLogger.info("Read poses")
    params = load_params(os.path.join(target_dir, "train/smplx_fit_new.npz"), device, dtype)
    (input_body_pose, input_betas_1, input_global_orient,
        input_transl, input_scale_1, 
        input_left_hand_pose, input_right_hand_pose) = params

    anchor = np.load(os.path.join(cfg.DataPath["Main"], cfg.DataPath["Anchor"]))
    n_v = len(anchor["noahnd_coord"])

    S = sp.sparse.load_npz(os.path.join(cfg.DataPath["Main"], "protocol_info/nohand_" + density_postfix + ".npz" ))
    n_dense_v = S.shape[0]

    manifold_coord_gpu = torch.from_numpy(S @ anchor["noahnd_coord"]).float().to(device)
    custom_lbs_weight = torch.tensor(S @ body_model.lbs_weights[anchor["smplx2nohand"]].cpu().numpy(), dtype=dtype, device=device)

    pose_map = get_posemap_custom().float().to(device)

    model_lap = MLP_Detail(3, 84).to(device)
    model_lap.load_state_dict(torch.load(os.path.join(target_dir, "net/lap_model" + "_e" + str(args.epoch) + ".pts")))
    model_lap.eval()

    #region Read coarse mesh
    cfg.rootLogger.info("Read coarse mesh")
    coarse_verts = np.zeros((recon_frames, S.shape[1], 3))
    for i in tqdm(range(recon_frames), desc="Read coarse mesh"):
        tmp_mesh = o3d.io.read_triangle_mesh(coarse_files[i])
        tmp_v = np.asarray(tmp_mesh.vertices)
        f = np.asarray(tmp_mesh.triangles)
        coarse_verts[i] = tmp_v[anchor["shead2nohand"]]
    #endregion

    #region Inference delta
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
    #endregion

    #region Rotate delta
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
                            left_hand_pose=input_left_hand_pose,
                            right_hand_pose=input_right_hand_pose,
                            expression=expression.repeat(recon_frames, 1),
                            jaw_pose=jaw_pose.repeat(recon_frames, 1),
                            leye_pose=leye_pose.repeat(recon_frames, 1),
                            reye_pose=reye_pose.repeat(recon_frames, 1),
                            return_verts=True).cpu().numpy()
    #endregion

    #region Laplacian reconstruction
    cfg.rootLogger.info("Gen Laplacian editor")

    #region Pick constraint vertices
    mmesh = o3d.geometry.TriangleMesh()
    mmesh.vertices = o3d.utility.Vector3dVector(anchor["noahnd_coord"])
    mmesh.triangles = o3d.utility.Vector3iVector(anchor["nohand_tri"])
    ppcd = mmesh.sample_points_poisson_disk(800)
    # ppcd = mmesh.sample_points_poisson_disk(density * 150)
    ttree = KDTree(anchor["shead_coord"])
    _, nn_idx = ttree.query(np.asarray(ppcd.points))
    picked_points = nn_idx[:, 0]
    hand_foot_v_idx = np.array([5213, 4788, 4914, 5009, 5126, 2283, 1856, 1982, 2094, 2195, 5586, 5589, 5675, 5599, 5596, 5756, 2697, 2780, 2694, 2792, 2797, 2867, 2918, 2882, 2849, 5738, 5775, 5813])
    
    # picked_points = np.load(os.path.join(cfg.DataPath["Main"], "protocol_info/sparse_marker.npy"))

    picked_points = np.concatenate((picked_points, hand_foot_v_idx), axis = 0)
    print(len(picked_points))
    #endregion

    with torch.no_grad():
        body_model_output = body_model(body_pose=input_body_pose,
                                        betas=input_betas_1,
                                        global_orient=input_global_orient,
                                        left_hand_pose=input_left_hand_pose,
                                        right_hand_pose=input_right_hand_pose,
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
        pcd_dir = os.path.join(target_dir, "train/pcd")
        pcds = sorted(glob.glob(os.path.join(pcd_dir, "*.ply")))

        pcd_list = []

        for pcd in pcds:
            tmp_pcd = o3d.io.read_point_cloud(pcd)
            pcd_list.append(tmp_pcd)

        o3d_nb_vis({"Mesh0" : {"vertices":recon_vertex_t, "triangles": anchor["shead_" + density_postfix + "_tri"]},
                    "O3D_PCD0" : {"pcd":pcd_list}
                    })


    # Save
    for t in tqdm(range(recon_frames), desc="Save"):
        basename = os.path.basename(coarse_files[t])
        filename_recon = os.path.join(recon_dir, os.path.splitext(basename)[0] + ".posed.ply")
        tmp_mesh = o3d.geometry.TriangleMesh()
        tmp_mesh.vertices = o3d.utility.Vector3dVector(recon_vertex_t[t])
        tmp_mesh.triangles = o3d.utility.Vector3iVector(anchor["shead_" + density_postfix + "_tri"])
        tmp_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(filename_recon, tmp_mesh)


