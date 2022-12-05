import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("../")
sys.path.append("./")

import numpy as np
import torch
import torch.nn as nn
import scipy as sp
import argparse
import glob
import copy
import time
import random

import progressbar
from smplpytorch_local.pytorch.smpl_layer import SMPL_Layer

from pytorch3d.transforms import matrix_to_quaternion

from LapFusion.laplacian_coords import *
import LapFusion.config as cfg
from LapFusion.dataset import *
from LapFusion.dvm2smpl.unwarp_delta import *
from LapFusion.model.mlp import *
from LapFusion.smplx.utils import load_params, get_posemap, get_posemap_custom, smplx_parents
from LapFusion.smplx.lbs import batch_rodrigues
import LapFusion.smplx as smplx

# import pickle
# with open('/NVME/cross_code/smplx/output/00071.pkl', 'rb') as f:
#     data = pickle.load(f)


g_play = False
g_break = False
g_refresh = False
g_separate = False
g_tar_idx = 0
frame_idx = 0
wait_frame = 0.03

def play_onoff():
    global g_play
    g_play = not g_play
def play_stop():
    global g_break
    g_break = True
def upframe():
    global frame_idx, g_refresh
    frame_idx += 1
    g_refresh = True
def downframe():
    global frame_idx, g_refresh
    frame_idx -= 1
    g_refresh = True
def separate():
    global g_separate, g_refresh
    g_separate = not g_separate
    g_refresh = True
def speedup():
    global wait_frame
    wait_frame -= 0.005
def speeddown():
    global wait_frame
    wait_frame += 0.005

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = ''):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue

random_seed = 32
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--root", 
    default='/NVME/LAPFusion/code/dataset')
parser.add_argument("--source_subj", 
    default='00096')
parser.add_argument("--target_subj", 
    default='hyomin_2')
    # default='anna')
    # default='hyomin_0119_3')
    # default='hyomin_0119_2')
parser.add_argument("--target_gender", 
    default='male')
parser.add_argument("--model_path_male", 
    default='models/smplx/SMPLX_MALE.npz')
parser.add_argument("--model_path_female", 
    default='models/smplx/SMPLX_FEMALE.npz')
parser.add_argument('--RGBD', default=True, action='store_true', help='Is Point Cloud?')

density = 16
use_lin = True

if __name__ == '__main__':
    device = torch.device('cuda' if cfg.is_cuda else 'cpu')
    dtype = torch.float32
    config = parser.parse_args()

    target_subj = config.target_subj
    source_subj = config.source_subj
    gender = config.target_gender

    model_path = os.path.join(config.root, (config.model_path_male if gender=='male' else config.model_path_female))
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
                        use_pca=False,
                        dtype=dtype,
                        )
    body_model = smplx.create(gender=gender, **model_params)
    body_model.to(device)



    target_dir = os.path.join(cfg.DataPath["Main"], target_subj)
    source_dir = os.path.join(cfg.DataPath["Main"], source_subj)

    density_postfix = "x" + str(density)

    recon_dir = os.path.join(target_dir, "train/detail_trans")
    os.makedirs(recon_dir, exist_ok=True)

    coarse_files = sorted(glob.glob(os.path.join(target_dir, "train/coarse/*.ply")))
    coarse_cano_files = sorted(glob.glob(os.path.join(target_dir, "train/coarse_cano/*.ply")))
    recon_frames = len(coarse_files)





    # smplx_dir = os.path.join(target_dir, "train/smplx")
    # os.makedirs(smplx_dir, exist_ok=True)
    # params = load_params(os.path.join(target_dir, "train/smplx_fit.npz"), device, dtype)
    # (input_body_pose, input_betas_1, input_global_orient,
    #     input_transl, input_scale_1, 
    #     input_left_hand_pose, input_right_hand_pose) = params
    # expression=torch.zeros([1, 10], dtype=dtype, device=device)
    # jaw_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    # leye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    # reye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    # body_model_output = body_model(body_pose=input_body_pose,
    #                                 betas=input_betas_1,
    #                                 global_orient=input_global_orient,
    #                                 left_hand_pose=input_left_hand_pose,
    #                                 right_hand_pose=input_right_hand_pose,
    #                                 expression=expression.repeat(recon_frames, 1),
    #                                 jaw_pose=jaw_pose.repeat(recon_frames, 1),
    #                                 leye_pose=leye_pose.repeat(recon_frames, 1),
    #                                 reye_pose=reye_pose.repeat(recon_frames, 1),
    #                                 return_verts=True)
    # smplx_verts_ = (body_model_output.vertices + input_transl).cpu().numpy()
    # smplx_triangles_ = body_model.faces.astype(np.int64)
    # for i in range(len(smplx_verts_)):
    #     basename = os.path.basename(coarse_files[i])
    #     print(basename)
    #     filename_smplx = os.path.join(smplx_dir, os.path.splitext(basename)[0] + ".ply")
    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(smplx_verts_[i])
    #     mesh.triangles = o3d.utility.Vector3iVector(smplx_triangles_)
    #     mesh.compute_vertex_normals()
    #     o3d.io.write_triangle_mesh(filename_smplx, mesh)


    print("Read poses Source")
    params = load_params(os.path.join(source_dir, "train/smplx_fit.npz"), device, dtype)
    (input_body_pose, input_betas_1, input_global_orient,
        input_transl, input_scale_1, 
        input_left_hand_pose, input_right_hand_pose) = params


    S = sp.sparse.load_npz(os.path.join(cfg.DataPath["Root"], "protocol_info/nohand_" + density_postfix + ".npz" ))
    n_dense_v = S.shape[0]

    anchor = np.load(os.path.join(cfg.DataPath["Root"], cfg.DataPath["Anchor"]))
    n_v = len(anchor["noahnd_coord"])

    manifold_coord_gpu = torch.from_numpy(S @ anchor["noahnd_coord"]).float().to(device)
    custom_lbs_weight = torch.tensor(S @ body_model.lbs_weights[anchor["smplx2nohand"]].cpu().numpy(), dtype=dtype, device=device)
    body_skin_weight = custom_lbs_weight[:, :22].cpu().numpy()

    # pred_sub = torch.from_numpy(S @ anchor["coord"]).float().to(device)

    # pose_map = get_posemap('children', 22, smplx_parents, 3, no_head = True).float().to(device) # 22 x 21
    # pose_map = get_posemap('both', 22, smplx_parents, 2, no_head = True).float().to(device) # 22 x 21
    pose_map = get_posemap_custom().float().to(device)

    # pose_map = torch.ones((22,21), dtype=dtype, device=device)

    if use_lin:
        model_lap = MLP_CDF_lin(3, 84).to(device)
        model_lap.load_state_dict(torch.load(os.path.join(source_dir, "net/1lap_model" + "_e" + str(110) + "_lin.pts")))
    else:
        model_lap = MLP_CDF(3, 84).to(device)
        model_lap.load_state_dict(torch.load(os.path.join(target_dir, "net/1lap_model" + "_e" + str(110) + ".pts")))

    model_lap.eval()

    print("Read coarse mesh")
    coarse_verts = np.zeros((recon_frames, S.shape[1], 3))
    coarse_cano_verts = np.zeros((recon_frames, S.shape[1], 3))
    bar = progressbar.ProgressBar(maxval=recon_frames,widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',]).start()
    for i in range(recon_frames):
        tmp_v, f = igl.read_triangle_mesh(coarse_files[i])
        coarse_verts[i] = tmp_v
        tmp_v, f = igl.read_triangle_mesh(coarse_cano_files[i])
        coarse_cano_verts[i] = tmp_v
        bar.update(i)
    bar.finish()

    print("inference delta")

    anchor_skin_max_idx = np.argmax(body_skin_weight, axis=-1)
    skin_idx_gpu = torch.from_numpy(anchor_skin_max_idx).long().to(device)

    dsds = len(input_body_pose)

    pose_code = matrix_to_quaternion(batch_rodrigues(input_body_pose.reshape(-1, 3))).reshape(dsds, 21, 4)
    # pose_code = input_body_pose.reshape(-1, 21, 3)
    
    all_infer_count = n_dense_v * dsds

    mapped_delta_s = np.zeros((dsds, S.shape[0], 3))

    n_infer_once = cfg.lap_infer_frame_max * n_dense_v
    bar = progressbar.ProgressBar(maxval=(all_infer_count // n_infer_once) + 1, widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',]).start()
    for i in range((all_infer_count // n_infer_once) + 1):
        data_idx_s = i * n_infer_once
        data_idx_e = min((i + 1) * n_infer_once, all_infer_count)

        tmp_index = np.arange(data_idx_s, data_idx_e)
        tmp_index_frame = tmp_index // n_dense_v
        tmp_index_vert = tmp_index % n_dense_v

        with torch.no_grad():
            mapped_delta_s[tmp_index_frame, tmp_index_vert, :] \
                    = model_lap(manifold_coord_gpu[tmp_index_vert], (pose_map[skin_idx_gpu[tmp_index_vert]].unsqueeze(2) * pose_code[tmp_index_frame]).reshape(-1, 84)).cpu().numpy()
        bar.update(i)
    bar.finish()


    mapped_delta = np.zeros((recon_frames, S.shape[0], 3))
    mapped_delta = np.repeat(mapped_delta_s[0, :, :][np.newaxis, :, :], recon_frames, axis=0)


    print("Read poses")
    params = load_params(os.path.join(target_dir, "train/smplx_fit.npz"), device, dtype)
    (input_body_pose, input_betas_1, input_global_orient,
        input_transl, input_scale_1, 
        input_left_hand_pose, input_right_hand_pose) = params


    print("Rotate delta")

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


    print("Glue Hand")
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
    smplx_S = sp.sparse.load_npz(os.path.join(cfg.DataPath["Root"], "protocol_info/smplx_" + density_postfix + ".npz" ))
    smplx_verts = np.zeros((recon_frames, smplx_S.shape[0], 3))

    for i in range(recon_frames):
        smplx_verts[i] = smplx_S @ smplx_verts_[i]

    shead_S = sp.sparse.load_npz(os.path.join(cfg.DataPath["Root"], "protocol_info/shead_" + density_postfix + ".npz" ))

    coarse_vertex_w_hand = smplx_verts[:, anchor["smplx2shead"], :]
    v_sub_dense = shead_S @ coarse_vertex_w_hand[0]

    mmesh = o3d.geometry.TriangleMesh()
    mmesh.vertices = o3d.utility.Vector3dVector(anchor["noahnd_coord"])
    mmesh.triangles = o3d.utility.Vector3iVector(anchor["nohand_tri"])
    ppcd = mmesh.sample_points_poisson_disk(density * 150)
    ttree = KDTree(anchor["shead_coord"])
    _, nn_idx = ttree.query(np.asarray(ppcd.points))
    picked_points = nn_idx[:, 0]

    hand_wrist = np.load("/NVME/dataset/CAPE/core_body/hand_wrist.npy")
    picked_points = np.setdiff1d(picked_points, hand_wrist)
    # face_v_idx = np.load("/NVME/dataset/CAPE/core_body/face_idx.npz")["v_idx"]
    # picked_points = np.setdiff1d(picked_points, face_v_idx)

    # hand_foot_v_idx = np.array([21895 ,5030  ,21219 ,4808  ,22216 ,12921 ,11580 ,11965 ,12329 ,12662 ,24071 ,24075 ,24320 ,24328 ,24082 ,5757  ,14882 ,15134 ,14880 ,15138 ,2707  ,2866])
    # hand_foot_v_idx = np.array([10993,11392,10828,10841,20227,20626,20065,20232,24071 ,24075 ,24320 ,24328 ,24082 ,5757  ,14882 ,15134 ,14880 ,15138 ,2707  ,2866])
    hand_foot_v_idx = np.array([5213, 4788, 4914, 5009, 5126, 2283, 1856, 1982, 2094, 2195, 5586, 5589, 5675, 5599, 5596, 5756, 2697, 2780, 2694, 2792, 2797, 2867, 2918, 2882, 2849, 5738, 5775, 5813])
    # hand_foot_v_idx = np.array([4710, 4393, 4709, 4447, 1461, 1506, 1516, 1779, 2906, 2852, 2879, 2897, 5838, 5802, 5743, 5769])
    picked_points = np.concatenate((picked_points, hand_foot_v_idx), axis = 0)
    
    L_dense = get_uniform_laplacian(v_sub_dense, anchor["shead_" + density_postfix + "_tri"])
    lap_editor = LapEditor(L_dense, v_sub_dense, picked_points)


    print("Reconstruction")
    recon_vertex_t = np.zeros((recon_frames, shead_S.shape[0], 3))
    bar = progressbar.ProgressBar(maxval=recon_frames,widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',]).start()
    for i in range(0, recon_frames):
        M_coarse = igl.massmatrix(coarse_verts[i], f, igl.MASSMATRIX_TYPE_VORONOI)
        m_new = (S @ np.squeeze(np.asarray(M_coarse.sum(1)))) / density

        # M_coarse2 = igl.massmatrix(coarse_vertex_w_hand[i], shead_mesh_protocol["c_v_face"], igl.MASSMATRIX_TYPE_VORONOI)
        # m_new2 = (shead_S @ np.squeeze(np.asarray(M_coarse2.sum(1)))) / 64

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
        bar.update(i)
    bar.finish()




    print("View")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(recon_vertex_t[0])
    mesh.triangles = o3d.utility.Vector3iVector(anchor["shead_" + density_postfix + "_tri"])
    mesh.compute_vertex_normals()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(256, lambda vis: play_stop()) #ESC
    vis.register_key_callback(262, lambda vis: upframe()) #
    vis.register_key_callback(263, lambda vis: downframe()) #
    vis.register_key_callback(266, lambda vis: speedup()) #
    vis.register_key_callback(267, lambda vis: speeddown()) #
    vis.register_key_callback(32, lambda vis: play_onoff())
    vis.register_key_callback(290, lambda vis: separate()) # F1
    vis.create_window()
    vis.add_geometry(mesh)
    frame_idx = 0


    pcd_dir2 = os.path.join(target_dir, "train/pcd")
    pcds2 = sorted(glob.glob(os.path.join(pcd_dir2, "*.ply")))
    pcd_list2 = []
    max_p_len = 0
    for pcd in pcds2:
        tmp_pcd2 = o3d.io.read_point_cloud(pcd)
        tmp_pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
        pcd_list2.append(tmp_pcd2)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_list2[0].points))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd_list2[frame_idx].normals))
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(pcd)

    start = time.time()
    redraw = True
    while g_break == False:
        frame_idx %= recon_frames

        end = time.time()
        if(end - start > wait_frame):
            # print(end - start)
            start = time.time()
            redraw = True
        else:
            pass

        if not g_play:
            redraw = False

        if redraw or g_refresh:
            mesh.vertices = o3d.utility.Vector3dVector(recon_vertex_t[frame_idx])
            mesh.compute_triangle_normals()
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_list2[frame_idx].points))
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd_list2[frame_idx].normals))
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            if g_separate:
                mesh.translate((-1, 0, 0))
                pcd.translate((1, 0, 0))
            vis.update_geometry(pcd)

            if redraw:
                frame_idx += 1
            g_refresh = False
            redraw = False
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

    print("Save")
    for t in range(recon_frames):
    # for t in range(5, n_frames, 10):
        basename = os.path.basename(coarse_files[t])
        print(basename)
        filename_recon = os.path.join(recon_dir, os.path.splitext(basename)[0] + ".posed.ply")
        # igl.write_obj(filename_recon, recon_vertex_t[t], f)
        tmp_mesh = o3d.geometry.TriangleMesh()
        tmp_mesh.vertices = o3d.utility.Vector3dVector(recon_vertex_t[t])
        tmp_mesh.triangles = o3d.utility.Vector3iVector(anchor["shead_" + density_postfix + "_tri"])
        tmp_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(filename_recon, tmp_mesh)
        # igl.write_off(filename_recon, recon_vertex_t[t], anchor["shead_" + density_postfix + "_tri"], np.ones_like(recon_vertex_t[t]) * 0.7)


