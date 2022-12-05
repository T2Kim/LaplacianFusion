
import sys
sys.path.append("../")
sys.path.append("./")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import random
import argparse
import glob
import open3d as o3d

from time import time
from tqdm import tqdm

from lib.chamferdist import chamfer_distancePP_diff, chamfer_distancePP_diff_both
from pytorch3d.ops.knn import knn_gather, knn_points
from lib.LaplCal import get_uniform_laplacian_1
from lib.human_fitting.utils_smplify import GMoF
import config as cfg
from lib.model.mlp import *

import lib.smplx as smplx
from lib.utils import load_params, get_posemap_custom
from lib.smplx.lbs import batch_rodrigues
from lib.custom_lbs import LBS_deform

from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import mesh_edge_loss

random_seed = 4332
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

demo_frame = 0
def upframe():
    global demo_frame, g_refresh
    demo_frame += 1
    g_refresh = True
def downframe():
    global demo_frame, g_refresh
    demo_frame -= 1
    g_refresh = True

def read_pcd_padded(filenames, device, dtype):
    in_data = {}
    pcd_list_o3d = []
    coarse_pcd_list = []
    max_p_len = 0
    recon_frames = len(filenames)

    timer_start = time()
    for pcd_name in tqdm(filenames, desc="read files"):
        tmp_pcd = o3d.io.read_point_cloud(pcd_name)
        pcd_list_o3d.append(tmp_pcd)
        tmp_uni_down_pcd = tmp_pcd
        new_p = np.asarray(tmp_uni_down_pcd.points)
        coarse_pcd_list.append(new_p)
        if max_p_len < len(new_p):
            max_p_len = len(new_p)

    tar_p_gpu = torch.zeros((recon_frames, max_p_len, 3)).to(device)
    mask_p_gpu = torch.zeros((recon_frames, max_p_len)).to(device)
    for i in range(recon_frames):
        tar_p_gpu[i, :len(coarse_pcd_list[i])] = torch.from_numpy(coarse_pcd_list[i]).to(device)
        mask_p_gpu[i, :len(coarse_pcd_list[i])] = 1
        
    in_data["pcd_torch"] = tar_p_gpu
    in_data["mask_torch"] = mask_p_gpu.bool()
    in_data["pcd_list_o3d"] = pcd_list_o3d

    timer_end = time()
    cfg.rootLogger.debug(f'Read time: {timer_end - timer_start}s')

    return in_data

parser = argparse.ArgumentParser()
parser.add_argument("--target_subj", default='hyomin_example')
parser.add_argument("--target_gender", default='male')
parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
parser.add_argument('--add_noise', default=False, help='Add noise?')
parser.add_argument("--lap_reg_weight", type=float, default=2)
parser.add_argument('--flathand', default=False)

args = parser.parse_args()

if __name__ == '__main__':
    cfg.make_dir_structure(args.target_subj)
    cfg.set_log_file(os.path.join(cfg.DataPath["Main"], "logs", args.target_subj, os.path.splitext(os.path.basename(__file__))[0]))
    cfg.rootLogger.info("Start learning pose dependent offset")
    target_dir = os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj)

    if cfg.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float32

    weight_vec = {}
    weight_vec["chamfer"] = 1
    weight_vec["lap_reg"] = args.lap_reg_weight

    #region SMPLX body model & parameters
    model_path = os.path.join(cfg.DataPath["Main"], (cfg.DataPath["model_path_male"] if args.target_gender=='male' else cfg.DataPath["model_path_female"] ))
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
                        use_pca=False,
                        dtype=dtype,
                        )
    body_model = smplx.create(gender=args.target_gender, **model_params)
    body_model.to(device)

    params = load_params(os.path.join(target_dir, "train/smplx_fit.npz"), device, dtype)
    (input_body_pose, input_betas_1, input_global_orient,
        input_transl, input_scale_1, 
        input_left_hand_pose, input_right_hand_pose) = params

    robustifier = GMoF(rho=100)

    expression=torch.zeros([1, 10], dtype=dtype, device=device)
    jaw_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    leye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    reye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    #endregion

    #region Read PCD
    pcd_dir = os.path.join(target_dir, "train/pcd")
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.ply")))

    recon_frames = len(pcd_files)

    in_data = read_pcd_padded(pcd_files, device, dtype)
    tar_p_gpu = in_data["pcd_torch"]
    mask_p_gpu = in_data["mask_torch"]
    pcd_list = in_data["pcd_list_o3d"]
    tar_p_weight_gpu = 2 * torch.ones_like(mask_p_gpu.float())
    tar_p_weight_gpu[mask_p_gpu] = torch.exp(torch.ones_like(mask_p_gpu.float())[mask_p_gpu] / torch.abs(in_data["pcd_torch"][:, :, 2][mask_p_gpu]))
    # tar_p_weight_gpu[mask_p_gpu] = torch.exp(-2 * torch.abs(in_data["pcd_torch"][:, :, 2][mask_p_gpu]))
    # tar_p_weight_gpu[mask_p_gpu] = torch.exp(torch.ones_like(mask_p_gpu.float())[mask_p_gpu] * (-2 * torch.abs(in_data["pcd_torch"][:, :, 2][mask_p_gpu])))
    # tar_p_weight_gpu = torch.ones_like(mask_p_gpu.float())
    #endregion

    #region Set base mesh protocol
    anchor = np.load(os.path.join(cfg.DataPath["Main"], cfg.DataPath["Anchor"]))
    custom_lbs_weight = body_model.lbs_weights[anchor["smplx2shead"]]
    
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
    
    template_v = np.asarray(b_mesh.vertices)
    template_f = np.asarray(b_mesh.triangles)
    template_v = template_v.astype(np.float32)

    #endregion

    #region Ready to optimze
    template_v_gpu = torch.from_numpy(template_v).to(device).unsqueeze(0)
    template_f_gpu = torch.from_numpy(template_f).to(device)

    L = get_uniform_laplacian_1(template_f)
    L = L.tocoo()
    L_gpu = torch.sparse.FloatTensor(torch.LongTensor([L.row.tolist(), L.col.tolist()]),
                                torch.FloatTensor(L.data.astype(np.float32))).to(device)
    L_gpu = L_gpu.type(torch.FloatTensor).to(device).to_dense()
    # L_gpu = torch.sparse.LongTensor(torch.LongTensor([L.row.tolist(), L.col.tolist()]),
    #                             torch.LongTensor(L.data.astype(np.int32))).to(device)
    # L_gpu = L_gpu.type(torch.FloatTensor).to(device).to_dense()

    manifold_coord_gpu = torch.from_numpy(anchor["noahnd_coord"]).float().to(device).unsqueeze(0)

    body_skin_weight = body_model.lbs_weights[anchor["smplx2nohand"]][:, :22] # n_verts x 22
    body_skin_weight = torch.where(body_skin_weight > 0.3, 1., 0.)
    pose_map = get_posemap_custom().float().to(device) # 22 x 21
    pose_map_vert = torch.where(torch.einsum('ij, jk -> ik', body_skin_weight, pose_map) > 0, 1., 0.) # n_verts x 21


    shead2nohand = torch.tensor(anchor["shead2nohand"]).to(device)

    model_residual = MLP_Coarse_res(3, 84).to(device)

    # optimizer = torch.optim.Adam(model_residual.parameters(), cfg.residual_learning_rate)
    params = []
    input_body_pose.requires_grad_(True)
    # params.append(input_body_pose)
    params += list(model_residual.parameters())
    optimizer = torch.optim.Adam(params, cfg.residual_learning_rate)

    n_verts = len(anchor["noahnd_coord"])
    n_all_verts = len(anchor["shead_coord"])
    cfg.shape_batch_size_residual = min(cfg.shape_batch_size_residual, recon_frames)
    epoch_iter = recon_frames // cfg.shape_batch_size_residual
    cfg.rootLogger.info("Iter per Epoch: " + str(epoch_iter))
    #endregion

    # Visualization
    if cfg.VISUALIZE:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(262, lambda vis: upframe()) #
        vis.register_key_callback(263, lambda vis: downframe()) #
        vis.create_window(width=1920, height=1080)

        mesh = o3d.geometry.TriangleMesh()
        with torch.no_grad():
            demo_deformed_verts = LBS_deform(body_model, custom_lbs_weight, template_v_gpu,
                                    body_pose=input_body_pose[demo_frame].unsqueeze(0),
                                    betas=input_betas_1,
                                    global_orient=input_global_orient[demo_frame].unsqueeze(0),
                                    left_hand_pose=input_left_hand_pose[demo_frame].unsqueeze(0),
                                    right_hand_pose=input_right_hand_pose[demo_frame].unsqueeze(0),
                                    expression=expression,
                                    jaw_pose=jaw_pose,
                                    leye_pose=leye_pose,
                                    reye_pose=reye_pose,
                                    return_verts=True)
            demo_deformed_verts = (demo_deformed_verts[0] + input_transl[demo_frame])

        mesh.vertices = o3d.utility.Vector3dVector(demo_deformed_verts.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(template_f)
        mesh.compute_vertex_normals()

        vis.add_geometry(mesh)

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_list[demo_frame].points))
        tmp_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd_list[demo_frame].colors))
        vis.add_geometry(tmp_pcd)

        demo_pos_enc_gpu = manifold_coord_gpu.squeeze()


    iteration_number = 0
    frame_index_table = torch.arange(cfg.shape_batch_size_residual, dtype=torch.int64, device=device).unsqueeze(1).unsqueeze(1)
    pre_anchor_mask = torch.zeros(n_all_verts, dtype=torch.bool, device=device)
    pre_anchor_mask[anchor["pre_anchor_idx"]] = True
    for epoch in range(0, cfg.epochs_residual):
        model_residual.train()
        iteration_number = 0

        start = time()

        get_arr = np.arange(recon_frames)
        np.random.shuffle(get_arr)
        get_arr = torch.from_numpy(get_arr).long().to(device)
        
        for i in range(0, epoch_iter):
                
            pose_code_gpu = matrix_to_quaternion(batch_rodrigues(input_body_pose.reshape(-1, 3))).reshape(recon_frames, 21, 4)
            data_idx_s = i * cfg.shape_batch_size_residual
            data_idx_e = min((i + 1) * cfg.shape_batch_size_residual, recon_frames)

            tmp_tar_p_gpu = tar_p_gpu[get_arr[data_idx_s:data_idx_e]].clone()
            tmp_pose_code_gpu = pose_code_gpu[get_arr[data_idx_s:data_idx_e]].clone().unsqueeze(1).repeat(1, n_verts, 1, 1)

            noise_pose = pose_code_gpu[get_arr[data_idx_s:data_idx_e]].clone()

            if args.add_noise:
                noise = 0.02 - 0.01 * torch.rand(noise_pose.shape[0], noise_pose.shape[1]).to(device)
                noise_axis = 0.1 - 0.05 * torch.rand(noise_pose.shape[0], noise_pose.shape[1], 3).to(device)
                noise_pose[:, :, 3] += noise
                noise_pose[:, :, :3] += noise_axis
            tmp_pose_code_gpu = noise_pose.unsqueeze(1).repeat(1, n_verts, 1, 1)

            tmp_pose_code_gpu = pose_map_vert.unsqueeze(0).unsqueeze(-1) * tmp_pose_code_gpu
            tmp_pose_code_gpu = tmp_pose_code_gpu.reshape(data_idx_e - data_idx_s, -1, 84)
            tmp_template_v_gpu = template_v_gpu.repeat(data_idx_e - data_idx_s, 1, 1)
        
            pos_enc_input = manifold_coord_gpu.repeat(data_idx_e - data_idx_s, 1, 1).reshape(-1, 3)
            pose_code_input = tmp_pose_code_gpu.reshape(-1, 84)

            try:
                residual_pred = model_residual(pos_enc_input, pose_code_input)

            except RuntimeError as e:
                cfg.rootLogger.error("Runtime error!", e)
                cfg.rootLogger.error("Exiting...")
                exit()

            tmp_new_v_cano_gpu = tmp_template_v_gpu
            tmp_new_v_cano_gpu[:, shead2nohand] += residual_pred.reshape(data_idx_e - data_idx_s, n_verts, 3)

            tmp_new_v_posed_gpu = LBS_deform(body_model, custom_lbs_weight, tmp_new_v_cano_gpu,
                                    body_pose=input_body_pose[get_arr[data_idx_s:data_idx_e]],
                                    betas=input_betas_1,
                                    global_orient=input_global_orient[get_arr[data_idx_s:data_idx_e]],
                                    left_hand_pose=input_left_hand_pose[get_arr[data_idx_s:data_idx_e]],
                                    right_hand_pose=input_right_hand_pose[get_arr[data_idx_s:data_idx_e]],
                                    expression=expression.repeat(data_idx_e - data_idx_s, 1),
                                    jaw_pose=jaw_pose.repeat(data_idx_e - data_idx_s, 1),
                                    leye_pose=leye_pose.repeat(data_idx_e - data_idx_s, 1),
                                    reye_pose=reye_pose.repeat(data_idx_e - data_idx_s, 1),
                                    return_verts=True)
            tmp_new_v_posed_gpu = tmp_new_v_posed_gpu + input_transl[get_arr[data_idx_s:data_idx_e]]
            aaa = tmp_new_v_posed_gpu.cpu().detach().numpy()

            new_src_mesh = Meshes(verts=tmp_new_v_posed_gpu, faces=template_f_gpu.unsqueeze(0).repeat(data_idx_e - data_idx_s, 1, 1))
            if args.RGBD:
                new_src_points = sample_points_from_meshes(new_src_mesh, 20000)
            else:
                new_src_points = sample_points_from_meshes(new_src_mesh, 20000)
            # new_src_points = tmp_new_v_posed_gpu

            if args.RGBD:
                dist, _ = chamfer_distancePP_diff(new_src_points, tar_p_gpu[get_arr[data_idx_s:data_idx_e]])
            else:
                dist1, dist2, _, _ = chamfer_distancePP_diff_both(new_src_points, tar_p_gpu[get_arr[data_idx_s:data_idx_e]])

            if args.RGBD:
                dist = tar_p_weight_gpu[get_arr[data_idx_s:data_idx_e]].unsqueeze(-1) * dist
                chamfer_loss = torch.mean(torch.sum(dist ** 2, dim=-1).view(-1)[mask_p_gpu[get_arr[data_idx_s:data_idx_e]].view(-1)])
            else:
                x_y_dist = mask_p_gpu[get_arr[data_idx_s:data_idx_e]].unsqueeze(-1) * dist1
                y_x_dist = dist2
                chamfer_loss = torch.mean(torch.sum(x_y_dist ** 2, dim=-1).view(-1)[mask_p_gpu[get_arr[data_idx_s:data_idx_e]].view(-1)])
                chamfer_loss += torch.mean(torch.sum(y_x_dist ** 2, dim=-1))
                
                
            dist_inv, _ = chamfer_distancePP_diff(tar_p_gpu[get_arr[data_idx_s:data_idx_e]], tmp_new_v_posed_gpu)
            y_nn = knn_points(tar_p_gpu[get_arr[data_idx_s:data_idx_e]], tmp_new_v_posed_gpu, K=1)
            tmp_pre_anchor_mask = torch.zeros(n_all_verts, dtype=torch.bool, device=device).unsqueeze(0).repeat(data_idx_e - data_idx_s, 1)
            for jj in range(data_idx_e - data_idx_s):
                tmp_pre_anchor_mask[jj][torch.unique(y_nn.idx[jj])] = True
            tmp_pre_anchor_mask = torch.logical_and(tmp_pre_anchor_mask, pre_anchor_mask.unsqueeze(0).repeat(data_idx_e - data_idx_s, 1))

            anchor_loss = torch.mean(torch.sum(dist_inv[tmp_pre_anchor_mask]** 2, dim=-1))

            if args.RGBD:
                E_lap = torch.mean(torch.sum(torch.matmul(tmp_new_v_posed_gpu.permute(0, 2, 1), L_gpu) ** 2, dim=1) * torch.median(tar_p_weight_gpu[get_arr[data_idx_s:data_idx_e]], dim = 1)[0].unsqueeze(-1))
            else:
                E_lap = torch.mean(torch.sum(torch.matmul(tmp_new_v_posed_gpu.permute(0, 2, 1), L_gpu) ** 2, dim=1))


            edge_loss = mesh_edge_loss(new_src_mesh)
            
            # loss = weight_vec["chamfer"] * chamfer_loss + weight_vec["lap_reg"] * E_lap + 2 * anchor_loss
            loss = weight_vec["chamfer"] * chamfer_loss + weight_vec["lap_reg"] * E_lap + 0.1 * edge_loss + 2 * anchor_loss
            
            
            optimizer.zero_grad()
            loss.backward()
            # input_body_pose.grad[:, :9] = 0
            # input_body_pose.grad[:, 15:18] = 0
            # input_body_pose.grad[:, 24:27] = 0
            # input_body_pose.grad[:, 36:42] = 0
            # input_body_pose.grad[:, 45:51] = 0
            optimizer.step()

            time_step = time() - start

            # simple evaluation
            if iteration_number % 1 == 0:
                cfg.rootLogger.debug("epoch {0}, iter {1}, chamfer_loss {2}, lap_reg {3}, demo_frame {4}".format(epoch, iteration_number, chamfer_loss.item(), E_lap.item(), demo_frame))
                # cfg.rootLogger.debug("edge loss {}".format(edge_loss))

            iteration_number = iteration_number + 1

            # Visualization
            if cfg.VISUALIZE:
                demo_frame += recon_frames
                demo_frame %= recon_frames
                with torch.no_grad():
                    demo_pose_code_gpu = pose_map_vert.unsqueeze(-1) * pose_code_gpu[demo_frame].clone().unsqueeze(0).repeat(n_verts, 1, 1)
                    demo_pose_code_gpu = demo_pose_code_gpu.reshape(-1, 84)
                    demo_residual_pred = model_residual(demo_pos_enc_gpu, demo_pose_code_gpu)

                    aaa = template_v_gpu.clone()
                    aaa[:, shead2nohand] += demo_residual_pred
                    demo_deformed_verts = LBS_deform(body_model, custom_lbs_weight, aaa,
                                            body_pose=input_body_pose[demo_frame].unsqueeze(0),
                                            betas=input_betas_1,
                                            global_orient=input_global_orient[demo_frame].unsqueeze(0),
                                            left_hand_pose=input_left_hand_pose[demo_frame].unsqueeze(0),
                                            right_hand_pose=input_right_hand_pose[demo_frame].unsqueeze(0),
                                            expression=expression,
                                            jaw_pose=jaw_pose,
                                            leye_pose=leye_pose,
                                            reye_pose=reye_pose,
                                            return_verts=True)
                    demo_deformed_verts = (demo_deformed_verts[0] + input_transl[demo_frame]).cpu().numpy()

                colors = 0.8 * np.ones_like(demo_deformed_verts)
                colors[anchor["pre_anchor_idx"]] = (1,0,0)
                mesh.vertices = o3d.utility.Vector3dVector(demo_deformed_verts)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                mesh.compute_triangle_normals()
                mesh.compute_vertex_normals()
                tmp_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_list[demo_frame].points))
                tmp_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd_list[demo_frame].colors))
                vis.update_geometry(mesh)
                vis.update_geometry(tmp_pcd)
                vis.poll_events()
                vis.update_renderer()

            start = time()
        
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model_residual.state_dict(), os.path.join(target_dir, "net/residual_model" + "_e" + str(epoch) + ".pts"))
            # with torch.no_grad():
            #     params_new = (input_body_pose, input_betas_1, input_global_orient,
            #                     input_transl, input_scale_1, 
            #                     input_left_hand_pose, input_right_hand_pose)
            #     save_params(os.path.join(target_dir, "net/smplx_fit" + "_e" + str(epoch) + ".npz"), params_new)

    torch.save(model_residual.state_dict(), os.path.join(target_dir, "net/residual_model" + "_e" + str(cfg.epochs_residual) + ".pts"))
    # with torch.no_grad():
    #     params_new = (input_body_pose, input_betas_1, input_global_orient,
    #                     input_transl, input_scale_1, 
    #                     input_left_hand_pose, input_right_hand_pose)
    #     save_params(os.path.join(target_dir, "net/smplx_fit" + "_e" + str(cfg.epochs_residual) + ".npz"), params_new)

