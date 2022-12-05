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
import open3d as o3d
import igl
from sklearn.neighbors import KDTree

from time import time
from tqdm import tqdm

from pytorch3d.transforms import matrix_to_quaternion

from lib.LaplCal import get_uniform_laplacian_theta, LapEditor
import config as cfg
from lib.model.mlp import *
from lib.smplx.utils import load_params, get_posemap_custom2, get_posemap_custom
from lib.smplx.lbs import batch_rodrigues
import lib.smplx as smplx

parser = argparse.ArgumentParser()

parser.add_argument("--target_subj", default='hyomin_raise_hand')
parser.add_argument("--target_gender", default='male')
parser.add_argument('--RGBD', default=True, help='Is Point Cloud?')
parser.add_argument('--flathand', default=False)
parser.add_argument('--epoch', default=30)

args = parser.parse_args()

density = 16

def text_3d(text, pos, direction=None, degree=-90.0, font='/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


if __name__ == '__main__':
    cfg.make_dir_structure(args.target_subj)
    cfg.rootLogger.info("Inference pose dependent details")
    target_dir = os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj)

    if cfg.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float32
    density_postfix = "x" + str(density)

    recon_dir = os.path.join(target_dir, "train/sharpening")
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
    params = load_params(os.path.join(target_dir, "train/smplx_fit.npz"), device, dtype)
    (input_body_pose, input_betas_1, input_global_orient,
        input_transl, input_scale_1, 
        input_left_hand_pose, input_right_hand_pose) = params

    anchor = np.load(os.path.join(cfg.DataPath["Main"], cfg.DataPath["Anchor"]))
    n_v = len(anchor["noahnd_coord"])

    S = sp.sparse.load_npz(os.path.join(cfg.DataPath["Main"], "protocol_info/nohand_" + density_postfix + ".npz" ))
    n_dense_v = S.shape[0]

    manifold_coord_gpu = torch.from_numpy(S @ anchor["noahnd_coord"]).float().to(device)
    custom_lbs_weight = torch.tensor(S @ body_model.lbs_weights[anchor["smplx2nohand"]].cpu().numpy(), dtype=dtype, device=device)

    pose_map = get_posemap_custom2().float().to(device)
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
    ppcd = mmesh.sample_points_poisson_disk(5000)
    # ppcd = mmesh.sample_points_poisson_disk(density * 150)
    ttree = KDTree(anchor["shead_coord"])
    _, nn_idx = ttree.query(np.asarray(ppcd.points))
    picked_points = nn_idx[:, 0]

    # hand_foot = np.load("/NVME/dataset/CAPE/core_body/hand_foot.npy")
    # picked_points = np.setdiff1d(picked_points, hand_foot)

    hand_foot_v_idx = np.array([5213, 4788, 4914, 5009, 5126, 2283, 1856, 1982, 2094, 2195, 5586, 5589, 5675, 5599, 5596, 5756, 2697, 2780, 2694, 2792, 2797, 2867, 2918, 2882, 2849, 5738, 5775, 5813])
    picked_points = np.concatenate((picked_points, hand_foot_v_idx), axis = 0)
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

    step = 100
    demo_frame = 43

    recon_vertex_t = np.zeros((step, shead_S.shape[0], 3))
    for i in tqdm(range(step), desc="Laplacian reconstruction"):
        M_coarse = igl.massmatrix(coarse_verts[demo_frame], f, igl.MASSMATRIX_TYPE_VORONOI)
        m_new = (0.5 + i * (1.5 / (step - 1))) * (S @ np.squeeze(np.asarray(M_coarse.sum(1)))) / density

        delta_smpl = L_dense @ shead_S @ coarse_vertex_w_hand[demo_frame]

        tmp_delta_new = mapped_delta[demo_frame] * m_new[:, np.newaxis]

        delta_new = delta_smpl
        delta_new[anchor["shead2nohand_" + density_postfix]] = tmp_delta_new

        tmp_coarse_verts = anchor["shead_coord"]
        tmp_coarse_verts[anchor["shead2nohand"]] = coarse_verts[demo_frame]
        tmp_coarse_verts[hand_foot_v_idx] = coarse_vertex_w_hand[demo_frame][hand_foot_v_idx]

        lap_editor.update_anchor(tmp_coarse_verts, picked_points)

        v_new = lap_editor.recon(delta_new)
        recon_vertex_t[i] = v_new
    #endregion


    # Visualization
    if cfg.VISUALIZE:
        cfg.rootLogger.info("View")
        from lib.O3D_NB_Vis import o3d_nb_vis

        pcd_list_o3d = []
        init_pos = np.mean(recon_vertex_t[0], axis=0)
        for frame_idx in range(step):
            pcd_list_o3d.append(text_3d("Original X {:.3f}".format((0.5 + frame_idx * (1.5 / (step - 1)))), init_pos + np.array([-0.5,1,0])))

        o3d_nb_vis({"Mesh0" : {"vertices":recon_vertex_t, "triangles": anchor["shead_" + density_postfix + "_tri"]},
                    "O3D_PCD0" : {"pcd":pcd_list_o3d}})

    # Save
    # for t in tqdm(range(recon_frames), desc="Save"):
    #     basename = os.path.basename(coarse_files[t])
    #     filename_recon = os.path.join(recon_dir, os.path.splitext(basename)[0] + ".posed.ply")
    #     tmp_mesh = o3d.geometry.TriangleMesh()
    #     tmp_mesh.vertices = o3d.utility.Vector3dVector(recon_vertex_t[t])
    #     tmp_mesh.triangles = o3d.utility.Vector3iVector(anchor["shead_" + density_postfix + "_tri"])
    #     tmp_mesh.compute_vertex_normals()
    #     o3d.io.write_triangle_mesh(filename_recon, tmp_mesh)


