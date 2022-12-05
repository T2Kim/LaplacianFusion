import torch

from lib.human_fitting.utils_smplify import GMoF, rel_change
from lib.LaplCal import get_uniform_laplacian_1
from lib.chamferdist import chamfer_distancePP_diff
from lib.smplx.lbs import batch_rodrigues

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.structures import Meshes
import config as cfg

#joint_correspondence_idx_map_openpose_to_smpl

j_map_o2s = {
    3: 19,
    6: 18,
    10: 5,
    13: 4,
    4: 21,
    7: 20,
    2: 17,
    5: 16,
    9: 2,
    12: 1,
    16: 23,
    15: 24,
}

open_joint = {
    'head': 0,
    'neck': 1,
    'right_shoulder': 2,
    'right_elbow': 3,
    'right_wrist': 4,
    'left_shoulder': 5,
    'left_elbow': 6,
    'left_wrist': 7,
    #'pelvis': 8,
    'right_hip': 9,
    'right_knee': 10,
    'right_ankle': 11,
    'left_hip': 12,
    'left_knee': 13,
    'left_ankle': 14,
    'right_eye': 15,
    'left_eye': 16,
}

smplx_face_verts = {
    'nose':		    9120,
    'reye':		    9929,
    'leye':		    9448,
    'rear':		    616,
    'lear':		    6
}
open_face_joint = {
    'nose': 30,
    'reye': 68,
    'leye': 69
}

dvm_map = {
    'head': list(range(0,13)),
    'neck': list(range(13,17)),
    'thorax': list(range(17,19)),
    'right_shoulder': list(range(19,22)),
    'left_shoulder': list(range(22,25)),
    'right_upper_arm': list(range(25,31)),
    'right_lower_arm': list(range(31,37)),
    'right_hand': list(range(37,38)),
    'left_upper_arm': list(range(38,44)),
    'left_lower_arm': list(range(44,50)),
    'left_hand': list(range(50,51)),
    'upper_body': list(range(51,59)),
    'lower_body': list(range(59,67)),
    'pelvis': list(range(67,75)),
    'right_upper_leg': list(range(75,81)),
    'right_lower_leg': list(range(81,87)),
    'right_foot': list(range(87,93)),
    'left_upper_leg': list(range(93,99)),
    'left_lower_leg': list(range(99,105)),
    'left_foot': list(range(105,111)),
    'all': list(range(37)) + list(range(38,50)) + list(range(51,111)),
}

# Left, Right
id_mapper_smpl_finger_thumb = [37,38,39,66]+[52,53,54,71]
id_mapper_smpl_finger_index = [25,26,27,67]+[40,41,42,72]
id_mapper_smpl_finger_middle = [28,29,30,68]+[43,44,45,73]
id_mapper_smpl_finger_pinky = [31,35,36,69]+[49,50,51,74] 
id_mapper_smpl_finger_ring = [34,32,33,70]+[46,47,48,75]

id_mapper_openpose_finger_thumb = [1,2,3,4,1+21,2+21,3+21,4+21]
id_mapper_openpose_finger_index = [5,6,7,8,5+21,6+21,7+21,8+21]
id_mapper_openpose_finger_middle = [9,10,11,12,9+21,10+21,11+21,12+21]
id_mapper_openpose_finger_pinky = [13,14,15,16,13+21,14+21,15+21,16+21]
id_mapper_openpose_finger_ring = [17,18,19,20,17+21,18+21,19+21,20+21]


def create_optimizer(input_params, step):
    (input_body_pose, input_betas, input_global_orient,
     input_transl, input_scale, 
     input_left_hand_pose, input_right_hand_pose) = input_params

    params = []
    if step == 'orient+transl':
        params.append(input_betas)
        params.append(input_global_orient)
        params.append(input_transl)
    elif step == 'orient+knee':
        params.append(input_global_orient)
        params.append(input_transl)
    elif step == 'orient+transl+scale':
        params.append(input_global_orient)
        params.append(input_transl)
        params.append(input_betas)
        params.append(input_scale)
    elif step == 'shoulder':
        params.append(input_transl)
        params.append(input_body_pose)
        # params.append(input_betas)
    elif step == 'elbow-L':
        params.append(input_body_pose)
        # params.append(input_betas)
    elif step == 'elbow-R':
        params.append(input_body_pose)
        # params.append(input_betas)
    elif step == 'wrist':
        params.append(input_body_pose)
        # params.append(input_betas)
    elif step == 'fingers':
        params.append(input_body_pose)
        # params.append(input_betas)
        params.append(input_left_hand_pose)
        params.append(input_right_hand_pose)
    elif step == 'local_arm_pose':
        params.append(input_body_pose)
        # params.append(input_betas)
        params.append(input_left_hand_pose)
        params.append(input_right_hand_pose)
    elif step == 'knee':
        params.append(input_body_pose)
        # params.append(input_betas)
    elif step == 'ankle':
        params.append(input_body_pose)
        # params.append(input_betas)
    elif step == 'toe':
        params.append(input_body_pose)
        # params.append(input_betas)
    elif step == 'local_leg_pose':
        params.append(input_body_pose)
        # params.append(input_betas)
    elif step == 'face':
        params.append(input_body_pose)
    elif step == 'global_pose':
        params.append(input_body_pose)
        params.append(input_betas)
        params.append(input_global_orient)
        params.append(input_transl)
        params.append(input_scale)
        # params.append(input_left_hand_pose)
        # params.append(input_right_hand_pose)
    elif step == 'global_pose_OP':
        params.append(input_body_pose)
        params.append(input_betas)
        params.append(input_global_orient)
        params.append(input_transl)
        params.append(input_scale)
        params.append(input_left_hand_pose)
        params.append(input_right_hand_pose)
    elif step == 'global_pose_P2P':
        params.append(input_body_pose)
        params.append(input_betas)
        params.append(input_global_orient)
        params.append(input_left_hand_pose)
        params.append(input_right_hand_pose)
    elif step == 'betas':
        params.append(input_betas)
    elif step == 'fingers+betas':
        params.append(input_body_pose)
        params.append(input_betas)
        params.append(input_left_hand_pose)
        params.append(input_right_hand_pose)
    else:
        raise ValueError(f'step {step} is not found !')

    optimizer = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=30,
        line_search_fn='strong_wolfe')

    return optimizer, params


def create_optimizer_CT(input_params, step):
    (input_body_pose, input_betas, input_global_orient,
     input_transl, input_scale, 
     input_left_hand_pose, input_right_hand_pose) = input_params

    params = []
    input_transl[:, :, 2] -= 3.0
    input_transl.requires_grad_(True)
    params.append(input_transl)
    optimizer = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=50,
        line_search_fn='strong_wolfe')

    return optimizer, params



def create_optimizer_CAPE(input_params):
    (input_body_pose, input_betas, input_global_orient,
     input_transl) = input_params

    params = []
    params.append(input_body_pose)
    params.append(input_betas)
    params.append(input_global_orient)
    params.append(input_transl)

    optimizer = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=30,
        line_search_fn='strong_wolfe')

    return optimizer, params


def create_optimizer_Resynth(input_params):
    (input_body_pose, input_betas, input_global_orient,
     input_transl, input_scale, 
     input_left_hand_pose, input_right_hand_pose) = input_params


    params = []
    params.append(input_body_pose)
    params.append(input_betas)
    params.append(input_global_orient)
    params.append(input_transl)

    optimizer = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=30,
        line_search_fn='strong_wolfe')

    return optimizer, params


def create_optimizer_NeuroGIF(input_params):
    (input_body_pose, input_betas, input_global_orient,
     input_transl, input_scale, 
     input_left_hand_pose, input_right_hand_pose) = input_params


    params = []
    params.append(input_betas)
    params.append(input_transl)
    params.append(input_global_orient)
    params.append(input_body_pose)

    optimizer = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=30,
        line_search_fn='strong_wolfe')

    return optimizer, params


import open3d as o3d
import numpy as np
demo_frame = 0
g_refresh = False
g_initialized = False
g_vis = o3d.visualization.VisualizerWithKeyCallback()

def upframe():
    global demo_frame, g_refresh
    demo_frame += 1
    print(demo_frame)
    g_refresh = True
def downframe():
    global demo_frame, g_refresh
    demo_frame -= 1
    print(demo_frame)
    g_refresh = True

def create_fitting_closure(optimizer,
                           body_model, 
                           input_params,
                           priors, 
                           step,
                           visualize=True):
    global g_vis, g_initialized

    robustifier = GMoF(rho=100)

    (input_body_pose, input_betas_1, input_global_orient,
     input_transl, input_scale_1, 
     input_left_hand_pose, input_right_hand_pose) = input_params

    prev_input_body_pose = torch.clone(input_body_pose.detach())
    prev_input_betas = torch.clone(input_betas_1.detach())
    prev_input_global_orient = torch.clone(input_global_orient.detach())
    prev_input_transl = torch.clone(input_transl.detach())
    prev_input_scale = torch.clone(input_scale_1.detach())
    prev_input_left_hand_pose = torch.clone(input_left_hand_pose.detach())
    prev_input_right_hand_pose = torch.clone(input_right_hand_pose.detach())

    recon_frames = input_body_pose.shape[0]

    if visualize:
        g_vis.clear_geometries()
        smplx_mesh = o3d.geometry.TriangleMesh()
        smplx_mesh.triangles = o3d.utility.Vector3iVector(body_model.faces)
        with torch.no_grad():
            input_betas_tmp = torch.ones((recon_frames, input_betas_1.shape[-1]), dtype = input_betas_1.dtype, device=input_betas_1.device)
            input_betas = input_betas_tmp * input_betas_1
            input_scale_tmp = torch.ones((recon_frames, 1, 1), dtype = input_scale_1.dtype, device=input_scale_1.device)
            input_scale = input_scale_tmp * input_scale_1

            body_model_output = body_model(body_pose=input_body_pose,
                                        betas=input_betas,
                                        global_orient=input_global_orient,
                                        left_hand_pose=input_left_hand_pose,
                                        right_hand_pose=input_right_hand_pose,
                                        return_verts=True)
            verts = body_model_output.vertices * input_scale + input_transl
        smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
        smplx_mesh.compute_triangle_normals()
        smplx_mesh.compute_vertex_normals()
        ccc = np.ones_like(np.asarray(smplx_mesh.vertices)) * 0.8
        ccc[priors["smplx_dvm_sparse"].cpu().numpy(), :] = [1,0,0]
        smplx_mesh.vertex_colors = o3d.utility.Vector3dVector(ccc)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(priors["pcd_list_o3d"][demo_frame].points))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(priors["pcd_list_o3d"][demo_frame].colors))
        # pcd.paint_uniform_color([0.8,0,0.8])
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        pcd_marker = o3d.geometry.PointCloud()
        valid_marker = torch.where(priors["tar_dvm_sparse"][demo_frame, :, 3] > 0)[0]
        pcd_marker.points = o3d.utility.Vector3dVector(priors["tar_dvm_sparse"][demo_frame, valid_marker, :3].cpu().numpy())
        pcd_marker.paint_uniform_color([0,1,0])

        if not g_initialized:
            g_vis.create_window(width=1920, height=1080)
            g_vis.register_key_callback(262, lambda vis: upframe()) #
            g_vis.register_key_callback(263, lambda vis: downframe()) #
            g_initialized = True
        g_vis.add_geometry(smplx_mesh)
        g_vis.add_geometry(pcd)
        g_vis.add_geometry(pcd_marker)
        g_vis.poll_events()
        g_vis.update_renderer()

    OP_weight = priors["OP_weight"]
    P2P_weight = priors["P2P_weight"]
    dvm_weight = priors["DVM_weight"]

    template_f_gpu_long = torch.from_numpy(body_model.faces.astype(np.int64)).to(input_body_pose.device)
    template_v_len_gpu = (torch.ones(recon_frames) * recon_frames).long().to(input_body_pose.device)

    def fitting_func(backward=True):
        global demo_frame, g_refresh, g_vis
        if backward:
            optimizer.zero_grad()

        input_betas_tmp = torch.ones((recon_frames, input_betas_1.shape[-1]), dtype = input_betas_1.dtype, device=input_betas_1.device)
        input_betas = input_betas_tmp * input_betas_1
        input_scale_tmp = torch.ones((recon_frames, 1, 1), dtype = input_scale_1.dtype, device=input_scale_1.device)
        input_scale = input_scale_tmp * input_scale_1

        body_model_output = body_model(body_pose=input_body_pose,
                                       betas=input_betas,
                                       global_orient=input_global_orient,
                                       left_hand_pose=input_left_hand_pose,
                                       right_hand_pose=input_right_hand_pose,
                                       return_verts=True)

        j_tr = body_model_output.joints + input_transl
        verts = body_model_output.vertices + input_transl

        if visualize:
            demo_frame += recon_frames
            demo_frame %= recon_frames
            with torch.no_grad():
                smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
                smplx_mesh.compute_triangle_normals()
                smplx_mesh.compute_vertex_normals()

                if g_refresh:
                    pcd.points = o3d.utility.Vector3dVector(np.asarray(priors["pcd_torch"][demo_frame].cpu().numpy()))
                    pcd.paint_uniform_color([0.8,0,0.8])
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
                    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
                    g_vis.update_geometry(pcd)
                    valid_marker = torch.where(priors["tar_dvm_sparse"][demo_frame, :, 3] > 0)[0]
                    pcd_marker.points = o3d.utility.Vector3dVector(priors["tar_dvm_sparse"][demo_frame, valid_marker, :3].cpu().numpy())
                    pcd_marker.paint_uniform_color([0,1,0])
                    g_vis.update_geometry(pcd_marker)

            g_vis.update_geometry(smplx_mesh)

            g_vis.poll_events()
            g_vis.update_renderer()

        betas_reg_prior = torch.sum(input_betas ** 2)
        body_pose_reg_prior = torch.sum(input_body_pose ** 2)
        head_pose_reg_prior = torch.sum(input_body_pose[:, 33:36] ** 2 + input_body_pose[:, 42:45] ** 2, dim=1)
        hand_pose_reg_prior = torch.sum(input_left_hand_pose**2 + input_right_hand_pose**2)
        temp_smooth_reg_prior = torch.sum(robustifier(input_body_pose[1:-1] - input_body_pose[2:])) + torch.sum(robustifier(input_body_pose[1:-1] - input_body_pose[:-2]))
        # temp_glo_smooth_reg_prior = torch.sum(robustifier(input_global_orient[1:-1] - input_global_orient[2:])) + torch.sum(robustifier(input_global_orient[1:-1] - input_global_orient[:-2]))
        temp_glo_smooth_reg_prior = torch.sum(robustifier(torch.norm(matrix_to_quaternion(batch_rodrigues(2 * input_global_orient[1:-1])) - matrix_to_quaternion(batch_rodrigues(input_global_orient[2:])) - matrix_to_quaternion(batch_rodrigues(input_global_orient[:-2])), dim=-1)))
        temp_glo_trans_reg_prior = 1000 * (torch.sum(robustifier(input_transl[1:-1] - input_transl[2:])) + torch.sum(robustifier(input_transl[1:-1] - input_transl[:-2])))

        body_pose_preserve_prior = ((prev_input_body_pose - input_body_pose) ** 2).sum()
        betas_preserve_prior = ((prev_input_betas - input_betas) ** 2).sum()
        global_orient_preserve_prior = ((prev_input_global_orient - input_global_orient) ** 2).sum()
        transl_preserve_prior = ((prev_input_transl - input_transl) ** 2).sum()
        scale_preserve_prior = ((prev_input_scale - input_scale) ** 2).sum()
        left_hand_pose_preserve_prior = ((prev_input_left_hand_pose - input_left_hand_pose) ** 2).sum()
        right_hand_pose_preserve_prior = ((prev_input_right_hand_pose - input_right_hand_pose) ** 2).sum()

        prior_loss = torch.zeros(1).to(verts.device)
        chamfer_loss = torch.zeros(1).to(verts.device)
        OP_loss = torch.zeros(1).to(verts.device)
        face_loss = torch.zeros(1).to(verts.device)
        head_pose_reg = torch.zeros(1).to(verts.device)
        
        if step != 'orient+transl':
            dist_normal = chamfer_distancePP_diff(verts, priors["pcd_torch"], template_v_len_gpu, priors["pcd_len_torch"])
            x_y_dist = priors["mask_torch"].unsqueeze(-1) * dist_normal[0]
            chamfer_loss = torch.sum(robustifier(x_y_dist))
            
            prior_loss += P2P_weight * chamfer_loss

        if step == 'global_pose' or step == 'global_pose_OP' or step == 'global_pose_P2P':

            id_mapper_openpose = [
                open_joint['left_hip'], open_joint['right_hip'],
                open_joint['left_knee'], open_joint['right_knee'],
                open_joint['left_shoulder'], open_joint['right_shoulder'],
            ]
            id_mapper_smpl = [j_map_o2s[jidx] for jidx in id_mapper_openpose]

            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["body_keypts"][:, id_mapper_openpose, :2] * joints_cv[:, :, 2].unsqueeze(-1) - joints_cv[:, :, :2]
                OP_diff = OP_diff * priors["body_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)

            # Face
            id_mapper_smpl=[15, 23, 24, 58, 59]
            id_mapper_openpose=[0, 16, 15, 17, 18]
            
            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["body_keypts"][:, id_mapper_openpose, :2] * joints_cv[:, :, 2].unsqueeze(-1) - joints_cv[:, :, :2]
                OP_diff = OP_diff * priors["body_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)

            # Elbow
            id_mapper_smpl=[18]
            id_mapper_openpose=[6]
            
            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["body_keypts"][:, id_mapper_openpose, :2] * joints_cv[:, :, 2].unsqueeze(-1) - joints_cv[:, :, :2]
                OP_diff = OP_diff * priors["body_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)

            id_mapper_smpl=[19]
            id_mapper_openpose=[3]

            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["body_keypts"][:, id_mapper_openpose, :2] * joints_cv[:, :, 2].unsqueeze(-1) - joints_cv[:, :, :2]
                OP_diff = OP_diff * priors["body_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)

            # Wrist
            id_mapper_openpose = [
                open_joint['left_wrist'], open_joint['right_wrist']]
            id_mapper_smpl = [j_map_o2s[jidx] for jidx in id_mapper_openpose]
            
            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["body_keypts"][:, id_mapper_openpose, :2] * joints_cv[:, :, 2].unsqueeze(-1) - joints_cv[:, :, :2]
                OP_diff = OP_diff * priors["body_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)
            
            # Finger
            id_mapper_smpl = \
                (id_mapper_smpl_finger_thumb +
                id_mapper_smpl_finger_index +
                id_mapper_smpl_finger_middle +
                id_mapper_smpl_finger_pinky +
                id_mapper_smpl_finger_ring)
            id_mapper_openpose = \
                (id_mapper_openpose_finger_thumb +
                id_mapper_openpose_finger_index +
                id_mapper_openpose_finger_middle +
                id_mapper_openpose_finger_pinky +
                id_mapper_openpose_finger_ring)

            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["hand_keypts"][:, id_mapper_openpose, :2] * joints_cv[:, :, 2].unsqueeze(-1) - joints_cv[:, :, :2]
                OP_diff = OP_diff * priors["hand_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)

            # face align
            if priors["use_RGBD"]:
                id_mapper_smpl= [9120, 9929, 9448]
                id_mapper_openpose= [30, 68, 69]

                joints_cv = torch.einsum('bki, ij -> bkj', verts[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=verts.dtype, device=verts.device))
                OP_diff = priors["face_keypts"][:, id_mapper_openpose, :2] * joints_cv[:, :, 2].unsqueeze(-1) - joints_cv[:, :, :2]
                OP_diff = OP_diff * priors["face_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss = torch.sum(OP_diff)
                face_loss = OP_loss

            id_mapper_dvm = dvm_map['all']

            dvm_diff = priors["tar_dvm_sparse"][:, :, :3] - verts[:, priors["smplx_dvm_sparse"]]
            dvm_diff = dvm_diff * priors["tar_dvm_sparse"][:, :, 3].unsqueeze(-1) # valid mask
            dvm_diff = dvm_diff[:, id_mapper_dvm, :]
            dvm_diff = robustifier(dvm_diff)
            dvm_loss = torch.sum(dvm_diff)

            prior_loss += dvm_weight * dvm_loss


            if priors["use_RGBD"] and False:
                head_pose_reg = torch.sum(priors["face_valid"] * head_pose_reg_prior)
            
            total_loss = prior_loss + \
                0.0001*body_pose_preserve_prior + \
                0.00001*betas_preserve_prior + \
                0.0001*global_orient_preserve_prior + \
                0.0001*transl_preserve_prior + \
                0.0001*scale_preserve_prior + \
                priors["pose_reg_weight"]*body_pose_reg_prior + \
                priors["temp_reg_weight"]*temp_smooth_reg_prior + \
                priors["glo_temp_reg_weight"]*temp_glo_smooth_reg_prior + \
                priors["glo_temp_reg_weight"]*temp_glo_trans_reg_prior + \
                0.0001*betas_reg_prior \
                # +0.001*head_pose_reg+\
                # 1*face_loss


        elif step == 'orient+transl':
            id_mapper_dvm = dvm_map['left_shoulder'] + dvm_map['right_shoulder'] + dvm_map['thorax'] + dvm_map['upper_body'] + dvm_map['lower_body']  + dvm_map['pelvis'] 

            dvm_diff = priors["tar_dvm_sparse"][:, :, :3] - verts[:, priors["smplx_dvm_sparse"]]
            dvm_diff = dvm_diff * priors["tar_dvm_sparse"][:, :, 3].unsqueeze(-1) # valid mask
            dvm_diff = dvm_diff[:, id_mapper_dvm, :]
            dvm_diff = robustifier(dvm_diff)
            dvm_loss = torch.sum(dvm_diff)

            prior_loss += dvm_weight * dvm_loss
            
            total_loss = prior_loss + \
                0.0 *temp_glo_smooth_reg_prior + \
                0.001 * global_orient_preserve_prior + \
                0.001 * transl_preserve_prior + \
                0.001 * betas_preserve_prior
        
        else:
            raise ValueError(f'step {step} is not found in clolsure!')

        if priors["use_RGBD"]:
            cfg.rootLogger.debug("All_Loss {0}, prior {1},chamfer {2}, dvm {3}, op_pose {4}".format(total_loss.item(), prior_loss.item(), chamfer_loss.item(), dvm_loss.item(), OP_loss.item()))
            cfg.rootLogger.debug("{0}, {1}, {2}".format(temp_glo_smooth_reg_prior.item(), temp_smooth_reg_prior.item(), temp_glo_trans_reg_prior.item()))
        else:
            cfg.rootLogger.debug("All_Loss {0}, prior {1},chamfer {2}, dvm {3}".format(total_loss.item(), prior_loss.item(), chamfer_loss.item(), dvm_loss.item()))
        # print("pose_reg {0}, temp_reg {1}".format(body_pose_reg_prior.item(), temp_smooth_reg_prior.item()))
        # print("head_pose_reg {0}, face_loss {1}".format(head_pose_reg.item(), face_loss.item()))

        if backward:
            total_loss.backward(create_graph=False)

            if step == 'orient+transl':
                input_betas_1.grad[0, 1:] = 0
            elif step == 'orient+knee':
                pass
            elif step == 'orient+transl+scale':
                pass
            elif step == 'shoulder':
                input_body_pose.grad[:, :36] = 0
                input_body_pose.grad[:, 42:] = 0
            elif step == 'elbow-L':
                input_body_pose.grad[:, :45] = 0
                input_body_pose.grad[:, 48:] = 0
            elif step == 'elbow-R':
                input_body_pose.grad[:, :48] = 0
                input_body_pose.grad[:, 51:] = 0
            elif step == 'wrist': # wrist
                input_body_pose.grad[:, :51] = 0
                input_body_pose.grad[:, 57:] = 0
            elif step == 'fingers': # finger-3 
                input_body_pose.grad[:, :57] = 0
                input_body_pose.grad[:, 63:] = 0
            elif step == 'local_arm_pose':
                input_body_pose.grad[:, :45] = 0
                input_body_pose.grad[:, 63:] = 0
            elif step == 'knee': # knee
                input_body_pose.grad[:, 6:] = 0
            elif step == 'ankle': # ankle
                input_body_pose.grad[:, :9] = 0
                input_body_pose.grad[:, 15:] = 0
            elif step == 'toe': # toe
                input_body_pose.grad[:, :18] = 0
                input_body_pose.grad[:, 24:] = 0
            elif step == 'local_leg_pose':
                input_body_pose.grad[:, :9] = 0
                input_body_pose.grad[:, 15:18] = 0
                input_body_pose.grad[:, 24:] = 0
            elif step == 'face':
                input_body_pose.grad[:, :33] = 0
                input_body_pose.grad[:, 36:42] = 0
                input_body_pose.grad[:, 45:] = 0
            elif step == 'global_pose' or step == 'global_pose_OP' or step == 'global_pose_P2P':
                pass
            elif step == 'betas': 
                pass
            elif step == 'fingers+betas': # finger-3 
                input_body_pose.grad[:, :57] = 0
                input_body_pose.grad[:, 63:] = 0

            else:
                raise ValueError('No step in backward')

        return total_loss

    return fitting_func



def create_fitting_closure_CT(optimizer,
                           body_model, 
                           input_params,
                           priors, 
                           step,
                           visualize=True):
    global g_vis, g_initialized

    robustifier = GMoF(rho=100)

    (input_body_pose, input_betas_1, input_global_orient,
     input_transl, input_scale_1, 
     input_left_hand_pose, input_right_hand_pose) = input_params

    prev_input_body_pose = torch.clone(input_body_pose.detach())
    prev_input_betas = torch.clone(input_betas_1.detach())
    prev_input_global_orient = torch.clone(input_global_orient.detach())
    prev_input_transl = torch.clone(input_transl.detach())
    prev_input_scale = torch.clone(input_scale_1.detach())
    prev_input_left_hand_pose = torch.clone(input_left_hand_pose.detach())
    prev_input_right_hand_pose = torch.clone(input_right_hand_pose.detach())

    recon_frames = input_body_pose.shape[0]

    if visualize:
        g_vis.clear_geometries()
        smplx_mesh = o3d.geometry.TriangleMesh()
        smplx_mesh.triangles = o3d.utility.Vector3iVector(body_model.faces)
        with torch.no_grad():
            input_betas_tmp = torch.ones((recon_frames, 10), dtype = input_betas_1.dtype, device=input_betas_1.device)
            input_betas = input_betas_tmp * input_betas_1
            input_scale_tmp = torch.ones((recon_frames, 1, 1), dtype = input_scale_1.dtype, device=input_scale_1.device)
            input_scale = input_scale_tmp * input_scale_1

            body_model_output = body_model(body_pose=input_body_pose,
                                        betas=input_betas,
                                        global_orient=input_global_orient,
                                        left_hand_pose=input_left_hand_pose,
                                        right_hand_pose=input_right_hand_pose,
                                        return_verts=True)
            verts = body_model_output.vertices * input_scale + input_transl
        smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
        smplx_mesh.compute_triangle_normals()
        smplx_mesh.compute_vertex_normals()
        ccc = np.ones_like(np.asarray(smplx_mesh.vertices)) * 0.8
        ccc[priors["smplx_dvm_sparse"].cpu().numpy(), :] = [1,0,0]
        smplx_mesh.vertex_colors = o3d.utility.Vector3dVector(ccc)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(priors["pcd_list_o3d"][demo_frame].points))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(priors["pcd_list_o3d"][demo_frame].colors))
        # pcd.paint_uniform_color([0.8,0,0.8])
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        pcd_marker = o3d.geometry.PointCloud()
        valid_marker = torch.where(priors["tar_dvm_sparse"][demo_frame, :, 3] > 0)[0]
        pcd_marker.points = o3d.utility.Vector3dVector(priors["tar_dvm_sparse"][demo_frame, valid_marker, :3].cpu().numpy())
        pcd_marker.paint_uniform_color([0,1,0])

        if not g_initialized:
            g_vis.create_window(width=1920, height=1080)
            g_vis.register_key_callback(262, lambda vis: upframe()) #
            g_vis.register_key_callback(263, lambda vis: downframe()) #
            g_initialized = True
        g_vis.add_geometry(smplx_mesh)
        g_vis.add_geometry(pcd)
        g_vis.add_geometry(pcd_marker)
        g_vis.poll_events()
        g_vis.update_renderer()

    OP_weight = priors["OP_weight"]
    P2P_weight = priors["P2P_weight"]
    dvm_weight = priors["DVM_weight"]

    template_f_gpu_long = torch.from_numpy(body_model.faces.astype(np.int64)).to(input_body_pose.device)
    template_v_len_gpu = (torch.ones(recon_frames) * recon_frames).long().to(input_body_pose.device)

    def fitting_func(backward=True):
        global demo_frame, g_refresh, g_vis
        if backward:
            optimizer.zero_grad()

        input_betas_tmp = torch.ones((recon_frames, 10), dtype = input_betas_1.dtype, device=input_betas_1.device)
        input_betas = input_betas_tmp * input_betas_1
        input_scale_tmp = torch.ones((recon_frames, 1, 1), dtype = input_scale_1.dtype, device=input_scale_1.device)
        input_scale = input_scale_tmp * input_scale_1

        body_model_output = body_model(body_pose=input_body_pose,
                                       betas=input_betas,
                                       global_orient=input_global_orient,
                                       left_hand_pose=input_left_hand_pose,
                                       right_hand_pose=input_right_hand_pose,
                                       return_verts=True)

        j_tr = body_model_output.joints + input_transl
        verts = body_model_output.vertices + input_transl

        if visualize:
            demo_frame += recon_frames
            demo_frame %= recon_frames
            with torch.no_grad():
                smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
                smplx_mesh.compute_triangle_normals()
                smplx_mesh.compute_vertex_normals()

                if g_refresh:
                    pcd.points = o3d.utility.Vector3dVector(np.asarray(priors["pcd_torch"][demo_frame].cpu().numpy()))
                    pcd.paint_uniform_color([0.8,0,0.8])
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
                    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
                    g_vis.update_geometry(pcd)
                    valid_marker = torch.where(priors["tar_dvm_sparse"][demo_frame, :, 3] > 0)[0]
                    pcd_marker.points = o3d.utility.Vector3dVector(priors["tar_dvm_sparse"][demo_frame, valid_marker, :3].cpu().numpy())
                    pcd_marker.paint_uniform_color([0,1,0])
                    g_vis.update_geometry(pcd_marker)

            g_vis.update_geometry(smplx_mesh)

            g_vis.poll_events()
            g_vis.update_renderer()

        betas_reg_prior = torch.sum(input_betas ** 2)
        body_pose_reg_prior = torch.sum(input_body_pose ** 2)
        head_pose_reg_prior = torch.sum(input_body_pose[:, 33:36] ** 2 + input_body_pose[:, 42:45] ** 2, dim=1)
        hand_pose_reg_prior = torch.sum(input_left_hand_pose**2 + input_right_hand_pose**2)
        temp_smooth_reg_prior = torch.sum(robustifier(input_body_pose[1:-1] - input_body_pose[2:])) + torch.sum(robustifier(input_body_pose[1:-1] - input_body_pose[:-2]))
        # temp_glo_smooth_reg_prior = torch.sum(robustifier(input_global_orient[1:-1] - input_global_orient[2:])) + torch.sum(robustifier(input_global_orient[1:-1] - input_global_orient[:-2]))
        temp_glo_smooth_reg_prior = torch.sum(robustifier(torch.norm(matrix_to_quaternion(batch_rodrigues(2 * input_global_orient[1:-1])) - matrix_to_quaternion(batch_rodrigues(input_global_orient[2:])) - matrix_to_quaternion(batch_rodrigues(input_global_orient[:-2])), dim=-1)))
        temp_glo_trans_reg_prior = 1000 * (torch.sum(robustifier(input_transl[1:-1] - input_transl[2:])) + torch.sum(robustifier(input_transl[1:-1] - input_transl[:-2])))

        body_pose_preserve_prior = ((prev_input_body_pose - input_body_pose) ** 2).sum()
        betas_preserve_prior = ((prev_input_betas - input_betas) ** 2).sum()
        global_orient_preserve_prior = ((prev_input_global_orient - input_global_orient) ** 2).sum()
        transl_preserve_prior = ((prev_input_transl - input_transl) ** 2).sum()
        scale_preserve_prior = ((prev_input_scale - input_scale) ** 2).sum()
        left_hand_pose_preserve_prior = ((prev_input_left_hand_pose - input_left_hand_pose) ** 2).sum()
        right_hand_pose_preserve_prior = ((prev_input_right_hand_pose - input_right_hand_pose) ** 2).sum()

        prior_loss = torch.zeros(1).to(verts.device)
        chamfer_loss = torch.zeros(1).to(verts.device)
        OP_loss = torch.zeros(1).to(verts.device)
        face_loss = torch.zeros(1).to(verts.device)
        head_pose_reg = torch.zeros(1).to(verts.device)
        
        if step == 'global_pose' or step == 'global_pose_OP' or step == 'global_pose_P2P':

            id_mapper_openpose = [
                open_joint['left_hip'], open_joint['right_hip'],
                open_joint['left_knee'], open_joint['right_knee'],
                open_joint['left_shoulder'], open_joint['right_shoulder'],
            ]
            id_mapper_smpl = [j_map_o2s[jidx] for jidx in id_mapper_openpose]

            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["body_keypts"][:, id_mapper_openpose, :2] - joints_cv[:, :, :2] / joints_cv[:, :, 2].unsqueeze(-1) 
                OP_diff = OP_diff * priors["body_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)

            # Face
            id_mapper_smpl=[15, 23, 24, 58, 59]
            id_mapper_openpose=[0, 16, 15, 17, 18]
            
            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["body_keypts"][:, id_mapper_openpose, :2] - joints_cv[:, :, :2] / joints_cv[:, :, 2].unsqueeze(-1) 
                OP_diff = OP_diff * priors["body_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)

            # Elbow
            id_mapper_smpl=[18]
            id_mapper_openpose=[6]
            
            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["body_keypts"][:, id_mapper_openpose, :2] - joints_cv[:, :, :2] / joints_cv[:, :, 2].unsqueeze(-1) 
                OP_diff = OP_diff * priors["body_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)

            id_mapper_smpl=[19]
            id_mapper_openpose=[3]

            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["body_keypts"][:, id_mapper_openpose, :2] - joints_cv[:, :, :2] / joints_cv[:, :, 2].unsqueeze(-1) 
                OP_diff = OP_diff * priors["body_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)

            # Wrist
            id_mapper_openpose = [
                open_joint['left_wrist'], open_joint['right_wrist']]
            id_mapper_smpl = [j_map_o2s[jidx] for jidx in id_mapper_openpose]
            
            if priors["use_RGBD"]:
                joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
                OP_diff = priors["body_keypts"][:, id_mapper_openpose, :2] - joints_cv[:, :, :2] / joints_cv[:, :, 2].unsqueeze(-1) 
                OP_diff = OP_diff * priors["body_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
                OP_diff = robustifier(OP_diff)
                OP_loss += torch.sum(OP_diff)
            
            # # Finger
            # id_mapper_smpl = \
            #     (id_mapper_smpl_finger_thumb +
            #     id_mapper_smpl_finger_index +
            #     id_mapper_smpl_finger_middle +
            #     id_mapper_smpl_finger_pinky +
            #     id_mapper_smpl_finger_ring)
            # id_mapper_openpose = \
            #     (id_mapper_openpose_finger_thumb +
            #     id_mapper_openpose_finger_index +
            #     id_mapper_openpose_finger_middle +
            #     id_mapper_openpose_finger_pinky +
            #     id_mapper_openpose_finger_ring)

            # if priors["use_RGBD"]:
            #     joints_cv = torch.einsum('bki, ij -> bkj', j_tr[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=j_tr.dtype, device=j_tr.device))
            #     OP_diff = priors["hand_keypts"][:, id_mapper_openpose, :2] - joints_cv[:, :, :2] / joints_cv[:, :, 2].unsqueeze(-1) 
            #     OP_diff = OP_diff * priors["hand_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
            #     OP_diff = robustifier(OP_diff)
            #     OP_loss += torch.sum(OP_diff)

            # # face align
            # if priors["use_RGBD"]:
            #     id_mapper_smpl= [9120, 9929, 9448]
            #     id_mapper_openpose= [30, 68, 69]

            #     joints_cv = torch.einsum('bki, ij -> bkj', verts[:,id_mapper_smpl], torch.tensor(np.array([[1.,0,0],[0,-1,0],[0,0,-1]]), dtype=verts.dtype, device=verts.device))
            #     OP_diff = priors["face_keypts"][:, id_mapper_openpose, :2]  - joints_cv[:, :, :2] / joints_cv[:, :, 2].unsqueeze(-1) 
            #     OP_diff = OP_diff * priors["face_keypts"][:, id_mapper_openpose, 2].unsqueeze(-1)
            #     OP_diff = robustifier(OP_diff)
            #     OP_loss = torch.sum(OP_diff)
            #     face_loss = OP_loss

            id_mapper_dvm = dvm_map['all']

            dvm_diff = priors["tar_dvm_sparse"][:, :, :3] - verts[:, priors["smplx_dvm_sparse"]]
            dvm_diff = dvm_diff * priors["tar_dvm_sparse"][:, :, 3].unsqueeze(-1) # valid mask
            dvm_diff = dvm_diff[:, id_mapper_dvm, :]
            dvm_diff = robustifier(dvm_diff)
            dvm_loss = torch.sum(dvm_diff)

            prior_loss += dvm_weight * dvm_loss


            if priors["use_RGBD"] and False:
                head_pose_reg = torch.sum(priors["face_valid"] * head_pose_reg_prior)
            
            total_loss = prior_loss + \
                0.0001*body_pose_preserve_prior + \
                0.00001*betas_preserve_prior + \
                0.0001*global_orient_preserve_prior + \
                0.0001*transl_preserve_prior + \
                0.0001*scale_preserve_prior + \
                priors["pose_reg_weight"]*body_pose_reg_prior + \
                priors["temp_reg_weight"]*temp_smooth_reg_prior + \
                priors["glo_temp_reg_weight"]*temp_glo_smooth_reg_prior + \
                priors["glo_temp_reg_weight"]*temp_glo_trans_reg_prior + \
                priors["OP_weight"]*OP_loss + \
                0.0001*betas_reg_prior +\
                0.001*head_pose_reg+\
                1*face_loss


        else:
            raise ValueError(f'step {step} is not found in clolsure!')

        if priors["use_RGBD"]:
            cfg.rootLogger.debug("All_Loss {0}, prior {1},chamfer {2}, dvm {3}, op_pose {4}".format(total_loss.item(), prior_loss.item(), chamfer_loss.item(), dvm_loss.item(), OP_loss.item()))
            cfg.rootLogger.debug("{0}, {1}, {2}".format(temp_glo_smooth_reg_prior.item(), temp_smooth_reg_prior.item(), temp_glo_trans_reg_prior.item()))
        else:
            cfg.rootLogger.debug("All_Loss {0}, prior {1},chamfer {2}, dvm {3}".format(total_loss.item(), prior_loss.item(), chamfer_loss.item(), dvm_loss.item()))
        # print("pose_reg {0}, temp_reg {1}".format(body_pose_reg_prior.item(), temp_smooth_reg_prior.item()))
        # print("head_pose_reg {0}, face_loss {1}".format(head_pose_reg.item(), face_loss.item()))

        if backward:
            total_loss.backward(create_graph=False)

        return total_loss

    return fitting_func


def create_fitting_closure_CAPE(optimizer,
                           body_model, 
                           input_params,
                           tar_verts,
                           visualize=True):
    global g_vis, g_initialized

    robustifier = GMoF(rho=100)

    (input_body_pose, input_betas_1, input_global_orient,
     input_transl) = input_params

    recon_frames = input_body_pose.shape[0]

    if visualize:
        g_vis.clear_geometries()
        smplx_mesh = o3d.geometry.TriangleMesh()
        smplx_mesh.triangles = o3d.utility.Vector3iVector(body_model.faces)
        with torch.no_grad():
            input_betas_tmp = torch.ones((recon_frames, 10), dtype = input_betas_1.dtype, device=input_betas_1.device)
            input_betas = input_betas_tmp * input_betas_1

            body_model_output = body_model(body_pose=input_body_pose,
                                        betas=input_betas,
                                        global_orient=input_global_orient,
                                        return_verts=True)
            verts = body_model_output.vertices + input_transl
        smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
        smplx_mesh.compute_triangle_normals()
        smplx_mesh.compute_vertex_normals()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tar_verts[demo_frame].cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        if not g_initialized:
            g_vis.create_window(width=1920, height=1080)
            g_vis.register_key_callback(262, lambda vis: upframe()) #
            g_vis.register_key_callback(263, lambda vis: downframe()) #
            g_initialized = True
        g_vis.add_geometry(smplx_mesh)
        g_vis.add_geometry(pcd)
        g_vis.poll_events()
        g_vis.update_renderer()

    def fitting_func(backward=True):
        global demo_frame, g_refresh, g_vis
        if backward:
            optimizer.zero_grad()

        input_betas_tmp = torch.ones((recon_frames, 10), dtype = input_betas_1.dtype, device=input_betas_1.device)
        input_betas = input_betas_tmp * input_betas_1

        body_model_output = body_model(body_pose=input_body_pose,
                                       betas=input_betas,
                                       global_orient=input_global_orient,
                                       return_verts=True)

        verts = body_model_output.vertices + input_transl

        if visualize:
            demo_frame += recon_frames
            demo_frame %= recon_frames
            with torch.no_grad():
                smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
                smplx_mesh.compute_triangle_normals()
                smplx_mesh.compute_vertex_normals()

                if g_refresh:
                    pcd.points = o3d.utility.Vector3dVector(np.asarray(tar_verts[demo_frame].cpu().numpy()))
                    pcd.paint_uniform_color([0.8,0,0.8])
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
                    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
                    g_vis.update_geometry(pcd)

            g_vis.update_geometry(smplx_mesh)
            g_vis.poll_events()
            g_vis.update_renderer()

        total_loss = torch.sum(robustifier(tar_verts - verts))
        # cfg.rootLogger.debug("All_Loss {0}".format(total_loss.item()))
        
        if backward:
            total_loss.backward(create_graph=False)

        return total_loss

    return fitting_func


def create_fitting_closure_Resynth(optimizer,
                           body_model, 
                           input_params,
                           tar_verts,
                           visualize=True):
    global g_vis, g_initialized

    robustifier = GMoF(rho=100)

    (input_body_pose, input_betas_1, input_global_orient,
     input_transl, input_scale_1, 
     input_left_hand_pose, input_right_hand_pose) = input_params


    recon_frames = input_body_pose.shape[0]

    if visualize:
        g_vis.clear_geometries()
        smplx_mesh = o3d.geometry.TriangleMesh()
        smplx_mesh.triangles = o3d.utility.Vector3iVector(body_model.faces)
        with torch.no_grad():
            input_betas_tmp = torch.ones((recon_frames, 10), dtype = input_betas_1.dtype, device=input_betas_1.device)
            input_betas = input_betas_tmp * input_betas_1

            body_model_output = body_model(body_pose=input_body_pose,
                                        betas=input_betas,
                                        global_orient=input_global_orient,
                                        left_hand_pose=input_left_hand_pose,
                                        right_hand_pose=input_right_hand_pose,
                                        return_verts=True)
            verts = body_model_output.vertices + input_transl
        smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
        smplx_mesh.compute_triangle_normals()
        smplx_mesh.compute_vertex_normals()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tar_verts[demo_frame].cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))

        if not g_initialized:
            g_vis.create_window(width=1920, height=1080)
            g_vis.register_key_callback(262, lambda vis: upframe()) #
            g_vis.register_key_callback(263, lambda vis: downframe()) #
            g_initialized = True
        g_vis.add_geometry(smplx_mesh)
        g_vis.add_geometry(pcd)
        g_vis.poll_events()
        g_vis.update_renderer()

    def fitting_func(backward=True):
        global demo_frame, g_refresh, g_vis
        if backward:
            optimizer.zero_grad()

        input_betas_tmp = torch.ones((recon_frames, 10), dtype = input_betas_1.dtype, device=input_betas_1.device)
        input_betas = input_betas_tmp * input_betas_1

        body_model_output = body_model(body_pose=input_body_pose,
                                    betas=input_betas,
                                    global_orient=input_global_orient,
                                    left_hand_pose=input_left_hand_pose,
                                    right_hand_pose=input_right_hand_pose,
                                    return_verts=True)

        verts = body_model_output.vertices + input_transl

        if visualize:
            demo_frame += recon_frames
            demo_frame %= recon_frames
            with torch.no_grad():
                smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
                smplx_mesh.compute_triangle_normals()
                smplx_mesh.compute_vertex_normals()

                if g_refresh:
                    pcd.points = o3d.utility.Vector3dVector(np.asarray(tar_verts[demo_frame].cpu().numpy()))
                    pcd.paint_uniform_color([0.8,0,0.8])
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
                    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
                    g_vis.update_geometry(pcd)

            g_vis.update_geometry(smplx_mesh)
            g_vis.poll_events()
            g_vis.update_renderer()

        total_loss = torch.sum(robustifier(tar_verts - verts))
        # cfg.rootLogger.debug("All_Loss {0}".format(total_loss.item()))
        
        if backward:
            total_loss.backward(create_graph=False)

        return total_loss

    return fitting_func


def create_fitting_closure_Resynth_shead(anchor,
                           optimizer,
                           body_model, 
                           input_params,
                           tar_verts,
                           visualize=True):
    global g_vis, g_initialized

    robustifier = GMoF(rho=100)

    (input_body_pose, input_betas_1, input_global_orient,
     input_transl, input_scale_1, 
     input_left_hand_pose, input_right_hand_pose) = input_params
    expression=torch.zeros([1, 10], dtype = input_betas_1.dtype, device=input_betas_1.device)
    jaw_pose=torch.zeros([1, 3], dtype = input_betas_1.dtype, device=input_betas_1.device)
    leye_pose=torch.zeros([1, 3], dtype = input_betas_1.dtype, device=input_betas_1.device)
    reye_pose=torch.zeros([1, 3], dtype = input_betas_1.dtype, device=input_betas_1.device)

    recon_frames = input_body_pose.shape[0]
    custom_lbs_weight = body_model.lbs_weights[anchor["smplx2shead"]]

    if visualize:
        g_vis.clear_geometries()
        smplx_mesh = o3d.geometry.TriangleMesh()
        smplx_mesh.triangles = o3d.utility.Vector3iVector(anchor["shead_tri"])
        with torch.no_grad():
            input_betas_tmp = torch.ones((recon_frames, 10), dtype = input_betas_1.dtype, device=input_betas_1.device)
            input_betas = input_betas_tmp * input_betas_1

            body_model_output = body_model(body_pose=input_body_pose,
                                        betas=input_betas,
                                        global_orient=input_global_orient,
                                        left_hand_pose=input_left_hand_pose,
                                        right_hand_pose=input_right_hand_pose,
                                        return_verts=True)
            verts = body_model_output.vertices + input_transl
        smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame, anchor["smplx2shead"], :].cpu().numpy())
        smplx_mesh.compute_triangle_normals()
        smplx_mesh.compute_vertex_normals()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tar_verts[demo_frame].cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))

        if not g_initialized:
            g_vis.create_window(width=1920, height=1080)
            g_vis.register_key_callback(262, lambda vis: upframe()) #
            g_vis.register_key_callback(263, lambda vis: downframe()) #
            g_initialized = True
        g_vis.add_geometry(smplx_mesh)
        g_vis.add_geometry(pcd)
        g_vis.poll_events()
        g_vis.update_renderer()

    def fitting_func(backward=True):
        global demo_frame, g_refresh, g_vis
        if backward:
            optimizer.zero_grad()

        input_betas_tmp = torch.ones((recon_frames, 10), dtype = input_betas_1.dtype, device=input_betas_1.device)
        input_betas = input_betas_tmp * input_betas_1



        body_model_output = body_model(body_pose=torch.zeros_like(input_body_pose),
                            betas=input_betas,
                            global_orient=torch.zeros_like(input_global_orient),
                            left_hand_pose=torch.zeros_like(input_left_hand_pose),
                            right_hand_pose=torch.zeros_like(input_right_hand_pose),
                            expression=expression.repeat(recon_frames, 1),
                            jaw_pose=jaw_pose.repeat(recon_frames, 1),
                            leye_pose=leye_pose.repeat(recon_frames, 1),
                            reye_pose=reye_pose.repeat(recon_frames, 1),
                            return_verts=True)

        deformed_verts = body_model.LBS_deform(custom_lbs_weight, body_model_output.vertices[:, anchor["smplx2shead"]],
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
        verts = deformed_verts + input_transl

        if visualize:
            demo_frame += recon_frames
            demo_frame %= recon_frames
            with torch.no_grad():
                smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
                smplx_mesh.compute_triangle_normals()
                smplx_mesh.compute_vertex_normals()

                if g_refresh:
                    pcd.points = o3d.utility.Vector3dVector(np.asarray(tar_verts[demo_frame].cpu().numpy()))
                    pcd.paint_uniform_color([0.8,0,0.8])
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
                    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
                    g_vis.update_geometry(pcd)

            g_vis.update_geometry(smplx_mesh)
            g_vis.poll_events()
            g_vis.update_renderer()

        total_loss = torch.sum(robustifier(tar_verts - verts))
        # cfg.rootLogger.debug("All_Loss {0}".format(total_loss.item()))
        
        if backward:
            total_loss.backward(create_graph=False)

        return total_loss

    return fitting_func



def create_fitting_closure_NeuroGIF(optimizer,
                           body_model, 
                           input_params,
                           tar_pcds,
                           visualize=True):
    global g_vis, g_initialized

    robustifier = GMoF(rho=100)

    (input_body_pose, input_betas_1, input_global_orient,
     input_transl, input_scale_1, 
     input_left_hand_pose, input_right_hand_pose) = input_params


    recon_frames = input_body_pose.shape[0]

    if visualize:
        g_vis.clear_geometries()
        smplx_mesh = o3d.geometry.TriangleMesh()
        smplx_mesh.triangles = o3d.utility.Vector3iVector(body_model.faces)
        with torch.no_grad():
            input_betas_tmp = torch.ones((recon_frames, 10), dtype = input_betas_1.dtype, device=input_betas_1.device)
            input_betas = input_betas_tmp * input_betas_1

            body_model_output = body_model(body_pose=input_body_pose,
                                        betas=input_betas,
                                        global_orient=input_global_orient,
                                        left_hand_pose=input_left_hand_pose,
                                        right_hand_pose=input_right_hand_pose,
                                        return_verts=True)
            verts = body_model_output.vertices + input_transl
        smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
        smplx_mesh.compute_triangle_normals()
        smplx_mesh.compute_vertex_normals()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tar_pcds[demo_frame].cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))

        if not g_initialized:
            g_vis.create_window(width=1920, height=1080)
            g_vis.register_key_callback(262, lambda vis: upframe()) #
            g_vis.register_key_callback(263, lambda vis: downframe()) #
            g_initialized = True
        g_vis.add_geometry(smplx_mesh)
        g_vis.add_geometry(pcd)
        g_vis.poll_events()
        g_vis.update_renderer()

    def fitting_func(backward=True):
        global demo_frame, g_refresh, g_vis
        if backward:
            optimizer.zero_grad()

        input_betas_tmp = torch.ones((recon_frames, 10), dtype = input_betas_1.dtype, device=input_betas_1.device)
        input_betas = input_betas_tmp * input_betas_1

        body_model_output = body_model(body_pose=input_body_pose,
                                    betas=input_betas,
                                    global_orient=input_global_orient,
                                    left_hand_pose=input_left_hand_pose,
                                    right_hand_pose=input_right_hand_pose,
                                    return_verts=True)

        verts = body_model_output.vertices + input_transl

        if visualize:
            demo_frame += recon_frames
            demo_frame %= recon_frames
            with torch.no_grad():
                smplx_mesh.vertices = o3d.utility.Vector3dVector(verts[demo_frame].cpu().numpy())
                smplx_mesh.compute_triangle_normals()
                smplx_mesh.compute_vertex_normals()

                if g_refresh:
                    pcd.points = o3d.utility.Vector3dVector(np.asarray(tar_pcds[demo_frame].cpu().numpy()))
                    pcd.paint_uniform_color([0.8,0,0.8])
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
                    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
                    g_vis.update_geometry(pcd)

            g_vis.update_geometry(smplx_mesh)
            g_vis.poll_events()
            g_vis.update_renderer()

        dist, _ = chamfer_distancePP_diff(verts, tar_pcds)
        dist2, _ = chamfer_distancePP_diff(tar_pcds, verts)

        temp_smooth_reg_prior = torch.sum(robustifier(input_body_pose[1:-1] - input_body_pose[2:])) + torch.sum(robustifier(input_body_pose[1:-1] - input_body_pose[:-2]))
        reg_prior = torch.sum(robustifier(input_body_pose))

        total_loss = torch.sum(robustifier(dist)) + 0.01 * temp_smooth_reg_prior + 0.1 * reg_prior
        # total_loss = torch.sum(robustifier(dist)) + torch.sum(robustifier(dist2)) + 0.01 * temp_smooth_reg_prior
        
        if backward:
            total_loss.backward(create_graph=False)
            
            # input_body_pose.grad[:, :9] = 0
            # input_body_pose.grad[:, 15:18] = 0
            # input_body_pose.grad[:, 24:27] = 0
            # input_body_pose.grad[:, 36:42] = 0
            # input_body_pose.grad[:, 57:] = 0
            input_body_pose.grad[:, :33] = 0
            input_body_pose.grad[:, 36:42] = 0
            input_body_pose.grad[:, 45:] = 0

        return total_loss

    return fitting_func



def create_fitting_closure_base_mesh_hand(optimizer,
                           body_model, 
                           input_params,
                           in_data, 
                           visualize=True):
    global g_vis, g_initialized

    robustifier = GMoF(rho=100)

    (input_body_pose, input_betas_1, input_global_orient,
        input_transl, input_scale_1, 
        input_left_hand_pose, input_right_hand_pose,
        v_residual, f_template) = input_params

    dtype = input_betas_1.dtype
    device = input_betas_1.device
    recon_frames = input_body_pose.shape[0]

    expression=torch.zeros([1, 10], dtype=dtype, device=device)
    jaw_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    leye_pose=torch.zeros([1, 3], dtype=dtype, device=device)
    reye_pose=torch.zeros([1, 3], dtype=dtype, device=device)

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

    v_init = body_model_output.vertices[0, in_data["anchor"]["smplx2shead"]].unsqueeze(0)
    custom_lbs_weight = body_model.lbs_weights[in_data["anchor"]["smplx2shead"]]
    
    in_data["v_init"] = v_init
    L = get_uniform_laplacian_1(in_data["anchor"]["shead_tri"])
    L = L.tocoo()
    L_gpu = torch.sparse.LongTensor(torch.LongTensor([L.row.tolist(), L.col.tolist()]),
                                torch.LongTensor(L.data.astype(np.int32))).to(device)
    L_gpu = L_gpu.type(torch.FloatTensor).to(device).to_dense()

    shead2nohand = torch.tensor(in_data["anchor"]["shead2nohand"]).to(device)


    if visualize:
        g_vis.clear_geometries()
        base_mesh = o3d.geometry.TriangleMesh()
        base_mesh.triangles = o3d.utility.Vector3iVector(in_data["anchor"]["shead_tri"])
        with torch.no_grad():
            demo_T_verts = v_init
            demo_T_verts[:, shead2nohand] += v_residual
            demo_deformed_verts = body_model.LBS_deform(custom_lbs_weight, demo_T_verts,
                                    body_pose=input_body_pose[demo_frame].unsqueeze(0),
                                    betas=input_betas_1,
                                    global_orient=input_global_orient[demo_frame].unsqueeze(0),
                                    left_hand_pose=torch.zeros_like(input_left_hand_pose[0]).unsqueeze(0),
                                    right_hand_pose=torch.zeros_like(input_right_hand_pose[0]).unsqueeze(0),
                                    expression=expression,
                                    jaw_pose=jaw_pose,
                                    leye_pose=leye_pose,
                                    reye_pose=reye_pose,
                                    return_verts=True)
            demo_deformed_verts = (demo_deformed_verts[0] + input_transl[demo_frame])
        base_mesh.vertices = o3d.utility.Vector3dVector(demo_deformed_verts.cpu().numpy())
        base_mesh.compute_triangle_normals()
        base_mesh.compute_vertex_normals()
        base_mesh.paint_uniform_color([0.8,0,0.8])


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(in_data["pcd_list_o3d"][demo_frame].points))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(in_data["pcd_list_o3d"][demo_frame].colors))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        g_vis.create_window(width=1920, height=1080)
        g_vis.register_key_callback(262, lambda vis: upframe()) #
        g_vis.register_key_callback(263, lambda vis: downframe()) #

        g_vis.add_geometry(base_mesh)
        g_vis.add_geometry(pcd)
        g_vis.poll_events()
        g_vis.update_renderer()
    

    def fitting_func(backward=True):
        global demo_frame, g_refresh, g_vis
        if backward:
            optimizer.zero_grad()

        T_verts_1 = v_init.clone()
        T_verts_1[:, shead2nohand] += v_residual
        T_verts_tmp = torch.ones((recon_frames, v_init.shape[1], 3), dtype = T_verts_1.dtype, device=T_verts_1.device)
        T_verts = T_verts_tmp * T_verts_1

        residual_reg_prior = torch.sum(v_residual ** 2)
        E_lap = torch.sum(torch.matmul(T_verts_1[0].permute(1, 0), L_gpu) ** 2)

        deformed_verts = body_model.LBS_deform(custom_lbs_weight, T_verts,
                                body_pose=input_body_pose,
                                betas=input_betas_1,
                                global_orient=input_global_orient,
                                left_hand_pose=torch.zeros_like(input_left_hand_pose),
                                right_hand_pose=torch.zeros_like(input_right_hand_pose),
                                expression=expression.repeat(recon_frames, 1),
                                jaw_pose=jaw_pose.repeat(recon_frames, 1),
                                leye_pose=leye_pose.repeat(recon_frames, 1),
                                reye_pose=reye_pose.repeat(recon_frames, 1),
                                return_verts=True)
        deformed_verts = (deformed_verts + input_transl)

        if visualize:
            demo_frame += recon_frames
            demo_frame %= recon_frames
            with torch.no_grad():
                base_mesh.vertices = o3d.utility.Vector3dVector(deformed_verts[demo_frame].cpu().numpy())
                base_mesh.compute_triangle_normals()
                base_mesh.compute_vertex_normals()
                if g_refresh:
                    pcd.points = o3d.utility.Vector3dVector(np.asarray(in_data["pcd_list_o3d"][demo_frame].points))
                    pcd.colors = o3d.utility.Vector3dVector(np.asarray(in_data["pcd_list_o3d"][demo_frame].colors))
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
                    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
                    g_vis.update_geometry(pcd)
            g_vis.update_geometry(base_mesh)

            g_vis.poll_events()
            g_vis.update_renderer()

        new_src_mesh = Meshes(verts=deformed_verts, faces=f_template.unsqueeze(0).repeat(recon_frames, 1, 1))
        new_src_points = sample_points_from_meshes(new_src_mesh, 10000)
        
        dist_normal = chamfer_distancePP_diff(new_src_points, in_data["pcd_torch"])
        x_y_dist = in_data["mask_torch"].unsqueeze(-1) * dist_normal[0]
        chamfer_loss = torch.sum(robustifier(x_y_dist))

        total_loss = chamfer_loss + in_data["Lap_reg_weight"] * E_lap + in_data["residual_reg_weight"] * residual_reg_prior

        cfg.rootLogger.debug("chamfer_loss {0}, lap_reg {1}, res_reg {2}".format(chamfer_loss.item(), E_lap.item(), residual_reg_prior.item()))

        if backward:
            total_loss.backward(create_graph=False)

        return total_loss

    return fitting_func

def run_fitting(optimizer, closure, params, body_model, max_iters, ftol, gtol):

    prev_loss = None

    for n in range(max_iters):
        loss = optimizer.step(closure)

        if torch.isnan(loss).sum() > 0:
            cfg.rootLogger.error('NaN loss value, stopping!')
            break

        if torch.isinf(loss).sum() > 0:
            cfg.rootLogger.error('Infinite loss value, stopping!')
            break

        if n > 0 and prev_loss is not None and ftol > 0:
            loss_rel_change = rel_change(prev_loss, loss.item())
            if loss_rel_change <= ftol:
                cfg.rootLogger.debug('loss converge')
                break
        
        if all([torch.abs(var.grad.view(-1).max()).item() < gtol
                for var in params if var.grad is not None]):
            cfg.rootLogger.debug('grad converge')
            break

        print(str(n) + " iteration")

        prev_loss = loss.item() 
    return prev_loss 


