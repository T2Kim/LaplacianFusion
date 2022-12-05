import numpy as np
import torch

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

def load_params(file_name, device, dtype=torch.float32, requires_grad=False):
    packed = np.load(file_name)

    input_body_pose = torch.tensor(packed["body_pose"],
                                    dtype=dtype,
                                    device=device,
                                    requires_grad=requires_grad)
    input_left_hand_pose = torch.tensor(packed["left_hand_pose"],
                                    dtype=dtype,
                                    device=device,
                                    requires_grad=requires_grad)
    input_right_hand_pose = torch.tensor(packed["right_hand_pose"],
                                    dtype=dtype,
                                    device=device,
                                    requires_grad=requires_grad)
    input_betas = torch.tensor(packed["betas"],
                                dtype=dtype,
                                device=device,
                                requires_grad=requires_grad)
    input_global_orient = torch.tensor(packed["global_orient"],
                                        dtype=dtype,
                                        device=device,
                                        requires_grad=requires_grad)
    input_transl = torch.tensor(packed["transl"],
                            dtype=dtype,
                            device=device,
                            requires_grad=requires_grad)
    input_scale = torch.tensor(packed["scale"], dtype=dtype, device=device, requires_grad=requires_grad)


    input_params = [
        input_body_pose,
        input_betas,
        input_global_orient,
        input_transl,
        input_scale,
        input_left_hand_pose,
        input_right_hand_pose
    ]

    return input_params

def save_params_SMPL(file_name, input_params):
    (input_body_pose, input_betas, input_global_orient,
     input_transl) = input_params

    np.savez(file_name, 
             body_pose=input_body_pose.detach().cpu().numpy(),
             betas=input_betas.detach().cpu().numpy(),
             global_orient=input_global_orient.detach().cpu().numpy(),
             transl=input_transl.detach().cpu().numpy())

def save_params_J(file_name, input_params):
    (input_body_pose, J, input_global_orient,
     input_transl) = input_params

    np.savez(file_name, 
             body_pose=input_body_pose,
             J=J,
             global_orient=input_global_orient,
             transl=input_transl)

def load_params_J(file_name, device, dtype=torch.float32, requires_grad=False):
    packed = np.load(file_name)

    input_body_pose = torch.tensor(packed["body_pose"],
                                    dtype=dtype,
                                    device=device,
                                    requires_grad=requires_grad)
    J = torch.tensor(packed["J"],
                                dtype=dtype,
                                device=device,
                                requires_grad=requires_grad)
    input_global_orient = torch.tensor(packed["global_orient"],
                                        dtype=dtype,
                                        device=device,
                                        requires_grad=requires_grad)
    input_transl = torch.tensor(packed["transl"],
                            dtype=dtype,
                            device=device,
                            requires_grad=requires_grad)


    input_params = [
        J,
        input_body_pose,
        input_global_orient,
        input_transl
    ]
    return input_params

# smplx 21 joint
kinematic_tree = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
smplx_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

def get_posemap(n_joints, n_traverse=1, normalize=False, no_head = False):
    A1 = np.zeros((n_joints, n_joints))
    for i in range(n_joints-1):
        A1[i + 1, kinematic_tree[i]] = 1
        A1[kinematic_tree[i], i + 1] = 1

    pose_map = np.zeros((n_traverse + 1, n_joints, n_joints))
    pose_map[0] = np.eye(n_joints)
    for i in range(n_traverse):
        pose_map[i + 1] = A1 @ pose_map[i]
    pose_map = np.sum(pose_map, axis=0)
    pose_map = np.where(pose_map > 0, 1, 0)

    pose_map = pose_map[:, 1:]
    
    if no_head:
        pose_map[15, :] = 0
    return torch.tensor(pose_map)

def get_posemap_custom():
    pose_map = np.ones((22, 22))
    pose_map[0, [4, 5, 7, 8, 10, 11, 12, 15, 18, 19, 20, 21]] = 0
    pose_map[3, [1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 18, 19, 20, 21]] = 0
    pose_map[6, [1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 18, 19, 20, 21]] = 0
    pose_map[9, [1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 18, 19, 20, 21]] = 0
    pose_map[13, [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 18, 19, 20, 21]] = 0
    pose_map[14, [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 18, 19, 20, 21]] = 0
    pose_map[16, [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 20, 21]] = 0
    pose_map[17, [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 20, 21]] = 0

    pose_map[1, :] = 0
    pose_map[1, [1,4]] = 1
    pose_map[2, :] = 0
    pose_map[2, [2,5]] = 1
    pose_map[4, :] = 0
    pose_map[4, [1,4]] = 1
    pose_map[5, :] = 0
    pose_map[5, [2,5]] = 1

    pose_map[7, :] = 0
    pose_map[7, [1,4]] = 1
    pose_map[8, :] = 0
    pose_map[8, [2,5]] = 1
    pose_map[10, :] = 0
    pose_map[11, :] = 0

    pose_map[12, :] = 0
    pose_map[12, [12, 13, 14, 16, 17]] = 1
    pose_map[15, :] = 0

    pose_map[18, :] = 0
    pose_map[18, [16,18]] = 1
    pose_map[19, :] = 0
    pose_map[19, [17,19]] = 1

    pose_map[20, :] = 0
    pose_map[20, [18,20]] = 1
    pose_map[21, :] = 0
    pose_map[21, [19,21]] = 1

    pose_map = pose_map[:, 1:]
    return torch.tensor(pose_map)

def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)
