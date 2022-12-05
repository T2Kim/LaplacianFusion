import torch
import torch.nn.functional as F

from lib.smplx.lbs import blend_shapes, batch_rodrigues, batch_rigid_transform, vertices2joints

def custom_lbs(betas, pose, v_template, shapedirs, J_regressor, parents,
        custom_lbs_weights, v_custom, only_rotation=False, inverse=False, pose2rot=True, displacements=None, dtype=torch.float32):

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = torch.einsum('bik,ji->bjk', [v_shaped, J_regressor])

    # 3. No pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3)).view([batch_size, -1, 3, 3])
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = custom_lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_custom.shape[1], 1],
                               dtype=dtype, device=device)
    if only_rotation:
        homogen_coord = torch.zeros([batch_size, v_custom.shape[1], 1],
                               dtype=dtype, device=device)
    v_custom_homo = torch.cat([v_custom, homogen_coord], dim=2)

    # inverse only rotation
    if inverse:
        v_homo = torch.matmul(T.permute(0, 1, 3, 2), torch.unsqueeze(v_custom_homo, dim=-1))
    else:
        v_homo = torch.matmul(T, torch.unsqueeze(v_custom_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts


def custom_lbs_smpl(pose, v_skin, v_custom, body_model, pose2rot=True, dtype=torch.float32):

    batch_size = pose.shape[0]
    device = pose.device

    J_regressor = body_model.J_regressor
    parents = body_model.parents
    lbs_weights = body_model.lbs_weights

    # Get the joints
    # NxJx3 array
    J = torch.einsum('bik,ji->bjk', [v_skin, J_regressor])

    # 3. No pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3)).view([batch_size, -1, 3, 3])
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_custom.shape[1], 1],
                               dtype=dtype, device=device)
    v_custom_homo = torch.cat([v_custom, homogen_coord], dim=2)

    # inverse only rotation
    v_homo = torch.matmul(T, torch.unsqueeze(v_custom_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]

    return verts, J


def get_global_R(pose, v_template, J_regressor, parents, dtype=torch.float32):
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view([pose.shape[0], -1, 3, 3])
    J = vertices2joints(J_regressor, v_template.view(1,-1,3).repeat(pose.shape[0], 1, 1))
    _, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    return A[:,:,:3,:3]


def LBS_deform(body_model, custom_lbs_weights, custom_v, only_rotation = False, inverse = False,
            betas=None, global_orient=None, body_pose=None,
            left_hand_pose=None, right_hand_pose=None, transl=None,
            expression=None, jaw_pose=None, leye_pose=None, reye_pose=None,
            return_verts=True, return_full_pose=False, pose2rot=True, **kwargs):

    # If no shape and pose parameters are passed along, then use the
    # ones from the module
    global_orient = (global_orient if global_orient is not None else
                        body_model.global_orient)
    body_pose = body_pose if body_pose is not None else body_model.body_pose
    betas = betas if betas is not None else body_model.betas

    left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                        body_model.left_hand_pose)
    right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                        body_model.right_hand_pose)
    jaw_pose = jaw_pose if jaw_pose is not None else body_model.jaw_pose
    leye_pose = leye_pose if leye_pose is not None else body_model.leye_pose
    reye_pose = reye_pose if reye_pose is not None else body_model.reye_pose
    expression = expression if expression is not None else body_model.expression

    apply_trans = transl is not None or hasattr(body_model, 'transl')
    if transl is None:
        if hasattr(body_model, 'transl'):
            transl = body_model.transl

    if body_model.use_pca:
        left_hand_pose = torch.einsum(
            'bi,ij->bj', [left_hand_pose, body_model.left_hand_components])
        right_hand_pose = torch.einsum(
            'bi,ij->bj', [right_hand_pose, body_model.right_hand_components])

    full_pose = torch.cat([global_orient, body_pose,
                            jaw_pose, leye_pose, reye_pose,
                            left_hand_pose,
                            right_hand_pose], dim=1)

    # Add the mean pose of the model. Does not affect the body, only the
    # hands when flat_hand_mean == False
    full_pose += body_model.pose_mean

    batch_size = max(betas.shape[0], global_orient.shape[0],
                        body_pose.shape[0])
    # Concatenate the shape and expression coefficients
    scale = int(batch_size / betas.shape[0])
    if scale > 1:
        betas = betas.expand(scale, -1)
    shape_components = torch.cat([betas, expression], dim=-1)
    
    shapedirs = torch.cat([body_model.shapedirs, body_model.expr_dirs], dim=-1)


    deformed_v = custom_lbs(shape_components, full_pose, body_model.v_template,
                            shapedirs, body_model.J_regressor, body_model.parents,
                            custom_lbs_weights, custom_v, 
                            only_rotation = only_rotation, inverse = inverse,
                            pose2rot=pose2rot,
                            dtype=body_model.dtype)

    return deformed_v

def convert_global_R(body_model, body_pose, global_orient, device='cuda'):
    num_batches = body_pose.shape[0]
    jaw_pose=torch.zeros([num_batches, 3], dtype=body_model.dtype, device=device)
    leye_pose=torch.zeros([num_batches, 3], dtype=body_model.dtype, device=device)
    reye_pose=torch.zeros([num_batches, 3], dtype=body_model.dtype, device=device)
    left_hand_pose = torch.zeros([num_batches, 45], dtype=body_model.dtype, device=device)
    right_hand_pose = torch.zeros([num_batches, 45], dtype=body_model.dtype, device=device)
    full_pose = torch.cat([global_orient, body_pose,
                            jaw_pose, leye_pose, reye_pose,
                            left_hand_pose,
                            right_hand_pose], dim=1)
    global_R = get_global_R(full_pose, body_model.v_template, body_model.J_regressor, body_model.parents)
    return global_R

