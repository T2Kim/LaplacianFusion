import sys
sys.path.append("../")
sys.path.append("./")
import os
import numpy as np
import scipy as sp
from sklearn.neighbors import KDTree
import igl
import open3d as o3d

import config as cfg

import open3d as o3d


smpl_v, _, _, smpl_f, _, _ = igl.read_obj(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smpl.obj"))
smplx_v, _, _, smplx_f, _, _ = igl.read_obj(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smplx_fit_shead.obj"))
v, tc, _, f, ftc, fn = igl.read_obj(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smpl_shead.obj"))

# smplx <- smpl v_idx
smplx_v_tree = KDTree(smplx_v)
nearest_dist, nearest_ind = smplx_v_tree.query(smpl_v, k=1)
smplx2smpl = nearest_ind[:, 0]

# shead <- smplx v_idx
smplx_v_tree = KDTree(smplx_v)
nearest_dist, nearest_ind = smplx_v_tree.query(v, k=1)
smplx2v = nearest_ind[:, 0]

# no hand
picked_points = np.where(((tc[:, 0] > 636 / 1024) & ((1 - tc[:, 1]) > 758 / 1024)))
picked_points = sorted(np.setdiff1d(np.arange(len(tc)), picked_points))
picked_faces_idx = np.logical_or(np.logical_or(np.isin(ftc[:, 0], picked_points), np.isin(ftc[:, 1], picked_points)), np.isin(ftc[:, 2], picked_points))

# picked_points = np.where(((tc[:, 0] > 636 / 1024) & ((1 - tc[:, 1]) > 758 / 1024)))
# picked_faces_idx = np.logical_and(np.logical_and(np.isin(ftc[:, 0], picked_points), np.isin(ftc[:, 1], picked_points)), np.isin(ftc[:, 2], picked_points))
# picked_faces_idx = np.arange(len(ftc))[picked_faces_idx]
# picked_faces_idx = sorted(np.setdiff1d(np.arange(len(ftc)), picked_faces_idx))

picked_faces = f[picked_faces_idx]

picked_points = np.unique(picked_faces.reshape(-1))
hand_foot_idx = sorted(np.setdiff1d(np.arange(len(v)), picked_points))

table = -np.ones(len(v)).astype(np.int64)
table[picked_points] = np.arange(len(picked_points))
new_v = v[picked_points]
new_f = table[picked_faces]

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(new_v)
mesh.triangles = o3d.utility.Vector3iVector(new_f)


mesh2 = o3d.geometry.TriangleMesh()
mesh2.vertices = o3d.utility.Vector3dVector(v)
mesh2.triangles = o3d.utility.Vector3iVector(f)
mesh2.paint_uniform_color([0.8,0,0.8])

# o3d.visualization.draw_geometries([mesh2, mesh])

o3d.io.write_triangle_mesh(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smpl_shead_nohand.obj"), mesh)

S_x4, new_f_x4 = igl.loop_subdivision_matrix(len(new_v), new_f)
new_v_x4 = S_x4 @ new_v

S_x4_x16, new_f_x16 = igl.loop_subdivision_matrix(S_x4.shape[0], new_f_x4)
new_v_x16 = S_x4_x16 @ new_v_x4
S_x16 = S_x4_x16 @ S_x4

S_x64, new_f_x64 = igl.loop_subdivision_matrix(S_x16.shape[0], new_f_x16)
new_v_x64 = S_x64 @ new_v_x16
S_x64 = S_x64 @ S_x16


# tex2coord
vtc = np.zeros((len(tc), 3))
vtc[ftc] = v[f]

shead_S_x4, shead_f_x4 = igl.loop_subdivision_matrix(len(v), f)
shead_v_x4 = shead_S_x4 @ v
shead_dense_tree = KDTree(shead_v_x4)
dist, near_w_idx = shead_dense_tree.query(new_v_x4, k=1)
shead2nohand_dense_table_x4 = near_w_idx[:, 0]

shead_S_tex_x4, shead_ftc_x4 = igl.loop_subdivision_matrix(len(vtc), ftc)
shead_vtc_x4 = shead_S_tex_x4 @ vtc
shead_dense_tree = KDTree(shead_vtc_x4)
dist, near_w_idx = shead_dense_tree.query(shead_v_x4, k=1)
shead2tex_table_x4 = near_w_idx[:, 0]




shead_S_x4_x16, shead_f_x16 = igl.loop_subdivision_matrix(shead_S_x4.shape[0], shead_f_x4)
shead_v_x16 = shead_S_x4_x16 @ shead_v_x4
shead_S_x16 = shead_S_x4_x16 @ shead_S_x4
shead_dense_tree = KDTree(shead_v_x16)
dist, near_w_idx = shead_dense_tree.query(new_v_x16, k=1)
shead2nohand_dense_table_x16 = near_w_idx[:, 0]

shead_S_tex_x4_x16, shead_ftc_x16 = igl.loop_subdivision_matrix(shead_S_tex_x4.shape[0], shead_ftc_x4)
shead_vtc_x16 = shead_S_tex_x4_x16 @ shead_vtc_x4
shead_S_tex_x16 = shead_S_tex_x4_x16 @ shead_S_tex_x4
shead_dense_tree = KDTree(shead_vtc_x16)
dist, near_w_idx = shead_dense_tree.query(shead_v_x16, k=1)
shead2tex_table_x16 = near_w_idx[:, 0]



shead_S_x64, shead_f_x64 = igl.loop_subdivision_matrix(shead_S_x16.shape[0], shead_f_x16)
shead_v_x64 = shead_S_x64 @ shead_v_x16
shead_S_x64 = shead_S_x64 @ shead_S_x16
shead_dense_tree = KDTree(shead_v_x64)
dist, near_w_idx = shead_dense_tree.query(new_v_x64, k=1)
shead2nohand_dense_table_x64 = near_w_idx[:, 0]

shead_S_tex_x64, shead_ftc_x64 = igl.loop_subdivision_matrix(shead_S_tex_x16.shape[0], shead_ftc_x16)
shead_vtc_x64 = shead_S_tex_x64 @ shead_vtc_x16
shead_S_tex_x64 = shead_S_tex_x64 @ shead_S_tex_x16
shead_dense_tree = KDTree(shead_vtc_x64)
dist, near_w_idx = shead_dense_tree.query(shead_v_x64, k=1)
shead2tex_table_x64 = near_w_idx[:, 0]


mmesh = o3d.geometry.TriangleMesh()
mmesh.vertices = o3d.utility.Vector3dVector(new_v)
mmesh.triangles = o3d.utility.Vector3iVector(new_f)
ppcd = mmesh.sample_points_poisson_disk(799)
ttree = KDTree(v)
_, nn_idx = ttree.query(np.asarray(ppcd.points))
pre_anchor_idx = nn_idx[:, 0]


ppcd = mmesh.sample_points_poisson_disk(1499)
ttree = KDTree(v)
_, nn_idx = ttree.query(np.asarray(ppcd.points))
pre_anchor_idx2000 = nn_idx[:, 0]


np.savez(os.path.join(cfg.DataPath["Main"], "protocol_info_new/shead_protocol.npz"),
        noahnd_coord = new_v, nohand_tri = new_f,
        smplx2nohand = smplx2v[picked_points],
        smplx2shead = smplx2v,
        smplx2smpl = smplx2smpl,
        shead_coord = v, shead_tri = f,
        shead_tex_coord = tc, shead_tex_tri = ftc,
        shead2nohand = picked_points,
        picked_tri = picked_faces_idx,
        hand_foot_idx = hand_foot_idx,
        pre_anchor_idx = pre_anchor_idx,
        pre_anchor_idx_2000 = pre_anchor_idx2000,
        noahnd_x4_coord = new_v_x4, noahnd_x4_tri = new_f_x4,
        noahnd_x16_coord = new_v_x16, noahnd_x16_tri = new_f_x16,
        noahnd_x64_coord = new_v_x64, noahnd_x64_tri = new_f_x64,
        shead_x4_coord  = shead_v_x4,  shead_x4_tri =  shead_f_x4,
        shead_x16_coord = shead_v_x16, shead_x16_tri = shead_f_x16,
        shead_x64_coord = shead_v_x64, shead_x64_tri = shead_f_x64,
        shead2nohand_x4 = shead2nohand_dense_table_x4,
        shead2nohand_x16 = shead2nohand_dense_table_x16,
        shead2nohand_x64 = shead2nohand_dense_table_x64,
        shead2tex_x4  = shead2tex_table_x4,
        shead2tex_x16 = shead2tex_table_x16,
        shead2tex_x64 = shead2tex_table_x64,
        )


sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/nohand_x4.npz" ), S_x4)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/nohand_x16.npz"), S_x16)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/nohand_x4_x16.npz"), S_x4_x16)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/nohand_x64.npz"), S_x64)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/shead_x4.npz"  ), shead_S_x4)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/shead_x16.npz" ), shead_S_x16)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/shead_x4_x16.npz" ), shead_S_x4_x16)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/shead_x64.npz" ), shead_S_x64)


smplx_S_x4, smplx_f_x4 = igl.loop_subdivision_matrix(len(smplx_v), smplx_f)
smplx_S_x16, smplx_f_x16 = igl.loop_subdivision_matrix(smplx_S_x4.shape[0], smplx_f_x4)
smplx_S_x16 = smplx_S_x16 @ smplx_S_x4
smplx_S_x64, smplx_f_x64 = igl.loop_subdivision_matrix(smplx_S_x16.shape[0], smplx_f_x16)
smplx_S_x64 = smplx_S_x64 @ smplx_S_x16
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smplx_x4.npz"  ), smplx_S_x4)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smplx_x16.npz" ), smplx_S_x16)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smplx_x64.npz" ), smplx_S_x64)



### SMPL
smpl_v, _, _, smpl_f, _, _ = igl.read_obj(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smpl.obj"))
v, tc, _, f, ftc, fn = igl.read_obj(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smpl_shead.obj"))

# shead <- smpl v_idx
smpl_v_tree = KDTree(smpl_v)
nearest_dist, nearest_ind = smpl_v_tree.query(v, k=1)
smpl2v = nearest_ind[:, 0]

# no hand
picked_points = np.where(((tc[:, 0] > 636 / 1024) & ((1 - tc[:, 1]) > 758 / 1024)))
picked_points = sorted(np.setdiff1d(np.arange(len(tc)), picked_points))
picked_faces_idx = np.logical_or(np.logical_or(np.isin(ftc[:, 0], picked_points), np.isin(ftc[:, 1], picked_points)), np.isin(ftc[:, 2], picked_points))
picked_faces = f[picked_faces_idx]

picked_points = np.unique(picked_faces.reshape(-1))

table = -np.ones(len(v)).astype(np.int64)
table[picked_points] = np.arange(len(picked_points))
new_v = v[picked_points]
new_f = table[picked_faces]

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(new_v)
mesh.triangles = o3d.utility.Vector3iVector(new_f)
o3d.io.write_triangle_mesh(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smpl_shead_nohand.obj"), mesh)

S_x4, new_f_x4 = igl.loop_subdivision_matrix(len(new_v), new_f)
new_v_x4 = S_x4 @ new_v

S_x4_x16, new_f_x16 = igl.loop_subdivision_matrix(S_x4.shape[0], new_f_x4)
new_v_x16 = S_x4_x16 @ new_v_x4
S_x16 = S_x4_x16 @ S_x4

S_x64, new_f_x64 = igl.loop_subdivision_matrix(S_x16.shape[0], new_f_x16)
new_v_x64 = S_x64 @ new_v_x16
S_x64 = S_x64 @ S_x16


shead_S_x4, shead_f_x4 = igl.loop_subdivision_matrix(len(v), f)
shead_v_x4 = shead_S_x4 @ v
shead_dense_tree = KDTree(shead_v_x4)
dist, near_w_idx = shead_dense_tree.query(new_v_x4, k=1)
shead2nohand_dense_table_x4 = near_w_idx[:, 0]

shead_S_x4_x16, shead_f_x16 = igl.loop_subdivision_matrix(shead_S_x4.shape[0], shead_f_x4)
shead_v_x16 = shead_S_x4_x16 @ shead_v_x4
shead_S_x16 = shead_S_x4_x16 @ shead_S_x4
shead_dense_tree = KDTree(shead_v_x16)
dist, near_w_idx = shead_dense_tree.query(new_v_x16, k=1)
shead2nohand_dense_table_x16 = near_w_idx[:, 0]

shead_S_x64, shead_f_x64 = igl.loop_subdivision_matrix(shead_S_x16.shape[0], shead_f_x16)
shead_v_x64 = shead_S_x64 @ shead_v_x16
shead_S_x64 = shead_S_x64 @ shead_S_x16
shead_dense_tree = KDTree(shead_v_x64)
dist, near_w_idx = shead_dense_tree.query(new_v_x64, k=1)
shead2nohand_dense_table_x64 = near_w_idx[:, 0]


np.savez(os.path.join(cfg.DataPath["Main"], "protocol_info_new/shead_protocol_smpl.npz"),
        noahnd_coord = new_v, nohand_tri = new_f,
        smpl2nohand = smpl2v[picked_points],
        smpl2shead = smpl2v,
        shead_coord = v, shead_tri = f,
        shead2nohand = picked_points,
        hand_foot_idx = hand_foot_idx,
        pre_anchor_idx = pre_anchor_idx,
        noahnd_x4_coord = new_v_x4, noahnd_x4_tri = new_f_x4,
        noahnd_x16_coord = new_v_x16, noahnd_x16_tri = new_f_x16,
        noahnd_x64_coord = new_v_x64, noahnd_x64_tri = new_f_x64,
        shead_x4_coord  = shead_v_x4,  shead_x4_tri =  shead_f_x4,
        shead_x16_coord = shead_v_x16, shead_x16_tri = shead_f_x16,
        shead_x64_coord = shead_v_x64, shead_x64_tri = shead_f_x64,
        shead2nohand_x4 = shead2nohand_dense_table_x4,
        shead2nohand_x16 = shead2nohand_dense_table_x16,
        shead2nohand_x64 = shead2nohand_dense_table_x64
        )


sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/nohand_x4.npz" ), S_x4)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/nohand_x16.npz"), S_x16)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/nohand_x4_x16.npz"), S_x4_x16)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/nohand_x64.npz"), S_x64)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/shead_x4.npz"  ), shead_S_x4)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/shead_x16.npz" ), shead_S_x16)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/shead_x4_x16.npz" ), shead_S_x4_x16)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/shead_x64.npz" ), shead_S_x64)


smpl_S_x4, smpl_f_x4 = igl.loop_subdivision_matrix(len(smpl_v), smpl_f)
smpl_S_x16, smpl_f_x16 = igl.loop_subdivision_matrix(smpl_S_x4.shape[0], smpl_f_x4)
smpl_S_x16 = smpl_S_x16 @ smpl_S_x4
smpl_S_x64, smpl_f_x64 = igl.loop_subdivision_matrix(smpl_S_x16.shape[0], smpl_f_x16)
smpl_S_x64 = smpl_S_x64 @ smpl_S_x16
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smpl_x4.npz"  ), smpl_S_x4)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smpl_x16.npz" ), smpl_S_x16)
sp.sparse.save_npz(os.path.join(cfg.DataPath["Main"], "protocol_info_new/smpl_x64.npz" ), smpl_S_x64)


