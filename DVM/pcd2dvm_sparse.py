import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
import glob

import numpy as np
import open3d as o3d

from DVM.Labeler_111 import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Res16UNet34C', help='Model name')
parser.add_argument('--weights', type=str, default='/DATA/lapfu/dvm_weight.pth')
parser.add_argument('--marker', type=int, default=111)
parser.add_argument('--bn_momentum', type=float, default=0.05)
parser.add_argument('--voxel_size', type=float, default=0.01)
parser.add_argument('--conv1_kernel_size', type=int, default=3)

config = parser.parse_args()
labeler = Labeler(config)
view = True

if __name__ == '__main__':
    target_dir = "/DATA/lapfu/subjects/hyomin_example/train"

    pcd_dir = os.path.join(target_dir, "pcd")
    pcds = sorted(glob.glob(os.path.join(pcd_dir, "*.ply")))

    if view:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)

        pcd = o3d.io.read_point_cloud(pcds[0])
        pcd2 = o3d.io.read_point_cloud(pcds[0])
        vis.add_geometry(pcd)
        vis.add_geometry(pcd2)
        vis.poll_events()
        vis.update_renderer()

    print("Read pcd files")
    sparse_marker_list = []
    for i, filename_pcd in enumerate(pcds):
        print(filename_pcd)
        tmp_pcd = o3d.io.read_point_cloud(filename_pcd)
        pcd_label, inv_idx, upcoord_pred = labeler.getLabel_points_fast2(np.asarray(tmp_pcd.points), y_filp=True)

        sparse_marker_list.append(pcd_label["max_coords_mask"][np.newaxis, :, :])

        # test_shapes = [tmp_pcd]
        # for p_w in pcd_label["max_coords_mask"]:
        #     if(p_w[3] > 0.3):
        #         mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        #         mesh_sphere.compute_vertex_normals()
        #         mesh_sphere.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_sphere.vertices) + np.array(p_w[:3]))
        #         mesh_sphere.paint_uniform_color([1, 0.1, 0.1])
        #         test_shapes.append(mesh_sphere)
        # o3d.visualization.draw_geometries(test_shapes)

        if view:
            pcd2.points = o3d.utility.Vector3dVector(np.asarray(tmp_pcd.points))
            pcd2.colors = o3d.utility.Vector3dVector(pcd_label["color"])
            pcd2.translate((1,0,0))
            pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
            pcd2.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            pcd.points = o3d.utility.Vector3dVector(np.asarray(tmp_pcd.points))
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(tmp_pcd.colors))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            vis.update_geometry(pcd)
            vis.update_geometry(pcd2)
            vis.poll_events()
            vis.update_renderer()

    sparse_markers = np.vstack(sparse_marker_list)
    np.save(os.path.join(target_dir, "dvm_sparse_markers.npy"), sparse_markers)
