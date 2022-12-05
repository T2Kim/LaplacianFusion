import open3d as o3d
import re
import numpy as np
import time

g_play = False
g_break = False
g_refresh = False
g_separate = False
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
def speedup():
    global wait_frame
    wait_frame -= 0.005
def speeddown():
    global wait_frame
    wait_frame += 0.005

def separate():
    global g_separate, g_refresh
    g_separate = not g_separate
    g_refresh = True

def __extractCase(case_name):
    Mesh_case = re.compile("Mesh*")
    PCD_case = re.compile("PCD*")
    O3D_PCD_case = re.compile("O3D_PCD*")
    if Mesh_case.match(case_name) != None:
        return "Mesh"
    elif PCD_case.match(case_name) != None:
        return "PCD"
    elif O3D_PCD_case.match(case_name) != None:
        return "O3D_PCD"
    else:
        return ""

def o3d_nb_vis(shape_vec):
    global g_play, g_break, frame_idx, g_refresh, wait_frame

    g_play = False
    g_break = False
    g_refresh = False
    frame_idx = 0
    wait_frame = 0.03

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(256, lambda vis: play_stop()) #ESC
    vis.register_key_callback(262, lambda vis: upframe()) #
    vis.register_key_callback(263, lambda vis: downframe()) #
    vis.register_key_callback(266, lambda vis: speedup()) #
    vis.register_key_callback(267, lambda vis: speeddown()) #
    vis.register_key_callback(32, lambda vis: play_onoff())
    vis.register_key_callback(290, lambda vis: separate()) # F1

    o3d_shapes = {}

    for key, value in shape_vec.items():
        if __extractCase(key) == "Mesh":
            o3d_shapes[key] = o3d.geometry.TriangleMesh()
            o3d_shapes[key].vertices = o3d.utility.Vector3dVector(value["vertices"][0])
            o3d_shapes[key].triangles = o3d.utility.Vector3iVector(value["triangles"])
            if "vertex_colors" in value:
                o3d_shapes[key].vertex_colors= o3d.utility.Vector3dVector(value["vertex_colors"][0])
            o3d_shapes[key].compute_triangle_normals()
            o3d_shapes[key].compute_vertex_normals()
            recon_frames = len(value["vertices"])


        elif __extractCase(key) == "PCD":
            o3d_shapes[key] = o3d.geometry.PointCloud()
            o3d_shapes[key].points = o3d.utility.Vector3dVector(value["points"][0])
            if "colors" in value:
                o3d_shapes[key].colors= o3d.utility.Vector3dVector(value["colors"][0])

            recon_frames = len(value["points"])

        elif __extractCase(key) == "O3D_PCD":
            o3d_shapes[key] = o3d.geometry.PointCloud()
            o3d_shapes[key].points = o3d.utility.Vector3dVector(np.asarray(value["pcd"][0].points))

            recon_frames = len(value["pcd"])

    vis.create_window(width=1920, height=1080)

    for key, value in o3d_shapes.items():
        vis.add_geometry(o3d_shapes[key])


    start = time.time()
    redraw = True
    while g_break == False:
        frame_idx %= recon_frames
        
        # frame_idx = min(frame_idx, recon_frames-1)
        # frame_idx = max(frame_idx, 0)

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

            for key, value in shape_vec.items():
                if __extractCase(key) == "Mesh":
                    o3d_shapes[key].vertices = o3d.utility.Vector3dVector(value["vertices"][frame_idx])
                    o3d_shapes[key].triangles = o3d.utility.Vector3iVector(value["triangles"])
                    o3d_shapes[key].compute_triangle_normals()
                    o3d_shapes[key].compute_vertex_normals()
                    if "vertex_colors" in value:
                        o3d_shapes[key].vertex_colors= o3d.utility.Vector3dVector(value["vertex_colors"][frame_idx])
                    if g_separate:
                        o3d_shapes[key].translate((1, 0, 0))
                elif __extractCase(key) == "PCD":
                    o3d_shapes[key].points = o3d.utility.Vector3dVector(value["points"][frame_idx])
                    if "colors" in value:
                        o3d_shapes[key].colors= o3d.utility.Vector3dVector(value["colors"][frame_idx])
                elif __extractCase(key) == "O3D_PCD":
                    o3d_shapes[key].points = o3d.utility.Vector3dVector(np.asarray(value["pcd"][frame_idx].points))
                    o3d_shapes[key].colors = o3d.utility.Vector3dVector(np.asarray(value["pcd"][frame_idx].colors))
                    if g_separate:
                        o3d_shapes[key].translate((-1, 0, 0))

            for key, value in shape_vec.items():
                vis.update_geometry(o3d_shapes[key])

            if redraw:
                frame_idx += 1
            g_refresh = False
            redraw = False
        vis.poll_events()
        vis.update_renderer()


