
import sys
sys.path.append("./")
sys.path.append("../")
import os

import open3d as o3d
import numpy as np
import glob
from sklearn.neighbors import KDTree

import config as cfg
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--target_subj", default='hyomin_example')
args = parser.parse_args()

def fit(P, p, n_neighbors = 20):   

    n_points = len(p)
    c = np.mean(P, axis=1)[:, np.newaxis, :]
    P_c = P - c
    PcTPc = np.einsum('ijk, ikl -> ijl', np.transpose(P_c, (0, 2, 1)), P_c)
    e_val, e_vec = np.linalg.eig(PcTPc)
    idx = np.argsort(e_val, axis=1)[:,::-1]

    e_val = e_val[np.repeat(np.arange(n_points)[:, np.newaxis], 3, axis=1), idx]
    e_vec = e_vec[np.repeat(np.repeat(np.arange(n_points)[:, np.newaxis, np.newaxis], 3, axis=1), 3, axis=2),\
                np.repeat(np.repeat(np.arange(3)[np.newaxis, :, np.newaxis], n_points, axis=0), 3, axis=2),\
                np.repeat(idx[:, np.newaxis, :], 3, axis=1)]

    P_p = P - p[:, np.newaxis, :]
    dist = np.linalg.norm(P_p, axis=-1)
    Kxyz = np.einsum('ikx, ixz-> ikz', P_p, e_vec)

    Vk = np.ones((n_points, n_neighbors, 6))
    Vk[:,:,1] = Kxyz[:,:,0]
    Vk[:,:,2] = Kxyz[:,:,1]
    Vk[:,:,3] = Kxyz[:,:,0] * Kxyz[:,:,0]
    Vk[:,:,4] = Kxyz[:,:,0] * Kxyz[:,:,1]
    Vk[:,:,5] = Kxyz[:,:,1] * Kxyz[:,:,1]

    h = np.max(dist, axis=1)
    w_d = np.exp(-(dist * dist) / (h * h)[:, np.newaxis])
    w_dd = np.where(dist == 0, 1, 1/n_neighbors)

    wVVT = np.sum(np.einsum('ijxy, ijyz -> ijxz', Vk[:,:,:,np.newaxis], Vk[:,:,np.newaxis,:]) * w_d[:,:,np.newaxis,np.newaxis], axis=1)
    wdVVT = np.sum(np.einsum('ijxy, ijyz -> ijxz', Vk[:,:,:,np.newaxis], Vk[:,:,np.newaxis,:]) * w_dd[:,:,np.newaxis,np.newaxis], axis=1)
    wVVT_1 = np.linalg.inv(wVVT)
    wdVVT_1 = np.linalg.inv(wdVVT)
    wVF = np.sum(Vk * Kxyz[:,:,2][:,:,np.newaxis] * w_d[:,:,np.newaxis], axis=1)

    wVFx = np.sum(Vk * P[:,:,0][:,:,np.newaxis] * w_dd[:,:,np.newaxis], axis=1)
    wVFy = np.sum(Vk * P[:,:,1][:,:,np.newaxis] * w_dd[:,:,np.newaxis], axis=1)
    wVFz = np.sum(Vk * P[:,:,2][:,:,np.newaxis] * w_dd[:,:,np.newaxis], axis=1)

    A_mat = np.einsum('ijk, ikl -> ijl', wVVT_1, wVF[:,:,np.newaxis])
    a2 = A_mat[:,1,0]
    a3 = A_mat[:,2,0]
    a4 = A_mat[:,3,0]
    a5 = A_mat[:,4,0]
    a6 = A_mat[:,5,0]

    D2 = np.reciprocal(a2 ** 2 + a3 ** 2 + 1) ** 2
    D4 = D2 * D2
    A1 = D4 * ((a2 * a3) * (a2 * a5 + 2 * a3 * a6) - (a3 ** 2 + 1) * (2 * a2 * a4 + a3 * a5))\
            + D2 * (a3 * a5 - 2 * a2 * a6)
    A2 = D4 * ((a2 * a3) * (a3 * a5 + 2 * a2 * a3) - (a2 ** 2 + 1) * (2 * a3 * a6 + a2 * a5))\
            + D2 * (a2 * a5 - 2 * a3 * a4)
    A3 = D2 * (a3 ** 2 + 1)
    A4 = -2 * D2 * a2 * a3
    A5 = D2 * (a2 ** 2 + 1)

    As = np.concatenate((A1[:, np.newaxis], A2[:, np.newaxis], 2 * A3[:, np.newaxis], A4[:, np.newaxis], 2 * A5[:, np.newaxis]), axis=1)


    Cx = np.einsum('ijk, ikl -> ijl', wdVVT_1, wVFx[:,:,np.newaxis])
    Cy = np.einsum('ijk, ikl -> ijl', wdVVT_1, wVFy[:,:,np.newaxis])
    Cz = np.einsum('ijk, ikl -> ijl', wdVVT_1, wVFz[:,:,np.newaxis])


    Lap_x = np.einsum('ij, ij -> i', As, Cx[:,1:,0])
    Lap_y = np.einsum('ij, ij -> i', As, Cy[:,1:,0])
    Lap_z = np.einsum('ij, ij -> i', As, Cz[:,1:,0])
    
    delta = np.concatenate((Lap_x[:, np.newaxis], Lap_y[:, np.newaxis], Lap_z[:, np.newaxis]), axis=1)
    return delta


def main(target_dir):
    pcd_dir = os.path.join(target_dir, "train/pcd")
    delta_dir = os.path.join(target_dir, "train/delta")

    os.makedirs(delta_dir, exist_ok=True)

    meshes = sorted(glob.glob(os.path.join(pcd_dir, "*")))

    for filename_mesh in tqdm(meshes, desc="Laplacian approximation"):
        # cfg.rootLogger.info(filename_mesh)
        basename = os.path.basename(filename_mesh)
        filename_delta = os.path.join(delta_dir, os.path.splitext(basename)[0] + ".npy")
        

        tmp_pcd = o3d.io.read_point_cloud(filename_mesh)
        p = np.asarray(tmp_pcd.points)
        pcd_tree = KDTree(p)

        fit_neighbors = 15
        try:
            dist, near_w_idx = pcd_tree.query(p, k=fit_neighbors)
            P = p[near_w_idx]
            delta_all = fit(P, p, fit_neighbors)
            np.save(filename_delta, delta_all)
            continue
        except:
            cfg.rootLogger.debug("30 neighbor")
        fit_neighbors = 20
        try:
            dist, near_w_idx = pcd_tree.query(p, k=fit_neighbors) 
            P = p[near_w_idx]
            delta_all = fit(P, p, fit_neighbors)
            np.save(filename_delta, delta_all)
            continue
        except:
            cfg.rootLogger.debug("40 neighbor")
        fit_neighbors = 30
        try:
            dist, near_w_idx = pcd_tree.query(p, k=fit_neighbors)
            P = p[near_w_idx]
            delta_all = fit(P, p, fit_neighbors)
            np.save(filename_delta, delta_all)
            continue
        except:
            cfg.rootLogger.debug("40 neighbor")


if __name__ == '__main__':
    cfg.rootLogger.info("Start pcd laplacian approximation")

    target_dir = os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj)
    
    main(target_dir)



