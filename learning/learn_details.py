import sys
sys.path.append("../")
sys.path.append("./")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import numpy as np
import random
import time
import argparse
import open3d as o3d

import config as cfg
from learning.dataset import MappedLapDataset
import lib.smplx as smplx
from lib.model.mlp import MLP_Detail
from lib.human_fitting.utils_smplify import GMoF

random_seed = 4332
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--target_subj", default='hyomin_example')
parser.add_argument("--target_gender", default='male')
parser.add_argument('--RGBD', default=False, help='Is Point Cloud?')
parser.add_argument('--add_noise', default=True, help='Add noise?')
parser.add_argument('--flathand', default=False)


args = parser.parse_args()

if __name__ == '__main__':
    cfg.make_dir_structure(args.target_subj)
    cfg.set_log_file(os.path.join(cfg.DataPath["Main"], "logs", args.target_subj, os.path.splitext(os.path.basename(__file__))[0]))
    cfg.rootLogger.info("Start learning pose dependent details")
    target_dir = os.path.join(cfg.DataPath["Main"], "subjects", args.target_subj)

    if cfg.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float32
    
    filename_mapped_delta = os.path.join(target_dir, "train/mapped_delta.npy")
    # filename_mapped_delta = os.path.join(target_dir, "train/mapped_offset.npy") # ablation <- use offset instead of laplacian

    filename_smpl_fit = os.path.join(target_dir, "train/smplx_fit.npz")

    anchor = np.load(os.path.join(cfg.DataPath["Main"], cfg.DataPath["Anchor"]))

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
    custom_lbs_weight = body_model.lbs_weights[anchor["smplx2shead"]]
    body_skin_weight = custom_lbs_weight[:, :22].cpu().numpy()

    train_dataset = MappedLapDataset(body_model, filename_mapped_delta, filename_smpl_fit, anchor, args.RGBD, args.add_noise, device, dtype)

    model_lap = MLP_Detail(3, 84).to(device)
    L2Loss = nn.MSELoss()
    robustifier = GMoF(rho=100)

    optimizer = torch.optim.Adam(model_lap.parameters(), cfg.lap_learning_rate)

    iteration_number = 0

    epoch_iter = len(train_dataset) // cfg.shape_batch_size_lap
    cfg.rootLogger.info("Iter per Epoch: " + str(epoch_iter))

    for epoch in range(0, cfg.epochs_lap):
        model_lap.train()
        iteration_number = 0

        start = time.time()

        get_arr = np.arange(len(train_dataset))
        np.random.shuffle(get_arr)
        get_arr = torch.from_numpy(get_arr).long().to(device)
        
        for i in range(0, epoch_iter):
            data_idx_s = i * cfg.shape_batch_size_lap
            data_idx_e = min((i + 1) * cfg.shape_batch_size_lap, len(train_dataset))
            data = train_dataset.get_item(get_arr[data_idx_s:data_idx_e])

            time_load = time.time() - start
            start = time.time()

            with torch.no_grad():
                feat_input, pose_input, fit_lap, z_weight, global_R = data

            time_data = time.time() - start
            start = time.time()

            try:
                lap_pred = torch.matmul(global_R, torch.unsqueeze(model_lap(feat_input, pose_input).unsqueeze(1), dim=-1)).squeeze()
                # lap_pred = model_lap(feat_input, pose_input)

            except RuntimeError as e:
                cfg.rootLogger.error("Runtime error!", e)
                cfg.rootLogger.error("Exiting...")
                exit()

            time_inference = time.time() - start
            start = time.time()

            if args.RGBD:
                # lap_loss = torch.mean(torch.sum((lap_pred - fit_lap) ** 2, dim=-1) * z_weight)
                lap_loss = torch.mean(((lap_pred - fit_lap) ** 2) * z_weight.unsqueeze(1))
                # lap_loss = torch.mean(torch.sum(torch.abs(lap_pred - fit_lap), dim=-1) * z_weight)
            else:
                lap_loss = torch.mean(torch.sum((lap_pred - fit_lap) ** 2, dim=-1))

            loss = lap_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time_step = time.time() - start

            if iteration_number % 100 == 0:
                cfg.rootLogger.debug("Epoch {0}, Iteration {1}, All_Loss {2}, Lap_Loss {3}".format(epoch, iteration_number, loss.item(), lap_loss.item()))
                cfg.rootLogger.debug("load time {0}, data time: {1}, Inference time: {2}, Step time: {3}".format(time_load, time_data, time_inference, time_step))

            iteration_number = iteration_number + 1

            start = time.time()
        
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model_lap.state_dict(), os.path.join(target_dir, "net/lap_model" + "_e" + str(epoch) + ".pts"))

    torch.save(model_lap.state_dict(), os.path.join(target_dir, "net/lap_model" + "_e" + str(cfg.epochs_lap) + ".pts"))

