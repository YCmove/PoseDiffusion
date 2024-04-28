# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in thed /kaggle/working/PoseDiffusion
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import traceback
import pickle
import glob
import json
import os
import re
import time
from operator import itemgetter
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Optional, Union
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate, get_original_cwd
import models
import time
from functools import partial
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.ops import corresponding_cameras_alignment
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.vis.plotly_vis import plot_scene

from util.utils import seed_all_random_engines
from util.match_extraction import extract_match
from util.load_img_folder import load_and_preprocess_images
from util.geometry_guided_sampling import geometry_guided_sampling
from util.metric import compute_ARE
from visdom import Visdom


@hydra.main(config_path="../cfgs/", config_name="default")
def demo(cfg: DictConfig) -> None:
    username = os.getlogin()
    OmegaConf.set_struct(cfg, False)
    print("Model Config:")
    print(OmegaConf.to_yaml(cfg))

    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sequence_name = '110_13051_23361'
    # cate_dir = f'/media/{username}/ac876a5c-83ac-4029-bad1-3ae5fa7c3831/data/co3d/data/apple'
    # pred_dir = f'{cate_dir}/pred/{sequence_name}'
    # pred_path = f'{pred_dir}/pred_cameras.pkl'

    # Image Matching Challenging 2024
    cate_dir = 'church'
    pred_dir = f'/media/{username}/ac876a5c-83ac-4029-bad1-3ae5fa7c3831/data/image-matching-challenge-2024/test/pred'
    pred_path = f'{pred_dir}/church.pkl'

    # Load and preprocess images
    original_cwd = get_original_cwd()  # Get original working directory
    folder_path = os.path.join(original_cwd, cfg.image_folder)
    images, image_info = load_and_preprocess_images(folder_path, cfg.image_size)

    if not os.path.exists(pred_path):
        # Instantiate the model
        model = instantiate(cfg.MODEL, _recursive_=False)


        # print(f'folder_path={folder_path}')

        # Load checkpoint
        ckpt_path = os.path.join(original_cwd, cfg.ckpt)
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded checkpoint from: {ckpt_path}")
        else:
            raise ValueError(f"No checkpoint found at: {ckpt_path}")

        # Move model and images to the GPU
        model = model.to(device)
        images = images.to(device)

        # Evaluation Mode
        model.eval()

        # Seed random engines
        seed_all_random_engines(cfg.seed)

        # Start the timer
        start_time = time.time()

        # Perform match extraction
        if cfg.GGS.enable:
            # Optional TODO: remove the keypoints outside the cropped region?

            kp1, kp2, i12 = extract_match(image_folder_path=folder_path, image_info=image_info)

            if kp1 is not None:
                keys = ["kp1", "kp2", "i12", "img_shape"]
                values = [kp1, kp2, i12, images.shape]
                matches_dict = dict(zip(keys, values))

                cfg.GGS.pose_encoding_type = cfg.MODEL.pose_encoding_type
                GGS_cfg = OmegaConf.to_container(cfg.GGS)

                cond_fn = partial(geometry_guided_sampling, matches_dict=matches_dict, GGS_cfg=GGS_cfg)
                print("[92m=====> Sampling with GGS <=====[0m")
            else:
                cond_fn = None
        else:
            cond_fn = None
            print("[92m=====> Sampling without GGS <=====[0m")

        images = images.unsqueeze(0)

        # Forward
        with torch.no_grad():
            # Obtain predicted camera parameters
            # pred_cameras is a PerspectiveCameras object with attributes
            # pred_cameras.R, pred_cameras.T, pred_cameras.focal_length

            # The poses and focal length are defined as
            # NDC coordinate system in
            # https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
            predictions = model(image=images, cond_fn=cond_fn, cond_start_step=cfg.GGS.start_step, training=False)

        pred_cameras = predictions["pred_cameras"]

        # os.makedirs(pred_dir)
        with open(pred_path, "wb" ) as f:
            pickle.dump(pred_cameras, f)

        # Stop the timer and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time taken: {:.4f} seconds".format(elapsed_time))

    else:
        with open(pred_path, 'rb') as f:
            pred_cameras = pickle.load(f)
            print(f'dir(pred_cameras)={dir(pred_cameras)}')



    # Compute metrics if gt is available
    json_path = f'{cate_dir}/frame_annotations.json'
    # Load gt poses
    if os.path.exists(os.path.join(folder_path, "gt_cameras.npz")):
        pass
        # gt_cameras_dict = np.load(os.path.join(folder_path, "gt_cameras.npz"))
        # gt_cameras = PerspectiveCameras(
        #     focal_length=gt_cameras_dict["gtFL"], R=gt_cameras_dict["gtR"], T=gt_cameras_dict["gtT"], device=device
        # )

        # # 7dof alignment, using Umeyama's algorithm
        # pred_cameras_aligned = corresponding_cameras_alignment(
        #     cameras_src=pred_cameras, cameras_tgt=gt_cameras, estimate_scale=True, mode="extrinsics", eps=1e-9
        # )

        # # Compute the absolute rotation error
        # ARE = compute_ARE(pred_cameras_aligned.R, gt_cameras.R).mean()
        # print(f"For {folder_path}: the absolute rotation error is {ARE:.6f} degrees.")

    # if os.path.exists(os.path.join(folder_path, "frame_annotations.json")):
    elif os.path.exists(json_path):
        with open(json_path) as f:
            print(f'reading the file ... ')
            # lines = f.readline()
            gt_list = json.load(f)
            target_gt_list = []

            # todo
            for gt in gt_list:
                if gt['sequence_name'] == sequence_name:
                    target_gt_list.append(gt)

            ordered_gt_list = sorted(target_gt_list, key=itemgetter('frame_number'))
            
            # print(f'type of lines = {lines}')
            # d = lines[0]
            # d = json.loads(json_data)[0]

            gtR = np.array([g["viewpoint"]['R'] for g in ordered_gt_list])
            gtT = np.array([g["viewpoint"]['T'] for g in ordered_gt_list])
            gtFL = np.array([g["viewpoint"]['focal_length'] for g in ordered_gt_list])
            # print('hi')
            # gtR = np.array(d["viewpoint"]['R'])
            # gtT = np.array(d["viewpoint"]['T'])
            # gtFL = np.array(d["viewpoint"]['focal_length'])

            # print(f'gtR={gtR}, gtT={gtT}, gtFL={gtFL}')
            print(f'gtR={gtR.shape}, gtT={gtT.shape}, gtFL={gtFL.shape}')

            gt_cameras_dict = {
                'gtR': gtR,
                'gtT': gtT,
                'gtFL': gtFL
            }

            print(f'** shape of gtR={gt_cameras_dict["gtR"].shape}, gtT={gt_cameras_dict["gtT"].shape}, gtFL={gt_cameras_dict["gtFL"].shape}')
            print(f'** shape of pred_cameras.R={pred_cameras.T.shape}, T={pred_cameras.T.shape}, focal_length={pred_cameras.focal_length.shape}')
            gt_cameras = PerspectiveCameras(
                focal_length=gt_cameras_dict["gtFL"], R=gt_cameras_dict["gtR"], T=gt_cameras_dict["gtT"], device=device
            )

            print(f'gt_cameras={gt_cameras}')

            # 7dof alignment, using Umeyama's algorithm
            pred_cameras_aligned = corresponding_cameras_alignment(
                cameras_src=pred_cameras, cameras_tgt=gt_cameras, estimate_scale=True, mode="extrinsics", eps=1e-9
            )

            # Compute the absolute rotation error
            ARE = compute_ARE(pred_cameras_aligned.R, gt_cameras.R).mean()
            print(f"For {folder_path}: the absolute rotation error is {ARE:.6f} degrees.")


    else:
        print(f"No GT provided. No evaluation conducted.")

        viz = Visdom()
        cams_show = {"ours_pred": pred_cameras}
        fig = plot_scene({f"{folder_path}": cams_show})
        viz.plotlyplot(fig, env=f'church_test', win="cams")


    # Visualization
    # try:
    #     viz = Visdom()

    #     # cams_show = {"ours_pred": pred_cameras}
    #     cams_show = {"ours_pred": pred_cameras, "ours_pred_aligned": pred_cameras_aligned, "gt_cameras": gt_cameras}

    #     fig = plot_scene({f"{folder_path}": cams_show})

    #     viz.plotlyplot(fig, env="visual", win="cams")
    # except:
    #     print(traceback.format_exc())
    #     print("Please check your visdom connection")


def test_gt() -> None:
    username = os.getlogin()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load and preprocess images
    # original_cwd = get_original_cwd()  # Get original working directory
    # folder_path = os.path.join(original_cwd, cfg.image_folder)
    # images, image_info = load_and_preprocess_images(folder_path, cfg.image_size)

    folder_path = '/media/{username}/ac876a5c-83ac-4029-bad1-3ae5fa7c3831/data/co3d/data/apple'
    json_path = os.path.join(folder_path, "frame_annotations.json")

    if os.path.exists(os.path.join(folder_path, "gt_cameras.npz")):
        gt_cameras_dict = np.load(os.path.join(folder_path, "gt_cameras.npz"))
        gt_cameras = PerspectiveCameras(
            focal_length=gt_cameras_dict["gtFL"], R=gt_cameras_dict["gtR"], T=gt_cameras_dict["gtT"], device=device
        )

        # 7dof alignment, using Umeyama's algorithm
        pred_cameras_aligned = corresponding_cameras_alignment(
            cameras_src=pred_cameras, cameras_tgt=gt_cameras, estimate_scale=True, mode="extrinsics", eps=1e-9
        )

        # Compute the absolute rotation error
        ARE = compute_ARE(pred_cameras_aligned.R, gt_cameras.R).mean()
        print(f"For {folder_path}: the absolute rotation error is {ARE:.6f} degrees.")

    elif os.path.exists(json_path):
        print('====================')
        with open(json_path) as f:
            print(f'reading the file ... ')
            # lines = f.readline()
            d = json.load(f)[0]
            # print(f'type of lines = {lines}')
            # d = lines[0]
            # d = json.loads(json_data)[0]
            gtR = np.array(d["viewpoint"]['R'])
            gtT = np.array(d["viewpoint"]['T'])
            gtFL = np.array(d["viewpoint"]['focal_length'])

            print(f'gtR={gtR}, gtT={gtT}, gtFL={gtFL}')
            print(f'gtR={gtR.shape}, gtT={gtT.shape}, gtFL={gtFL.shape}')

            gt_cameras_dict = {
                'gtR': np.expand_dims(gtR, axis=0),
                'gtT': np.expand_dims(gtT, axis=0),
                'gtFL': np.expand_dims(gtFL, axis=0)
            }

            print(f'gtR={gt_cameras_dict["gtR"].shape}, gtT={gt_cameras_dict["gtT"].shape}, gtFL={gt_cameras_dict["gtFL"].shape}')

            gt_cameras = PerspectiveCameras(
                focal_length=gt_cameras_dict["gtFL"], R=gt_cameras_dict["gtR"], T=gt_cameras_dict["gtT"], device=device
            )

            print(f'gt_cameras={gt_cameras}')


    else:
        print(f"No GT provided. No evaluation conducted.")


if __name__ == "__main__":
    
    demo()
    # test_gt()
