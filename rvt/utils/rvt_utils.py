# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Utility function for Our Agent
"""
import pdb
import argparse
import sys
import signal
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DistributedDataParallel as DDP

import rvt.utils.peract_utils as peract_utils
from rvt.models.peract_official import PreprocessAgent2
from PIL import Image, ImageDraw
import numpy as np
import os
from typing import List
from rlbench.backend.observation import Observation

def get_pc_img_feat(obs, pcd, bounds=None):
    """
    preprocess the data in the peract to our framework
    """
    # obs, pcd = peract_utils._preprocess_inputs(batch)
    bs = obs[0][0].shape[0]
    # concatenating the points from all the cameras
    # (bs, num_points, 3)
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)
    _img_feat = [o[0] for o in obs]
    img_dim = _img_feat[0].shape[1]
    # (bs, num_points, 3)
    img_feat = torch.cat(
        [p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1
    )

    img_feat = (img_feat + 1) / 2

    # x_min, y_min, z_min, x_max, y_max, z_max = bounds
    # inv_pnt = (
    #     (pc[:, :, 0] < x_min)
    #     | (pc[:, :, 0] > x_max)
    #     | (pc[:, :, 1] < y_min)
    #     | (pc[:, :, 1] > y_max)
    #     | (pc[:, :, 2] < z_min)
    #     | (pc[:, :, 2] > z_max)
    # )

    # # TODO: move from a list to a better batched version
    # pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    # img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]

    return pc, img_feat


def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
    """
    :param no_op: no operation
    """
    if no_op:
        return pc, img_feat

    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    )

    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    return pc, img_feat


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TensorboardManager:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            if "image" in k:
                for i, x in enumerate(v):
                    self.writer.add_image(f"{split}_{step}", x, i)
            elif "hist" in k:
                if isinstance(v, list):
                    self.writer.add_histogram(k, v, step)
                elif isinstance(v, dict):
                    hist_id = {}
                    for i, idx in enumerate(sorted(v.keys())):
                        self.writer.add_histogram(f"{split}_{k}_{step}", v[idx], i)
                        hist_id[i] = idx
                    self.writer.add_text(f"{split}_{k}_{step}_id", f"{hist_id}")
                else:
                    assert False
            else:
                self.writer.add_scalar("%s_%s" % (split, k), v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def short_name(cfg_opts):
    SHORT_FORMS = {
        "peract": "PA",
        "sample_distribution_mode": "SDM",
        "optimizer_type": "OPT",
        "lr_cos_dec": "LCD",
        "num_workers": "NW",
        "True": "T",
        "False": "F",
        "pe_fix": "pf",
        "transform_augmentation_rpy": "tar",
        "lambda_weight_l2": "l2",
        "resume": "RES",
        "inp_pre_pro": "IPP",
        "inp_pre_con": "IPC",
        "cvx_up": "CU",
        "stage_two": "ST",
        "feat_ver": "FV",
        "lamb": "L",
        "img_size": "IS",
        "img_patch_size": "IPS",
        "rlbench": "RLB",
        "move_pc_in_bound": "MPIB",
        "rend": "R",
        "xops": "X",
        "warmup_steps": "WS",
        "epochs": "E",
        "amp": "A",
    }

    if "resume" in cfg_opts:
        cfg_opts = cfg_opts.split(" ")
        res_idx = cfg_opts.index("resume")
        cfg_opts.pop(res_idx + 1)
        cfg_opts = " ".join(cfg_opts)

    cfg_opts = cfg_opts.replace(" ", "_")
    cfg_opts = cfg_opts.replace("/", "_")
    cfg_opts = cfg_opts.replace("[", "")
    cfg_opts = cfg_opts.replace("]", "")
    cfg_opts = cfg_opts.replace("..", "")
    for a, b in SHORT_FORMS.items():
        cfg_opts = cfg_opts.replace(a, b)

    return cfg_opts


def get_num_feat(cfg):
    num_feat = cfg.num_rotation_classes * 3
    # 2 for grip, 2 for collision
    num_feat += 4
    return num_feat


def get_eval_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["insert_onto_square_peg"]
    )
    parser.add_argument("--model-folder", type=str, default=None)
    parser.add_argument("--eval-datafolder", type=str, default="./data/val/")
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="start to evaluate from which episode",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="how many episodes to be evaluated for each task",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=25,
        help="maximum control steps allowed for each episode",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--ground-truth", action="store_true", default=False)
    parser.add_argument("--exp_cfg_path", type=str, default=None)
    parser.add_argument("--mvt_cfg_path", type=str, default=None)
    parser.add_argument("--peract_official", action="store_true")
    parser.add_argument(
        "--peract_model_dir",
        type=str,
        default="runs/peract_official/seed0/weights/600000",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--use-input-place-with-mean", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--skip", action="store_true")
    
    parser.add_argument("--visualize_bbox", action="store_true",default=False)
    parser.add_argument("--zoom_in", action="store_true",default=False)
    parser.add_argument("--visualize", action="store_true",default=False)
    parser.add_argument("--visualize_root_dir", type=str, default="")
    parser.add_argument("--colosseum", action="store_true", default=False)
    parser.add_argument("--lang_type", type=str, default='clip')
    parser.add_argument("--agent_type", type=str, default='original')
    return parser


RLBENCH_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]

COLOSSEUM_TASKS = [
    "basketball_in_hoop",
    "close_box",
    "empty_dishwasher",
    "get_ice_from_fridge",
    "hockey",
    "meat_on_grill",
    "move_hanger",
    "wipe_desk",
    "open_drawer",
    "slide_block_to_target",
    "reach_and_drag",
    "put_money_in_safe",
    "place_wine_at_rack_location",
    "insert_onto_square_peg",
    "turn_oven_on",
    "straighten_rope",
    "setup_chess",
    "scoop_with_spatula",
    "close_laptop_lid",
    "stack_cups",
]

def load_agent(agent_path, agent=None, only_epoch=False):
    if isinstance(agent, PreprocessAgent2):
        assert not only_epoch
        agent._pose_agent.load_weights(agent_path)
        return 0

    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network
        optimizer = agent._optimizer
        lr_sched = agent._lr_sched

        if isinstance(model, DDP):
            model = model.module

        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            try:
                print(
                    "WARNING: loading states in mvt1. "
                    "Be cautious if you are using a two stage network."
                )
                model.mvt1.load_state_dict(checkpoint["model_state"])
            except RuntimeError:
                print(
                    "WARNING: loading states with strick=False! "
                    "KNOW WHAT YOU ARE DOING!!"
                )
                model.load_state_dict(checkpoint["model_state"], strict=False)

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print(
                "WARNING: No optimizer_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

        if "lr_sched_state" in checkpoint:
            lr_sched.load_state_dict(checkpoint["lr_sched_state"])
        else:
            print(
                "WARNING: No lr_sched_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

    return epoch

def save_point_cloud_with_color(filename, points, colors, keypoint=None):
    """
    Save the point cloud and colors to a PLY file, automatically handling the color value range.
    :param filename: Output file name (e.g. 'point_cloud.ply')
    :param points: Point cloud coordinates (N,3) np.array
    :param colors: Color values (N,3) np.array (0-255 or 0-1)
    :param keypoint: Keypoint coordinates (3,) np.array (optional)
    """

    # Ensure data dimensions are correct
    assert points.shape[1] == 3 
    assert colors.shape[1] == 3
    
    # Automatically detect color value range and convert to 0-255
    if colors.max() <= 1.0:  # If color values are between 0-1
        colors = (colors * 255).astype(np.uint8)
    else:  # If color values are between 0-255
        colors = colors.astype(np.uint8)
    
    # Add keypoint (optional)
    if keypoint is not None:
        points = np.vstack([points, keypoint])
        colors = np.vstack([colors, np.array([255, 0, 0])])  # Mark keypoint in red

    # Write to PLY file
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, clr in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(clr[0])} {int(clr[1])} {int(clr[2])}\n")


def visualize_images(
    color_tensor: torch.Tensor,  #  (3, 3, 224, 224) 
    gray_tensor: torch.Tensor,   #  (224, 224, 3) 
    save_dir: str = "/opt/tiger/3D_OpenVLA/3d_policy/RVT/rvt_our/debug"
) -> None:
    """
    1. original_0.png, original_1.png, original_2.png   (original image)
    2. gray_0.png, gray_1.png, gray_2.png              (gray image)
    3. overlay_0.png, overlay_1.png, overlay_2.png     (transparent image + annotation)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    color_imgs = color_tensor.cpu().numpy().transpose(0, 2, 3, 1) 
    gray_imgs = gray_tensor.cpu().numpy().transpose(2, 0, 1)     
    
    for i in range(3):

        original_img = np.clip(color_imgs[i], 0, 1) * 255
        original_img = original_img.astype(np.uint8)
        Image.fromarray(original_img).save(os.path.join(save_dir, f"original_{i}.png"))
        

        gray_img = np.clip(gray_imgs[i], 0, 1) * 255
        gray_img = gray_img.astype(np.uint8)
        Image.fromarray(gray_img, mode="L").save(os.path.join(save_dir, f"gray_{i}.png"))
        

        rgba = np.zeros((*original_img.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = original_img  
        rgba[..., 3] = 77            
        
    
        overlay_img = Image.fromarray(rgba, mode="RGBA")
        draw = ImageDraw.Draw(overlay_img)
        
        
        max_pos = np.unravel_index(gray_imgs[i].argmax(), gray_imgs[i].shape)
        x = max_pos[1]  
        y = max_pos[0]  
        
      
        point_radius = 5
        draw.ellipse(
            [x-point_radius, y-point_radius, x+point_radius, y+point_radius],
            fill=(255, 0, 0, 255)  
        )
        
        overlay_img.save(os.path.join(save_dir, f"overlay_{i}.png"))


def apply_channel_wise_softmax(gray_tensor):
    """
    Apply softmax normalization independently to each grayscale channel
    Input shape: (H, W, C) -> Output shape: (H, W, C)
    All elements in each channel are processed by softmax and sum to 1
    """
    # Convert to PyTorch tensor (if not already)
    if not isinstance(gray_tensor, torch.Tensor):
        gray_tensor = torch.tensor(gray_tensor, dtype=torch.float32)
    
    # Separate each channel (C, H, W)
    channels = gray_tensor.permute(2, 0, 1)
    
    # Apply softmax to each channel and flatten
    softmax_channels = []
    for c in range(channels.shape[0]):
        channel = channels[c].flatten()
        softmax_channel = torch.softmax(channel, dim=0)
        softmax_channels.append(softmax_channel.view_as(channels[c]))
    
    # Merge channels and restore original shape (H, W, C)
    return torch.stack(softmax_channels, dim=2)

def gripper_change(demo, i, threshold=2):    
    start = max(0, i - threshold)
    for k in range(start, i):
        if demo[k].gripper_open != demo[i].gripper_open:
            return True
    return False

def _is_stopped(low_dim_obs: List[Observation], i, stopped_buffer, delta):
    """判断机器人是否停止运动
    
    Args:
        low_dim_obs: RLBench观测序列
        i: 当前时间步
        stopped_buffer: 停止缓冲计数器
        delta: 速度阈值
    """
    next_is_not_final = i == (len(low_dim_obs) - 2)
    
    gripper_state_no_change = i < (len(low_dim_obs) - 2) and (
        low_dim_obs[i].gripper_open == low_dim_obs[i + 1].gripper_open
        and low_dim_obs[i].gripper_open == low_dim_obs[max(0, i - 1)].gripper_open
        and low_dim_obs[max(0, i - 2)].gripper_open == low_dim_obs[max(0, i - 1)].gripper_open
    )
    
    small_delta = np.allclose(low_dim_obs[i].joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped

def keypoint_discovery(low_dim_obs: List[Observation], stopping_delta=0.1) -> List[int]:
    """发现轨迹中的关键点

    Args:
        low_dim_obs: RLBench观测序列
        stopping_delta: 判断停止的速度阈值
        
    Returns:
        episode_keypoints: 关键点索引列表
    """
    episode_keypoints = []
    prev_gripper_open = low_dim_obs[0].gripper_open
    stopped_buffer = 0

    for i in range(len(low_dim_obs)):
        stopped = _is_stopped(low_dim_obs, i, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # 如果夹持器状态改变或到达序列末尾
        last = i == (len(low_dim_obs) - 1)
        if i != 0 and (low_dim_obs[i].gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = low_dim_obs[i].gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints