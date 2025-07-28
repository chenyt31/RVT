# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# initial source: https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing
# adapted to support loading from disk for faster initialization time

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py
import os
import torch
import pickle
import logging
import numpy as np
from typing import List

import clip
import peract_colab.arm.utils as utils

from peract_colab.rlbench.utils import get_stored_demo
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from rvt.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS, CONTACT_BASE_CATEGORIES
# from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from rvt.libs.peract.helpers.utils import extract_obs

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def gripper_change(demo, i, threshold=2):    
    start = max(0, i - threshold)
    for k in range(start, i):
        if demo[k].gripper_open != demo[i].gripper_open:
            return True
    return False

def touch_change(demo, i, threshold=4, delta=0.005):
    start = max(0, i - threshold)
    for k in range(start, i):
        if np.allclose(demo[k].gripper_touch_forces, 0, atol=delta) != \
            np.allclose(demo[i].gripper_touch_forces, 0, atol=delta):
            return True
    return False


def keypoint_discovery(demo: Demo,
                       task_str: str="",
                       stopping_delta=0.1,
                       method='heuristic') -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)

        if task_str == "":
            print('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
            return episode_keypoints
        else:
            # Determine goal keypoints based on gripper_touch_forces changes
            goal_keypoints = []
            for i in range(len(episode_keypoints) - 1):
                if task_str in CONTACT_BASE_CATEGORIES:
                    if touch_change(demo, episode_keypoints[i]):
                        goal_keypoints.append(episode_keypoints[i])
                else:
                    if gripper_change(demo, episode_keypoints[i]) or touch_change(demo, episode_keypoints[i]):
                        goal_keypoints.append(episode_keypoints[i])
            goal_keypoints.append(episode_keypoints[-1])  # The last keypoint is always a goal
            
            print('Found %d keypoints and %d goal keypoints.' % (len(episode_keypoints), len(goal_keypoints)),
                        episode_keypoints, goal_keypoints)
            return goal_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError

def create_replay(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    voxel_sizes,
    replay_size=3e5,
):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512
    lang_emb_dim_t5 = 4096
    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_depth" % cname,
                (
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement(
                "rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32
            ),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement(
                "lang_goal_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal_embs_bbox",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal_embs_t5",
                (
                    max_token_seq_len,
                    lang_emb_dim_t5,
                ),  # extracted from T5's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal_embs_t5_bbox",
                (
                    max_token_seq_len,
                    lang_emb_dim_t5,
                ),  # extracted from T5's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
            ReplayElement("sub_goal_trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement("sub_goal_rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32),
            ReplayElement("sub_goal_gripper_pose", (gripper_pose_size,), np.float32),  # 6-DoF gripper pose
            ReplayElement("sub_goal_wpt", (3,), np.float32),  # 3D world point
        ]
    )

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
    ]

    replay_buffer = (
        UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
            disk_saving=disk_saving,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    )
    return replay_buffer


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )


# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb

import bisect
def find_first_greater_index(episode_keypoints, sample_frame):
    index = bisect.bisect_right(episode_keypoints, sample_frame)
    return index if index < len(episode_keypoints) else -1

def get_current_timestep(episode_keypoints, subgoal_keypoints, sample_frame):
    next_keypoint_idx = find_first_greater_index(episode_keypoints, sample_frame)
    next_subgoal_keypoints_idx = find_first_greater_index(subgoal_keypoints, sample_frame)

    if next_subgoal_keypoints_idx == 0:
        start_idx = 0
    elif next_subgoal_keypoints_idx >= 1:
        start_idx = episode_keypoints.index(subgoal_keypoints[next_subgoal_keypoints_idx-1]) + 1

    return next_keypoint_idx - start_idx

# add individual data points to a replay
def _add_keypoints_to_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],
    sub_goal_keypoints: List[int],
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    clip_model=None,
    t5_embedder=None,
    device="cpu",
    sub_goal_episode_keypoints: List[int] = None,
):
    prev_action = None
    obs = inital_obs
    for k in range(
        next_keypoint_idx, len(episode_keypoints)
    ):  # confused here, it seems that there are many similar samples in the replay
        keypoint = episode_keypoints[k]
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(
            obs_tp1,
            obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )
        sub_goal_keypoint = sub_goal_episode_keypoints[k]
        sub_goal_obs_tp1 = demo[sub_goal_keypoint]
        sub_goal_obs_tm1 = demo[max(0, sub_goal_keypoint - 1)]
        (
            sub_goal_trans_indicies,
            sub_goal_rot_grip_indicies,
            sub_goal_ignore_collisions,
            sub_goal_action,
            sub_goal_attention_coordinates,
        ) = _get_action(
            sub_goal_obs_tp1,
            sub_goal_obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )

        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * 1.0 if terminal else 0
        timestep = get_current_timestep(episode_keypoints, sub_goal_keypoints, sample_frame)

        obs_dict = extract_obs(
            obs,
            CAMERAS,
            t=timestep,
            # t=k - next_keypoint_idx,
            # prev_action=prev_action,
            episode_length=15,
        )
        # print('description', description)
        # print("Focus on red bounding box, " + description)
        tokens = clip.tokenize([description, "Focus on red bounding box, " + description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
        obs_dict["lang_goal_embs_bbox"] = lang_embs[1].float().detach().cpu().numpy()
        # print('lang_embs[0].shape', lang_embs[0].shape)
        # print('lang_embs[1].shape', lang_embs[1].shape)
        if t5_embedder is not None:
            embeddings, _ = t5_embedder.get_text_embeddings([description, "Focus on red bounding box, " + description])
            obs_dict["lang_goal_embs_t5"] = embeddings[0].float().detach().cpu().numpy()
            obs_dict["lang_goal_embs_t5_bbox"] = embeddings[1].float().detach().cpu().numpy()
            # print('embeddings[0].shape', embeddings[0].shape)
            # print('embeddings[1].shape', embeddings[1].shape)

        prev_action = np.copy(action)

        if k == 0:
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([description], dtype=object),
            "sub_goal_trans_action_indicies": sub_goal_trans_indicies,
            "sub_goal_rot_grip_action_indicies": sub_goal_rot_grip_indicies,
            "sub_goal_gripper_pose": sub_goal_obs_tp1.gripper_pose,
            "sub_goal_wpt": sub_goal_obs_tp1.gripper_pose[:3]
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        obs = obs_tp1
        sample_frame = keypoint

        break
    # final step
    # obs_dict_tp1 = extract_obs(
    #     obs_tp1,
    #     CAMERAS,
    #     t=k + 1 - next_keypoint_idx,
    #     prev_action=prev_action,
    #     episode_length=25,
    # )
    # obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
    # obs_dict_tp1["lang_goal_embs_bbox"] = lang_embs[1].float().detach().cpu().numpy()
    # if t5_embedder is not None:
    #     embeddings, _ = t5_embedder.get_text_embeddings([description, "Focus on red bounding box, " + description])
    #     obs_dict_tp1["lang_goal_embs_t5"] = embeddings[0].float().detach().cpu().numpy()
    #     obs_dict_tp1["lang_goal_embs_t5_bbox"] = embeddings[1].float().detach().cpu().numpy()

    # obs_dict_tp1.pop("wrist_world_to_cam", None)
    # obs_dict_tp1.update(final_obs)
    # replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)


def fill_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    clip_model=None,
    t5_embedder=None,
    device="cpu",
):

    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            print(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)

    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        print("Filling replay ...")
        for d_idx in range(start_idx, start_idx + num_demos):
            try:
                demo = get_stored_demo(data_path=data_path, index=d_idx)
                print("Filling demo %d in %s" % (d_idx, data_path))
            except:
                print(f"Demo {d_idx} not found in {data_path}")
                continue

            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)
                try:
                    descs = descs['oracle_half']
                except:
                    pass

            # extract keypoints
            episode_keypoints = keypoint_discovery(demo)
            if episode_keypoints[0] == 1:
                episode_keypoints = episode_keypoints[1:]
            sub_goal_keypoints = keypoint_discovery(demo, task_str=task)
            sub_goal_episode_keypoints = []
            sub_goal_episode_keypoints_index = []
            # Populate goal action list for each keypoint with the nearest future goal
            goal_index = 0
            for i in range(len(episode_keypoints)):
                while goal_index < len(sub_goal_keypoints) and episode_keypoints[i] > sub_goal_keypoints[goal_index]:
                    goal_index += 1
                if goal_index == len(sub_goal_keypoints):
                    break
                sub_goal_episode_keypoints.append(sub_goal_keypoints[goal_index])
                sub_goal_episode_keypoints_index.append(goal_index)
            # sub_goal_episode_keypoints.append(sub_goal_keypoints[-1])
            print('episode_keypoints', episode_keypoints)
            print('sub_goal_episode_keypoints', sub_goal_episode_keypoints)
            print('sub_goal_episode_keypoints_index', sub_goal_episode_keypoints_index)
            next_keypoint_idx = 0
            for i in range(len(demo) - 1):
                if not demo_augmentation and i > 0:
                    break
                if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                    continue

                obs = demo[i]
                
                # if our starting point is past one of the keypoints, then remove it
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                if next_keypoint_idx == len(episode_keypoints):
                    break
                try:
                    desc = descs[0].split('\n')[sub_goal_episode_keypoints_index[next_keypoint_idx]]
                    print('desc:', desc)
                except:
                    print('desc not found for demo %d in %s' % (d_idx, data_path))
                    continue
                _add_keypoints_to_replay(
                    replay,
                    task,
                    task_replay_storage_folder,
                    d_idx,
                    i,
                    obs,
                    demo,
                    episode_keypoints,
                    sub_goal_keypoints,
                    cameras,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                    crop_augmentation,
                    next_keypoint_idx=next_keypoint_idx,
                    description=desc,
                    clip_model=clip_model,
                    t5_embedder=t5_embedder,
                    device=device,
                    sub_goal_episode_keypoints=sub_goal_episode_keypoints,
                )

        # save TERMINAL info in replay_info.npy
        task_idx = replay._task_index[task]
        with open(
            os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        ) as fp:
            np.save(
                fp,
                replay._store["terminal"][
                    replay._task_replay_start_index[
                        task_idx
                    ] : replay._task_replay_start_index[task_idx]
                    + replay._task_add_count[task_idx].value
                ],
            )

        print("Replay filled with demos.")
