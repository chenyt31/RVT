# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from typing import List
import numpy as np
from rlbench.backend.observation import Observation
from rvt.libs.peract.helpers.custom_rlbench_env import CustomMultiTaskRLBenchEnv
from rvt.utils.rvt_utils import gripper_change, keypoint_discovery

class CustomMultiTaskRLBenchEnv2(CustomMultiTaskRLBenchEnv):
    def __init__(self, *args, **kwargs):
        super(CustomMultiTaskRLBenchEnv2, self).__init__(*args, **kwargs)

    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        return self._previous_obs_dict

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )[0]

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict

    def get_ground_truth_action(self, i, keypoint_func='heuristic_goal', stopping_delta=0.1):
        all_action = np.empty((0, 9))
        for i in range(len(self._d._observations)):
            step_action = np.hstack([self._d._observations[i].gripper_pose, self._d._observations[i].gripper_open, self._d._observations[i].ignore_collisions])
            all_action = np.vstack([all_action,step_action])
        all_idx = np.array(list(range(len(self._d._observations))))

        keypoint_action = np.empty((0, 9))
        if keypoint_func == 'heuristic':
            keypoints_idx = keypoint_discovery(self._d._observations, stopping_delta=stopping_delta)
            for i in keypoints_idx:
                step_action = np.hstack([self._d._observations[i].gripper_pose, self._d._observations[i].gripper_open, self._d._observations[i].ignore_collisions])
                keypoint_action = np.vstack([keypoint_action,step_action])
        elif keypoint_func == 'heuristic_goal':
            keypoints_idx = keypoint_discovery(self._d._observations, stopping_delta=stopping_delta)
            for i in keypoints_idx:
                step_action = np.hstack([self._d._observations[i].gripper_pose, self._d._observations[i].gripper_open, self._d._observations[i].ignore_collisions])
                keypoint_action = np.vstack([keypoint_action,step_action])
                # Determine goal keypoints based on gripper_touch_forces changes
            goal_keypoints = []
            for i in range(len(keypoints_idx) - 1):
                if gripper_change(self._d._observations, keypoints_idx[i]):
                    goal_keypoints.append(keypoints_idx[i])
            goal_keypoints.append(keypoints_idx[-1])  # The last keypoint is always a goal
        
            # Populate goal action list for each keypoint with the nearest future goal
            goal_index = 0
            keypoint_action = np.empty((0, 3))
            for i in range(len(keypoints_idx)):
                while goal_index < len(goal_keypoints) and keypoints_idx[i] > goal_keypoints[goal_index]:
                    goal_index += 1
                if goal_index == len(goal_keypoints):
                    break
                goal_action = self._d._observations[goal_keypoints[goal_index]].gripper_pose[:3]
                keypoint_action = np.vstack([keypoint_action,goal_action])

        return keypoint_action, keypoints_idx, all_action, all_idx

