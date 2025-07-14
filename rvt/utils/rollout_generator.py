# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling RVT or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from multiprocessing import Value
import os

import numpy as np
import torch
from rvt.models.remote_agent import WebsocketClientPolicyAgent
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from yarr.agents.agent import ActResult

class RolloutGenerator(object):

    def __init__(self, env_device = 'cuda:0'):
        self._env_device = env_device

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False):

        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
            # get ground-truth action sequence
            if replay_ground_truth:
                actions = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        for step in range(episode_length):

            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            if not replay_ground_truth:
                act_result = agent.act(step_signal.value, prepped_data,
                                    deterministic=eval)
            else:
                if step >= len(actions):
                    return
                act_result = ActResult(actions[step])

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return

    def generator_goal(self, step_signal: Value, env: Env, agent: Agent | WebsocketClientPolicyAgent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False,
                  visual_prompt_type=[],
                  visualize=False,
                  visualize_save_dir="",
                  agent_type='original',
                  ):

        if "bbox" in visual_prompt_type or "zoom_in" in visual_prompt_type:
            use_sub_goal = True
        else:
            use_sub_goal = False
        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
            # get ground-truth action sequence
            if use_sub_goal:
                goal_actions, keypoints_idx, all_action, all_idx = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()
        if agent_type != 'remote':
            agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        if visualize:
            visualize_save_dir=os.path.join(visualize_save_dir, env._lang_goal)
            if not os.path.exists(visualize_save_dir):
                os.makedirs(visualize_save_dir)
        for step in range(episode_length):
            if agent_type == 'remote':
                prepped_data = {k:np.array([v]) for k, v in obs_history.items()}
                prepped_data["lang_goal"] = env._lang_goal
                prepped_data["lang_goal_bbox"] = 'Focus on red bounding box, ' + env._lang_goal
                if use_sub_goal:
                    prepped_data["sub_goal_wpt"] = np.array([goal_actions[step%len(goal_actions)]], dtype=np.float32)
                prepped_data['visualize'] = visualize
                prepped_data['visualize_save_dir'] = visualize_save_dir   
                prepped_data['visual_prompt_type'] = visual_prompt_type 
                action = agent.act(prepped_data)
                act_result = ActResult.from_dict(action['action'])
            else:
                prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}

                prepped_data["lang_goal"] = env._lang_goal
                prepped_data["lang_goal_bbox"] = 'Focus on red bounding box, ' + env._lang_goal

                if not use_sub_goal:
                    act_result = agent.act(step_signal.value, prepped_data,
                                        deterministic=eval,
                                        visualize=visualize,
                                        visualize_save_dir=visualize_save_dir,
                                        )
                else:
                    prepped_data["sub_goal_wpt"] = torch.tensor(np.array([goal_actions[step%len(goal_actions)]], dtype=np.float32), device=self._env_device)
                    act_result = agent.act(step_signal.value, prepped_data,
                                        deterministic=eval,
                                        visualize=visualize,
                                        visual_prompt_type=visual_prompt_type,
                                        visualize_save_dir=visualize_save_dir,
                                        )

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    if agent_type == 'remote':
                        prepped_data = {k:np.array([v]) for k, v in obs_history.items()}
                        prepped_data["lang_goal"] = env._lang_goal
                        prepped_data["lang_goal_bbox"] = 'Focus on red bounding box, ' + env._lang_goal
                        if use_sub_goal:
                            prepped_data["sub_goal_wpt"] = np.array([goal_actions[step%len(goal_actions)]], dtype=np.float32)
                        prepped_data['visualize'] = visualize
                        prepped_data['visualize_save_dir'] = visualize_save_dir   
                        prepped_data['visual_prompt_type'] = visual_prompt_type 
                        action = agent.act(prepped_data)
                        act_result = ActResult.from_dict(action['action'])
                    else:
                        prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                        prepped_data["lang_goal"] = env._lang_goal
                        prepped_data["lang_goal_bbox"] = 'Focus on red bounding box, ' + env._lang_goal
                        if use_sub_goal:
                            prepped_data["sub_goal_wpt"] = None
                        act_result = agent.act(step_signal.value, prepped_data,
                                            deterministic=eval,
                                            visualize=visualize,
                                            visual_prompt_type=visual_prompt_type,
                                            visualize_save_dir=visualize_save_dir,
                                            )
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
