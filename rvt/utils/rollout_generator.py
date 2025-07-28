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
from pyrep.objects.dummy import Dummy

class RolloutGenerator(object):

    def __init__(self, env_device = 'cuda:0'):
        self._env_device = env_device

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent, # type: ignore
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

    def _init_episode(self, env:Env, eval, demo_seed, use_sub_goal, planner_type):
        if eval:
            obs = env.reset_to_demo(demo_seed)
            goal_actions = keypoints_idx = all_action = all_idx = None
            if use_sub_goal:
                goal_actions, keypoints_idx, all_action, all_idx = env.get_ground_truth_action(demo_seed)
        else:
            obs = env.reset()
            goal_actions = None

        oracle_lang_goal = []
        if planner_type == "oracle":
            oracle_lang_goal = env.descriptions["oracle_half"][0].split('\n')
        return obs, oracle_lang_goal, goal_actions

    def _get_home_pose(self):
        home_pose = np.array(Dummy("Panda_tip").get_pose())
        return np.concatenate((home_pose, [1, 0]))
    
    def _init_obs_history(self, obs, timesteps):
        return {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}

    def _prepare_vis_dir(self, base_dir, goal_str):
        dir_path = os.path.join(base_dir, goal_str)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    def _handle_gripper(self, act_result, prev_gripper, planner_type, oracle_index, oracle_lang_goal):
        curr_gripper = 1 - int(act_result.action[-2])
        return_home = False

        if curr_gripper != prev_gripper:
            act_result.gripper_change = True
            if planner_type == "oracle":
                oracle_index = (oracle_index + 1) % len(oracle_lang_goal)
            if curr_gripper == 0:
                act_result.gripper_change_twice = True
                return_home = True

        return act_result, curr_gripper, oracle_index, return_home
    
    def _combine_obs_elements(self, obs, act_result):
        combined = dict(obs)
        combined.update({k: np.array(v) for k, v in act_result.observation_elements.items()})
        combined.update({k: np.array(v) for k, v in act_result.replay_elements.items()})
        return combined
    
    def _prepare_data(self, obs_history, agent_type, use_sub_goal,
                  oracle_lang_goal, oracle_index, goal_actions, planner_type,
                  env, visualize, visualize_save_dir, visual_prompt_type):
        if agent_type == 'remote':
            data = {k: np.array([v]) for k, v in obs_history.items()}
        else:
            data = {k: torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}

        if planner_type == 'oracle':
            lang = oracle_lang_goal[oracle_index]
            data["lang_goal"] = lang
            data["lang_goal_bbox"] = 'Focus on red bounding box, ' + lang
            data["lang_goal_tokens"] = data["oracle_lang_goal_tokens"][:, :, oracle_index, :]
            data["lang_goal_tokens_bbox"] = data["oracle_lang_goal_tokens_bbox"][:, :, oracle_index, :]
            if use_sub_goal:
                data["sub_goal_wpt"] = np.array([goal_actions[oracle_index]], dtype=np.float32)
        elif planner_type == "vanilla":
            data["lang_goal"] = env._lang_goal
            data["lang_goal_bbox"] = 'Focus on red bounding box, ' + env._lang_goal

        data.update({
            'visualize': visualize,
            'visualize_save_dir': visualize_save_dir,
            'visual_prompt_type': visual_prompt_type,
        })
        return data

    def _run_agent(self, agent, agent_type, step_signal, prepped_data, eval):
        if agent_type == 'remote':
            action = agent.act(prepped_data)
            return ActResult.from_dict(action['action'])
        else:
            return agent.act(step_signal.value, prepped_data,
                            deterministic=eval,
                            visualize=prepped_data['visualize'],
                            visual_prompt_type=prepped_data.get('visual_prompt_type', []),
                            visualize_save_dir=prepped_data['visualize_save_dir'])
    
    def _final_step_act(self, obs_history, oracle_lang_goal, oracle_index, goal_actions,
                    planner_type, env, agent, step_signal, visualize,
                    visualize_save_dir, visual_prompt_type, agent_type,
                    eval, use_sub_goal, replay_transition, obs_tp1):
        if agent_type == 'remote':
            prepped_data = {k: np.array([v]) for k, v in obs_history.items()}
        else:
            prepped_data = {k: torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}

        if planner_type == "oracle":
            prepped_data["lang_goal"] = oracle_lang_goal[oracle_index]
            prepped_data["lang_goal_bbox"] = 'Focus on red bounding box, ' + oracle_lang_goal[oracle_index]
            prepped_data["lang_goal_tokens"] = prepped_data["oracle_lang_goal_tokens"][:, :, oracle_index, :]
            prepped_data['lang_goal_tokens_bbox'] = prepped_data["oracle_lang_goal_tokens_bbox"][:, :, oracle_index, :]
            if use_sub_goal:
                prepped_data["sub_goal_wpt"] = np.array([goal_actions[oracle_index]], dtype=np.float32)
        elif planner_type == "vanilla":
            prepped_data["lang_goal"] = env._lang_goal
            prepped_data["lang_goal_bbox"] = 'Focus on red bounding box, ' + env._lang_goal

        act_result = self._run_agent(agent, agent_type, step_signal, prepped_data, eval)
        agent_obs_tp1 = {k: np.array(v) for k, v in act_result.observation_elements.items()}
        obs_tp1.update(agent_obs_tp1)
        replay_transition.final_observation = obs_tp1
        return replay_transition

    def generator_goal(self, step_signal: Value, env: Env, agent: Agent | WebsocketClientPolicyAgent, # type: ignore
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False,
                  visual_prompt_type=[],
                  visualize=False,
                  visualize_save_dir="",
                  agent_type='original',
                  planner_type="oracle",
                  ):     

        use_sub_goal = 'bbox' in visual_prompt_type or 'zoom_in' in visual_prompt_type
        obs, oracle_lang_goal, goal_actions = self._init_episode(env, eval, eval_demo_seed, use_sub_goal, planner_type)
        home_pose = self._get_home_pose()
        if agent_type != 'remote':
            agent.reset()
        obs_history = self._init_obs_history(obs, timesteps)
        if visualize:
            visualize_save_dir = self._prepare_vis_dir(visualize_save_dir, env._lang_goal)

        prev_gripper = 0
        return_home = False
        oracle_index = 0
        for step in range(episode_length):
            if return_home:
                act_result = ActResult(home_pose)
                return_home = False
            else:
                prepped_data = self._prepare_data(obs_history, agent_type, use_sub_goal,
                                              oracle_lang_goal, oracle_index,
                                              goal_actions, planner_type, env,
                                              visualize, visualize_save_dir, visual_prompt_type)
                act_result = self._run_agent(agent, agent_type, step_signal, prepped_data, eval)

            act_result, curr_gripper, oracle_index, return_home = self._handle_gripper(
                act_result, prev_gripper, planner_type, oracle_index, oracle_lang_goal)
            prev_gripper = curr_gripper

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = step == episode_length - 1 and not transition.terminal
            if timeout:
                transition.terminal = True
                if "needs_reset" in transition.info:
                    transition.info["needs_reset"] = True

            obs_and_replay_elems = self._combine_obs_elements(obs, act_result)
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
                    replay_transition = self._final_step_act(obs_history, oracle_lang_goal, oracle_index, goal_actions,
                                                     planner_type, env, agent, step_signal, visualize,
                                                     visualize_save_dir, visual_prompt_type, agent_type,
                                                     eval, use_sub_goal, replay_transition, obs_tp1)

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
