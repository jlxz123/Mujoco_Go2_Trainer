import os
import random
import time
from dataclasses import dataclass, MISSING
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import env
from train import Agent

@dataclass
class Args:
    env_id: str = "Unitree-Go2"
    """环境名称"""
    capture_video: bool = True
    """是否用视频记录agent表现(check out `videos` folder)"""
    gamma: float = 0.99
    """奖励折扣计算汇报"""
    model_path: str = "./runs/Unitree-Go2__ppo__1__1744465327/ppo.pth"


def make_env(env_id, capture_video, video_path, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{video_path}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def play(args: Args):
    video_path = args.env_id + "_play_model__" + args.model_path[len("./runs/Unitree-Go2_"):-len(".pth")]
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.capture_video, video_path, args.gamma)])
    device = torch.device("cuda")
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    start_time = time.time()
    terminated = False
    while (time.time() - start_time) <= 60 and not terminated:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, terminated, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)
    
