"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.A2C import A2C_Actor_Critic

from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# from omni.isaac.lab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

steps_done = 0

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    num_of_action = 7
    action_range = [-25, 25]
    learning_rate = 0.00001
    hidden_dim = 64
    n_episodes = 10
    discount = 0.95
    batch_size = 64
    buffer_size = 1000
    is_discrete = True
    entropy_coef = 0.01

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print("device: ", device)

    agent = A2C_Actor_Critic(
        device=device,
        num_of_action=num_of_action,
        action_range=action_range,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        discount_factor = discount,
        buffer_size=buffer_size,
        batch_size = batch_size,
        is_discrete = is_discrete,
        entropy_coef = entropy_coef
    )

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "A2C" 
    exp_name = "learning_rate_0.00001"
    episode = 1900
    q_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}.pt"
    full_path = os.path.join(f"model/{task_name}", f"{Algorithm_name}/{exp_name}")
    agent.load_model(full_path, q_value_file)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    state_log = []
    steps = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
        
            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                state_log.append([])
                step = 0

                while not done:
                    # agent stepping
                    state = agent.extract_policy_state(obs)
                    state_tensor = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(agent.device)
                    action_idx, action = agent.select_action(state_tensor)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    done = terminated or truncated
                    obs = next_obs

                    state_log[-1].append(state[0])
                    step += 1
                steps.append(step)
                print(episode)

            max_step = np.argmax(steps)
            min_step = np.argmin(steps)

            episode_data = {
                "max_step_episode": int(max_step),
                "min_step_episode": int(min_step),
                "max_episode": [[float(x) for x in state] for state in state_log[max_step]],
                "min_episode": [[float(x) for x in state] for state in state_log[min_step]]
            }

            # Save to JSON file
            output_dir = os.path.join("results", Algorithm_name, exp_name)
            os.makedirs(output_dir, exist_ok=True)

            json_path = os.path.join(output_dir, f"ep{episode}_log.json")
            with open(json_path, "w") as f:
                json.dump(episode_data, f, indent=4)

            print(f"Saved episode data to {json_path}")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()