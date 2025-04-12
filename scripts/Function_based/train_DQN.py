"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.DQN import DQN

from tqdm import tqdm
import wandb

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
    learning_rate = 0.0001
    hidden_dim = 64
    n_episodes = 5000
    initial_epsilon = 1.0
    epsilon_decay = 0.9995  
    final_epsilon = 0.01
    discount = 0.95
    buffer_size = 1000
    batch_size = 256


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

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "DQN"

    agent = DQN(
        device=device,
        num_of_action=num_of_action,
        action_range=action_range,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        initial_epsilon = initial_epsilon,
        epsilon_decay = epsilon_decay,
        final_epsilon = final_epsilon,
        discount_factor = discount,
        buffer_size = buffer_size,
        batch_size = batch_size,
    )

    moving_avg_window = deque(maxlen=100)  # For smoothing rewards
    moving_avg_window2 = deque(maxlen=100) # For smoothing step
    moving_avg_window3 = deque(maxlen=100) # For smoothing loss

    # Start a new wandb run to track this script.
    wandb.init(
        # Set the wandb project where this run will be logged.
        project="DRL_HW3",
        name="DQN_test"
    ) 

    # reset environment
    timestep = 0
    sum_reward = 0
    moving_avg_loss = 0
    loss = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        # with torch.inference_mode():

        for episode in tqdm(range(n_episodes)):
            cumulative_rewards, steps, loss = agent.learn(env)

            cumulative_reward = sum(cumulative_rewards) / len(cumulative_rewards)
            step = sum(steps) / len(steps)

            moving_avg_window.append(cumulative_reward)
            moving_avg_reward = sum(moving_avg_window) / len(moving_avg_window)

            moving_avg_window2.append(step)
            moving_avg_step = sum(moving_avg_window2) / len(moving_avg_window2)

            if loss != None:
                moving_avg_window3.append(loss)
                moving_avg_loss = sum(moving_avg_window3) / len(moving_avg_window3)

            wandb.log({
                "avg_reward" : moving_avg_reward,
                "reward" : cumulative_reward,
                "epsilon" : agent.epsilon,
                "avg_step" : moving_avg_step,
                "step" : step,
                "avg_loss" : moving_avg_loss,
                "loss" : loss,
            })

            sum_reward += cumulative_reward
            if episode % 100 == 0:
                print("avg_score: ", sum_reward / 100.0)
                sum_reward = 0
                print(agent.epsilon)

                # Save Q-Learning agent
                model_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}.pt"
                full_path = os.path.join(f"w/{task_name}", Algorithm_name)
                agent.save_model(full_path, model_file)
        
        print('Complete')
        agent.plot_durations(show_result=False)
        plt.ioff()
        plt.show()
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    # close the simulator
    wandb.finish()
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()