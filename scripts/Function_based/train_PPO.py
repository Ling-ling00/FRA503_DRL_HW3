"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.PPO import PPO_Actor_Critic

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
    num_of_action = 1
    action_range = [-25, 25]
    learning_rate = 0.001
    hidden_dim = 128
    n_episodes = 5000
    discount = 0.95
    clip_epsilon = 0.2
    epochs = 10
    batch_size = 256
    is_discrete = False


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
    Algorithm_name = "PPO"

    agent = PPO_Actor_Critic(
        device=device,
        num_of_action=num_of_action,
        action_range=action_range,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        discount_factor = discount,
        clip_epsilon = clip_epsilon,
        epochs = epochs,
        batch_size = batch_size,
        is_discrete = is_discrete,
    )

    moving_avg_window = deque(maxlen=100)  # For smoothing rewards
    moving_avg_window2 = deque(maxlen=100) # For smoothing step
    moving_avg_window3 = deque(maxlen=100) # For smoothing loss actor
    moving_avg_window4 = deque(maxlen=100) # For smoothing loss critic

    # Start a new wandb run to track this script.
    wandb.init(
        # Set the wandb project where this run will be logged.
        project="DRL_HW3",
        name="PPO_test"
    ) 

    # reset environment
    timestep = 0
    sum_reward = 0
    moving_avg_loss_a = 0
    actor_loss = 0
    critic_loss = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        # with torch.inference_mode():

        for episode in tqdm(range(n_episodes)):
            cumulative_rewards, steps, actor_loss, critic_loss = agent.learn(env)

            cumulative_reward = sum(cumulative_rewards) / len(cumulative_rewards)
            step = sum(steps) / len(steps)

            moving_avg_window.append(cumulative_reward)
            moving_avg_reward = sum(moving_avg_window) / len(moving_avg_window)

            moving_avg_window2.append(step)
            moving_avg_step = sum(moving_avg_window2) / len(moving_avg_window2)

            if actor_loss != None:
                a_loss = sum(actor_loss) / len(actor_loss)
                moving_avg_window3.append(a_loss)
                moving_avg_loss_a = sum(moving_avg_window3) / len(moving_avg_window3)

            if critic_loss != None:
                c_loss = sum(critic_loss) / len(critic_loss)
                moving_avg_window4.append(c_loss)
                moving_avg_loss_c = sum(moving_avg_window4) / len(moving_avg_window4)

            wandb.log({
                "avg_reward" : moving_avg_reward,
                "reward" : cumulative_reward,
                "avg_step" : moving_avg_step,
                "step" : step,
                "actor_avg_loss" : moving_avg_loss_a,
                "actor_loss" : a_loss,
                "critic_avg_loss" : moving_avg_loss_c,
                "critic_loss" : c_loss,
            })

            sum_reward += cumulative_reward
            if episode % 100 == 0:
                print("avg_score: ", sum_reward / 100.0)
                sum_reward = 0

                # Save Q-Learning agent
                model_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}.pt"
                full_path = os.path.join(f"model/{task_name}", Algorithm_name)
                agent.save_model(full_path, model_file)
        
        print('Complete')
        # agent.plot_durations(show_result=False)
        # plt.ioff()
        # plt.show()
            
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