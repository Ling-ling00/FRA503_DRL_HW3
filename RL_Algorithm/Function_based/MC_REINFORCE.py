from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
import os

class MC_REINFORCE_network(nn.Module):
    """
    Neural network for the MC_REINFORCE algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        # ========= put your code here ========= #
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1)  # Output probabilities
        )
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor representing action probabilities.
        """
        # ========= put your code here ========= #
        return self.net(x)
        # ====================================== #

class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            discount_factor: float = 0.95,
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            device (torch.device): Device to run computations on.
            num_of_action (int): Number of discrete actions.
            action_range (list): Range for continuous scaled actions.
            hidden_dim (int): Size of the hidden layer in the neural network.
            dropout (float): Dropout rate in the network.
            learning_rate (float): Learning rate for optimizer.
            discount_factor (float): Discount factor for future rewards.
        """     

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.LR = learning_rate

        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device = device
        self.episode_durations = []

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
    
    def calculate_stepwise_returns(self, rewards):
        """
        Compute stepwise returns for the trajectory.

        Args:
            rewards (list(float)): List of rewards obtained in the episode.
        
        Returns:
            Tensor: Normalized stepwise returns.
        """
        # ========= put your code here ========= #
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.discount_factor * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
        # ====================================== #

    def select_action(self, state):
        """
        Selects an action based on the current policy.
        
        Args:
        state (Tensor): The current state of the environment.

        Returns:
            Tuple[int, Tensor, distributions.Categorical]:
                - int: Index of the selected action.
                - Tensor: Scaled continuous action.
                - Categorical: Torch distribution object used for sampling/log_probs.
        """
        # ========= put your code here ========= #
        with torch.no_grad():
            probs = self.policy_net(state).to(self.device)
            dist = distributions.Categorical(probs)
            action_idx = dist.sample()
            action = self.scale_action(action_idx.item())
            return action_idx.item(), action, dist
        # ====================================== #

    def generate_trajectory(self, env):
        """
        Generate a trajectory by interacting with the environment. (can using with multi environments)

        Args:
            env: The environment object.
        
        Returns:
            Tuple(List[float], List[int], List[Tensor], List[Tensor], List[List[Tuple]]):
            - List[float]: Total return for each environment.
            - List[int]: Episode length for each environment.
            - List[Tensor]: Discounted and normalized return for each step in each environment.
            - List[Tensor]: Log probabilities of the actions taken at each step per environment.
            - List[List[Tuple]]: Full trajectory (state, action, reward) per environment.
           
        """
        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Store state-action-reward history (list)
        # Store log probabilities of actions (list)
        # Store rewards at each step (list)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        obs_list, _ = env.reset()
        state_list = self.extract_policy_state(obs_list)
        num_envs = len(state_list)
        dones = [False] * num_envs
        cumulative_rewards = [0.0] * num_envs
        steps = [0] * num_envs
        log_probs_list = [[] for _ in range(num_envs)]
        rewards_list = [[] for _ in range(num_envs)]
        trajectory_list = [[] for _ in range(num_envs)]
        timestep = 0
        # ====================================== #
        
        # ===== Collect trajectory through agent-environment interaction ===== #
        while not all(dones):
            
            # Predict action from the policy network
            # ========= put your code here ========= #
            actions_idx = []
            actions = []
            dists = []

            for i, state in enumerate(state_list):
                if dones[i]:
                    actions_idx.append(0)
                    actions.append(torch.tensor([[0.0]], dtype=torch.float32))
                    dists.append(None)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action_idx, action, dist = self.select_action(state_tensor)
                    actions.append(action)
                    actions_idx.append(action_idx)
                    dists.append(dist)
            actions = torch.cat(actions, dim=0).to(self.device)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            next_obs_list, rewards, terminations, truncations, _ = env.step(actions)
            next_state_list = self.extract_policy_state(next_obs_list)
            # ====================================== #

            # Store action log probability reward and trajectory history
            # ========= put your code here ========= #
            for i in range(num_envs):
                if not dones[i]:
                    done = bool(terminations[i].item()) or bool(truncations[i].item())
                    log_probs_list[i].append(dists[i].log_prob(torch.tensor(actions_idx[i]).to(self.device)))
                    rewards_list[i].append(rewards[i].item())
                    trajectory_list[i].append((state_list[i], actions_idx[i], rewards[i].item()))
                    cumulative_rewards[i] += rewards[i].item()
                    steps[i] += 1
                    dones[i] = done
                    state_list[i] = next_state_list[i]
            # ====================================== #
            
            # Update state

            timestep += 1
            # if done:
            #     self.plot_durations(timestep)
            #     break

        # ===== Stack log_prob_actions &  stepwise_returns ===== #
        # ========= put your code here ========= #
        all_returns = []
        all_log_probs = []
        for i in range(num_envs):
            stepwise_returns = self.calculate_stepwise_returns(rewards_list[i])
            all_returns.append(stepwise_returns)
            all_log_probs.append(torch.stack(log_probs_list[i]).squeeze(-1))

        return cumulative_rewards, steps, all_returns, all_log_probs, trajectory_list
        # ====================================== #
    
    def calculate_loss(self, returns_batch, log_probs_batch):
        """
        Compute the loss for policy optimization.

        Args:
            returns_batch (List[Tensor]): List of return tensors for each trajectory.
            log_probs_batch (List[Tensor]): List of log-probability tensors for each trajectory.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #
        loss = torch.tensor(0.0, device=self.device)
        for R, log_probs in zip(returns_batch, log_probs_batch):
            loss += -(log_probs * R).sum()
        loss /= sum(len(R) for R in returns_batch)
        return loss
        # ====================================== #

    def update_policy(self, returns_batch, log_probs_batch):
        """
        Update the policy using the calculated loss.

        Args:
            returns_batch (List[Tensor]): List of return tensors for each trajectory.
            log_probs_batch (List[Tensor]): List of log-probability tensors for each trajectory.
        
        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        loss = self.calculate_loss(returns_batch, log_probs_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        # ====================================== #
    
    def learn(self, env):
        """
        Train the agent on a single episode. (can using with multi environments)

        Args:
            env: The environment to train in.
        
        Returns:
            Tuple(List[float], List[int], float, List[List[Tuple]]):
                - List[float]: Total return per environment.
                - List[int]: Episode length per environment.
                - float: Policy loss after the update.
                - List[List[Tuple]]: Trajectory of (state, action, reward) per env.
        """
        # ========= put your code here ========= #
        self.policy_net.train()
        episode_return, step, stepwise_returns, log_prob_actions, trajectory = self.generate_trajectory(env)
        loss = self.update_policy(stepwise_returns, log_prob_actions)
        return episode_return, step, loss, trajectory
        # ====================================== #


    # Consider modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    # ================================================================================== #

    def save_model(self, path, filename):
        """
        Save model network.

        Args:
            path (str): Directory to save model.
            filename (str): Name of the file.
        """
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, full_path)

    def load_model(self, path, filename):
        """
        Load model network.

        Args:
            path (str): Directory to save model.
            filename (str): Name of the file.
        """
        full_path = os.path.join(path, filename)
        checkpoint = torch.load(full_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])