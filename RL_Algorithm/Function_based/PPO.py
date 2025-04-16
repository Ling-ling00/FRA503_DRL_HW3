import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.distributions as distributions
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm
import os

class PPO_Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, is_discrete=True):
        """
        PPO_Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            is_discrete (bool): To select output type as a discrete or continuous.
        """
        super(PPO_Actor, self).__init__()

        # ========= put your code here ========= #
        self.is_discrete = is_discrete

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if self.is_discrete:
            self.fc1 = nn.Linear(hidden_dim, output_dim)  # for logits → Categorical
        else:
            self.mu_net = nn.Linear(hidden_dim, 1)       # for mean
            self.std_net = nn.Parameter(torch.zeros(1))   # learnable std

        self.init_weights()
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Policy Probability.(in continuous return as mu and std of normal distribution)
        """
        # ========= put your code here ========= #
        x = self.net(state)
        if self.is_discrete:
            logits = self.fc1(x)
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            mu = self.mu_net(x)
            std = self.std_net.exp().expand_as(mu)
            return mu, std
        # ====================================== #

class PPO_Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        """
        PPO_Critic network for Value approximation.

        Args:
            state_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
        """
        super(PPO_Critic, self).__init__()

        # ========= put your code here ========= #
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.init_weights()
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for Value function estimation.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Estimated Value function V(s).
        """
        # ========= put your code here ========= #
        return self.net(state)
        # ====================================== #

class PPO_Actor_Critic(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 1,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                learning_rate: float = 0.01,
                discount_factor: float = 0.95,
                clip_epsilon=0.2,
                epochs=10,
                batch_size=64,
                is_discrete=True
                ):
        """
        PPO_Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            clip_epsilon (float, optional): Clipping range for the policy update loss. Defaults to 0.2.
            epochs (int, optional): Number of optimization steps per policy update. Defaults to 10.
            batch_size (int, optional): Size of training batches. Defaults to 64.
            is_discrete (bool, optional): Whether the action space is discrete (True) or continuous (False). Defaults to True.
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.is_discrete = is_discrete

        self.actor = PPO_Actor(n_observations, hidden_dim, num_of_action, is_discrete).to(device)
        self.critic = PPO_Critic(n_observations, hidden_dim).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.
        # ====================================== #

        super(PPO_Actor_Critic, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
        )

    def select_action(self, state):
        """
        Selects an action based on the current policy.
        
        Args:
        state (Tensor): The current state of the environment.

        Returns:
            Tuple[int or Tensor, Tensor, float]:
                - Discrete: (action_idx, scaled_action, log_prob)
                - Continuous: (raw_action, scaled_action, log_prob)
        """
        # ========= put your code here ========= #
        with torch.no_grad():
            if self.is_discrete:
                probs = self.actor(state)
                dist = distributions.Categorical(probs)
                action_idx = dist.sample()
                action = self.scale_action(action_idx.item())
                log_prob = dist.log_prob(action_idx)
                return action_idx.item(), action, log_prob.item()
            else:
                mu, std = self.actor(state)
                dist = distributions.Normal(mu, std)
                action_raw = dist.sample()
                action = torch.tanh(action_raw) * self.action_range[1]
                log_prob = dist.log_prob(action_raw).sum(-1)
                log_prob -= (2*(np.log(2) - action_raw - F.softplus(-2*action_raw))).sum(-1)
                return action_raw, action, log_prob.item()
        # ====================================== #
    
    def compute_advantages(self, rewards, values, dones, next_value):
        """
        Compute returns and advantages.

        Args:
            rewards (list of float): Collected rewards from the episode.
            values (list of float): Estimated value function for each state.
            dones (list of bool): Done flags indicating episode termination.
            next_value (float): Estimated value of the final next state.

        Returns:
            Tuple[Tensor, Tensor]:
                - returns: Discounted cumulative rewards.
                - advantages: Advantage estimates (returns - values).
        """
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.discount_factor * R * (1 - dones[step])
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        advantages = returns - values
        return returns, advantages
    
    def generate_trajectory(self, env):
        """
        Collect trajectories from all environments using the current policy. (can using with multi environments)

        Args:
            env: The environment object.

        Returns:
            Tuple(states, actions, log_probs, rewards, dones, values, next_values, episode_return, episode_lenght):
        """
        obs_list, _ = env.reset()
        state_list = self.extract_policy_state(obs_list)
        num_envs = len(state_list)
        
        states = [[] for _ in range(num_envs)]
        actions_list = [[] for _ in range(num_envs)]
        log_probs_list = [[] for _ in range(num_envs)]
        rewards_list = [[] for _ in range(num_envs)]
        dones_list = [[] for _ in range(num_envs)]
        values_list = [[] for _ in range(num_envs)]
        actions_idx_list = [[] for _ in range(num_envs)]
        cumulative_rewards = [0.0] * num_envs
        done_flags = [False] * num_envs
        steps = [0] * num_envs

        while not all(done_flags):
            current_actions_idx, current_actions, current_log_probs, current_values = [], [], [], []
            for i, state in enumerate(state_list):
                if done_flags[i]:
                    current_actions.append(torch.tensor([[0.0]], dtype=torch.float32).to(self.device))
                    current_log_probs.append(torch.tensor([0.0]))
                    current_values.append(0.0)
                    current_actions_idx.append(0)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    value = self.critic(state_tensor).item()
                    action_idx, action, log_prob = self.select_action(state_tensor)

                    current_actions_idx.append(action_idx)
                    current_actions.append(action.to(self.device))
                    current_log_probs.append(log_prob)
                    current_values.append(value)
            
            actions = torch.cat(current_actions, dim=0).to(self.device)
            next_obs_list, rewards, terminations, truncations, _ = env.step(actions)
            next_state_list = self.extract_policy_state(next_obs_list)

            for i in range(num_envs):
                if not done_flags[i]:
                    cumulative_rewards[i] += rewards[i].item()
                    done = bool(terminations[i].item()) or bool(truncations[i].item())
                    log_probs_list[i].append(current_log_probs[i])
                    rewards_list[i].append(rewards[i].item())
                    dones_list[i].append(done)
                    values_list[i].append(current_values[i])
                    actions_list[i].append(current_actions[i])
                    actions_idx_list[i].append(current_actions_idx[i])
                    states[i].append(state_list[i].copy())
                    steps[i] += 1
                    done_flags[i] = done
                    state_list[i] = next_state_list[i]

        next_value_list = [0.0]*num_envs
        return states, actions_list, actions_idx_list, log_probs_list, rewards_list, dones_list, values_list, next_value_list, cumulative_rewards, steps

    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        update actor and critic using PPO clipped objective.

        Args:
            states (np.ndarray): List of states from trajectory.
            actions (np.ndarray): Taken actions or action indices.
            old_log_probs (np.ndarray): Log-probs of actions before the update.
            returns (Tensor): Target returns for critic.
            advantages (Tensor): Computed advantages for actor.

        Returns:
            Tuple[float, float]:
                - float: Actor loss.
                - float: Critic loss.
        """
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)

        for _ in range(self.epochs):
            if self.is_discrete:
                probs = self.actor(states)
                dist = distributions.Categorical(probs)
                log_probs = dist.log_prob(actions)
            else:
                mu, std = self.actor(states)
                dist = distributions.Normal(mu, std)
                log_probs = dist.log_prob(actions).sum(-1)
                log_probs -= (2*(np.log(2) - actions - F.softplus(-2*actions))).sum(-1)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            values = self.critic(states).squeeze(-1)
            critic_loss = F.mse_loss(values, returns)

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        return actor_loss.item(), critic_loss.item()

    def learn(self, env):
        """
        Train the PPO agent for 1 episode from the environment. (can using with multi environments)

        Args:
            env: The environment to collect trajectories from.

        Returns:
            Tuple(List[float], List[int], List[float], List[float]):
                - List[float]: Total reward from each environment.
                - List[int]: Total steps per environment.
                - List[float]: Loss values for actor update.
                - List[float]: Loss values for critic update.
        """
        states, actions, actions_idx, log_probs, rewards, dones, values, next_values, cumulative_rewards, steps = self.generate_trajectory(env)
        all_actor_loss, all_critic_loss = [], []
        for i in range(len(states)):
            returns, advantages = self.compute_advantages(rewards[i], values[i], dones[i], next_values[i])
            actor_loss, critic_loss = self.update(states[i], actions_idx[i], log_probs[i], returns, advantages)
            all_actor_loss.append(actor_loss)
            all_critic_loss.append(critic_loss)
        return cumulative_rewards, steps, all_actor_loss, all_critic_loss
    
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
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
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
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
