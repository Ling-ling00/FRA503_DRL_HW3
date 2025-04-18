import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.distributions as distributions
from RL_Algorithm.RL_base_function import BaseAlgorithm
import torch.nn.functional as F
import os

class A2C_Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, is_discrete=True):
        """
        A2C_Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            is_discrete (bool): To select output type as a discrete or continuous.
        """
        super(A2C_Actor, self).__init__()

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
            self.mu_net = nn.Linear(hidden_dim, 1)
            self.std_net = nn.Linear(hidden_dim, 1)

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
            log_std = self.std_net(x)
            std = torch.exp(log_std.clamp(-20, 2))  # stability
            return mu, std
        # ====================================== #

class A2C_Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        """
        A2C_Critic network for Value approximation.

        Args:
            state_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
        """
        super(A2C_Critic, self).__init__()

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

class A2C_Actor_Critic(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                learning_rate: float = 0.01,
                discount_factor: float = 0.95,
                buffer_size: int = 256,
                batch_size: int = 1,
                entropy_coef: float = 0.01,
                is_discrete = True
                ):
        """
        Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.actor = A2C_Actor(n_observations, hidden_dim, num_of_action, is_discrete).to(device)
        self.critic = A2C_Critic(n_observations, hidden_dim).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        self.is_discrete = is_discrete
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        # ====================================== #

        super(A2C_Actor_Critic, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    def select_action(self, state):
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
        state (Tensor): The current state of the environment.
        noise (float, optional): The standard deviation of noise for exploration. Defaults to 0.0.

        Returns:
            Tuple[Tensor, Tensor]: 
                - scaled_action: The final action after scaling.
                - clipped_action: The action before scaling but after noise adjustment.
        """
        # ========= put your code here ========= #
        with torch.no_grad():
            if self.is_discrete:
                probs = self.actor(state)
                dist = distributions.Categorical(probs)
                action_idx = dist.sample()
                action = self.scale_action(action_idx.item())
                return action_idx.item(), action
            else:
                mu, std = self.actor(state)
                base_dist = distributions.Normal(mu, std)
                dist = distributions.TransformedDistribution(base_dist, [distributions.TanhTransform(cache_size=1)])
                action = dist.sample()
                scaled_action = action * self.action_range[1]
                return action, scaled_action
        # ====================================== #
    
    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
                - next_state_batch (Tensor): The batch of next states received.
                - done_batch (Tensor): The batch of dones received.
        """
        # Ensure there are enough samples in memory before proceeding
        
        # Sample a batch from memory
        # ========= put your code here ========= #
        if len(self.memory) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample()
        non_final_mask = ~dones
        # non_final_next_states = next_states[non_final_mask]
        return non_final_mask, next_states, states, actions, rewards
        # ====================================== #

    def calculate_loss(self, non_final_mask, next_states, state_batch, action_batch, reward_batch):
        """
        Computes the loss for policy optimization.

        Args:
            - states (Tensor): The batch of current states.
            - actions (Tensor): The batch of actions taken.
            - rewards (Tensor): The batch of rewards received.
            - next_states (Tensor): The batch of next states received.
            - dones (Tensor): The batch of dones received.

        Returns:
            Tensor: Computed critic & actor loss.
        """
        # ========= put your code here ========= #
        value = self.critic(state_batch).squeeze(-1)  # Estimate the value of the current state
        next_values = self.critic(next_states).squeeze(-1).detach()
        non_final_mask = non_final_mask.float()

        # Target with full batch
        target = reward_batch + non_final_mask * self.discount_factor * next_values
        critic_loss = F.mse_loss(value, target)  # Calculate the critic loss

        advantage = (target - value).detach()

        if self.is_discrete:
            probs = self.actor(state_batch)
            dist = distributions.Categorical(probs)
            log_probs = dist.log_prob(action_batch)
            entropy = dist.entropy().sum(dim=-1)
        else:
            mu, std = self.actor(state_batch)
            base_dist = distributions.Normal(mu, std)
            dist = distributions.TransformedDistribution(base_dist, [distributions.TanhTransform(cache_size=1)])
            eps = 1e-6
            action_batch = action_batch / self.action_range[1]  # scale to [-1, 1]
            action_batch = action_batch.clamp(-1 + eps, 1 - eps)

            log_probs = dist.log_prob(action_batch).sum(dim=-1)
            entropy = base_dist.entropy().sum(dim=-1)

        actor_loss = (-log_probs * advantage - self.entropy_coef * entropy).mean()

        return actor_loss, critic_loss
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return 0.0, 0.0
        non_final_mask, next_states, state_batch, action_batch, reward_batch = sample

        # Normalize rewards (optional but often helpful)
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-7)

        # Compute critic and actor loss
        actor_loss, critic_loss = self.calculate_loss(non_final_mask, next_states, state_batch, action_batch, reward_batch)
        
        # Backpropagate and update critic network parameters
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()


        return actor_loss.item(), critic_loss.item()
        # Backpropagate and update actor network parameters
        # ====================================== #

    def learn(self, env):
        """
        Train the agent for 1 episode. (can using with multi environments)

        Args:
            env: The environment to train in.

        Returns:
            Tuple[List[float], List[int], List[float]]:
                - List[float]: Episode return for each environment.
                - List[int]: Alive time steps for each environment.
                - List[float]: Average TD error for each environment.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
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
        actor_losses = []
        critic_losses = []
        # ====================================== #

        while not all(dones):
            # Predict action from the policy network
            # ========= put your code here ========= #
            actions_idx = []
            actions = []

            for i, state in enumerate(state_list):
                if dones[i]:
                    actions_idx.append(0)
                    actions.append(torch.tensor([[0.0]], dtype=torch.float32))
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    a_idx, a_cont = self.select_action(state_tensor)
                    actions_idx.append(a_idx)
                    actions.append(a_cont)
            actions = torch.cat(actions, dim=0)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            next_obs_list, rewards, terminations, truncations, _ = env.step(actions)
            next_state_list = self.extract_policy_state(next_obs_list)
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            for i in range(num_envs):
                if not dones[i]:
                    done = bool(terminations[i].item()) or bool(truncations[i].item())
                    self.memory.add(
                        torch.tensor(state_list[i], dtype=torch.float32),
                        actions_idx[i],
                        rewards[i].item(),
                        torch.tensor(next_state_list[i], dtype=torch.float32),
                        done
                    )
                    cumulative_rewards[i] += rewards[i].item()
                    steps[i] += 1
                    dones[i] = done
                    state_list[i] = next_state_list[i]
            
            # ====================================== #

            # Update state

            # Perform one step of the optimization (on the policy network)
            actor_loss, critic_loss = self.update_policy()
            # Soft update of the target network's weights
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        return cumulative_rewards, steps, np.mean(actor_losses), np.mean(critic_loss)
    
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