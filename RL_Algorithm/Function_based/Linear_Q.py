from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm
import torch
import os
import json

class Linear_Q(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            num_of_action (int): Number of discrete actions available.
            action_range (list): Scale for continuous action mapping.
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time. (exp decay)
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """        
        self.w = np.zeros((4, num_of_action))

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(
        self,
        state,
        action_idx: int,
        reward: float,
        next_state,
        terminated: bool
    ):
        """
        Updates the weight vector using the Temporal Difference (TD) error 
        in Q-learning with linear function approximation.

        Args:
            state (np.array): The current state observation, containing feature representations.
            action_idx (int): The action index of action taken in the current state.
            reward (float): The reward received for taking the action.
            next_state (np.array): The next state observation.
            terminated (bool): Whether the episode has ended.

        Returns:
            float: Temporal Difference (TD) error
        """
        # ========= put your code here ========= #
        q_current = self.q(state, action_idx)
        q_next = np.max(self.q(next_state))
        td_target = reward + (self.discount_factor * q_next)
        td_error = td_target - q_current

        # Gradient descent update
        self.w[:, action_idx] += self.lr * td_error * state

        return td_error
        # ====================================== #

    def q(self, state, a=None):
        """
        Linearly estimates the Q-value for a given state and (optionally) action.

        Args:
            state (np.array): The current state observation, containing feature representations.
            a (int, optional): Action index. If None, returns Q-values for all actions.

        Returns:
            float or np.array: Q(s, a) if action is specified; otherwise, Q(s, :) for all actions.
        """
        # ========= put your code here ========= #
        if a==None:
            # Get q values from all action in state
            return state @ self.w
        else:
            # Get q values given action & state
            return state @ self.w[:, a]
        # ====================================== #

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (np.array): The current state of the environment.
        
        Returns:
            tuple (int, Tensor):
                - int: Index of the selected action.
                - Tensor: The selected action.
        """
        # ========= put your code here ========= #
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(self.num_of_action)
        else:
            q_values = self.q(state)
            action_index = int(np.argmax(q_values))

        return action_index, self.scale_action(action_index)
            # ====================================== #

    def learn(self, env):
        """
        Train the agent for 1 episode. (can using with multi environments)

        Args:
            env: The environment in which the agent interacts.

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
        losses = [[] for _ in range(num_envs)]

        while not all(dones):
            # agent stepping
            actions_idx = []
            actions = []

            for i, state in enumerate(state_list):
                if dones[i]:
                    actions_idx.append(0)
                    actions.append(torch.tensor([[0.0]], dtype=torch.float32))
                else:
                    a_idx, a_cont = self.select_action(state)
                    actions_idx.append(a_idx)
                    actions.append(a_cont)
            actions = torch.cat(actions, dim=0)

            # env stepping
            next_obs_list, rewards, terminations, truncations, _ = env.step(actions)
            next_state_list = self.extract_policy_state(next_obs_list)
            
            for i in range(num_envs):
                if not dones[i]:
                    done = bool(terminations[i].item()) or bool(truncations[i].item())
                    loss = self.update(state_list[i], actions_idx[i], rewards[i].item(), next_state_list[i], done)
                    losses[i].append(loss)
                    cumulative_rewards[i] += rewards[i].item()
                    steps[i] += 1
                    dones[i] = done
                    state_list[i] = next_state_list[i]
            
        self.decay_epsilon()
        avg_losses = [np.mean(l) if l else 0.0 for l in losses]

        return cumulative_rewards, steps, avg_losses
        # ====================================== #
    
    def save_model(self, path, filename):
        """
        Save weight parameters.

        Args:
            path (str): Directory to save model.
            filename (str): Name of the file.
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as f:
            json.dump(self.w.tolist(), f)
        # ====================================== #
            
    def load_model(self, path, filename):
        """
        Load weight parameters.

        Args:
            path (str): Directory to save model.
            filename (str): Name of the file.
        """
        # ========= put your code here ========= #
        full_path = os.path.join(path, filename)
        with open(full_path, 'r') as f:
            self.w = np.array(json.load(f))
        # ====================================== #
    




    