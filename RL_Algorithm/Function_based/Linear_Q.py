from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm
import torch


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
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """        

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
        obs,
        action: int,
        reward: float,
        next_obs,
        terminated: bool
    ):
        """
        Updates the weight vector using the Temporal Difference (TD) error 
        in Q-learning with linear function approximation.

        Args:
            obs (dict): The current state observation, containing feature representations.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_obs (dict): The next state observation.
            next_action (int): The action taken in the next state (used in SARSA).
            terminated (bool): Whether the episode has ended.

        """
        # ========= put your code here ========= #
        q_current = self.q(obs, action)
        q_next = np.max(self.q(next_obs))
        td_target = reward + (self.discount_factor * q_next)
        td_error = td_target - q_current

        # Gradient descent update
        self.w[:, action] += self.lr * td_error * obs

        self.training_error.append(td_error)
        # ====================================== #

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(self.num_of_action)
        else:
            q_values = self.q(state)
            action_index = int(np.argmax(q_values))

        return action_index, self.scale_action(action_index)
            # ====================================== #

    def learn(self, env, max_steps):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        obs, _ = env.reset()
        done = False
        cumulative_reward = 0
        step = 0

        while not done and step < max_steps:
            # agent stepping
            state = self.extract_policy_state(obs)
            action_idx, action = self.select_action(state)

            # env stepping
            next_obs, reward, terminated, truncated, _ = env.step(action)
            reward_value = reward.item()
            terminated_value = terminated.item() 
            cumulative_reward += reward_value
            done = terminated or truncated
            
            state = self.extract_policy_state(obs)
            next_state = self.extract_policy_state(next_obs)
            self.update(state, action_idx, reward_value, next_state, done)

            obs = next_obs
            step += 1
        self.decay_epsilon()

        return cumulative_reward, step
        # ====================================== #
    




    