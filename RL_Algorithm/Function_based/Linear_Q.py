from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm
import torch


class Linear_QN(BaseAlgorithm):
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
        x = obs                           # Current state vector
        x_next = next_obs                # Next state vector

        q_current = x @ self.w[:, action]           # Q(s, a)
        q_next = np.max(x_next @ self.w)            # max_a' Q(s', a')
        td_target = reward + (0 if terminated else self.discount_factor * q_next)
        td_error = td_target - q_current

        td_error = float(td_error.detach().cpu()) if isinstance(td_error, torch.Tensor) else float(td_error)

        # Gradient descent update
        self.w[:, action] += self.lr * td_error * x

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
        policy_tensor = obs['policy']
        state = np.array([0,0,0,0], dtype=np.float32)
        state[0] = policy_tensor[0, 0].item()  # First value
        state[1] = policy_tensor[0, 1].item()  # Second value
        state[2] = policy_tensor[0, 2].item()   # Third value
        state[3] = policy_tensor[0, 3].item()   # Fourth value
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action_index, action_continuous = self.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action_continuous)
            done = terminated | truncated

            next_policy_tensor = next_obs['policy']
            next_state = np.array([0,0,0,0], dtype=np.float32)
            next_state[0] = next_policy_tensor[0, 0].item()  # First value
            next_state[1] = next_policy_tensor[0, 1].item()  # Second value
            next_state[2] = next_policy_tensor[0, 2].item()   # Third value
            next_state[3] = next_policy_tensor[0, 3].item()   # Fourth value

            self.update(
                obs=state,
                action=action_index,
                reward=reward,
                next_obs=next_state,
                terminated=done
            )

            state = next_state
            total_reward += reward
            steps += 1
        self.decay_epsilon()

        return total_reward, steps
        # ====================================== #
    




    