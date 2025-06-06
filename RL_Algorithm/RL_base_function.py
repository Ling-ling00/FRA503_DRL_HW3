import numpy as np
from collections import defaultdict, namedtuple, deque
import random
from enum import Enum
import torch
import torch.nn as nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size = 1):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of experiences the buffer can hold.
            batch_size (int): Number of experiences to sample per batch.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action_idx, reward, next_state, done):
        """
        Adds an experience to the replay buffer.

        Args:
            state (Tensor): The current state of the environment.
            action_idx (int): The action index of action taken at this state.
            reward (float): The reward received after taking the action.
            next_state (Tensor): The next state resulting from the action.
            done (bool): Whether the episode has terminated.
        """
        self.memory.append((state, action_idx, reward, next_state, done))

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            tuple:
                - state_batch (Tensor): Batch of states.
                - action_batch (Tensor): Batch of actions.
                - reward_batch (Tensor): Batch of rewards.
                - next_state_batch (Tensor): Batch of next states.
                - done_batch (Tensor): Batch of terminal state flags.
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*experiences)

        return (
            torch.stack(state_batch).to(device),
            torch.tensor(action_batch, dtype=torch.long).to(device),
            torch.tensor(reward_batch, dtype=torch.float).to(device),
            torch.stack(next_state_batch).to(device),
            torch.tensor(done_batch, dtype=torch.bool).to(device),
        )

    def __len__(self):
        """
        Returns the current size of the replay buffer.

        Returns:
            int: The number of stored experiences.
        """
        return len(self.memory)


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        learning_rate (float): Learning rate for updates.
        initial_epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays. (exp decay)
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        buffer_size (int): Maximum number of experiences the buffer can hold.
        batch_size (int): Number of experiences to sample per batch.
    """

    def __init__(
        self,
        num_of_action: int = 2,
        action_range: list = [-2.0, 2.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
        buffer_size: int = 1000,
        batch_size: int = 1,
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range  # [action_min, action_max]

        self.memory = ReplayBuffer(buffer_size, batch_size)        
    
    def scale_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n].
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        # ========= put your code here ========= #
        min_action, max_action = self.action_range
        action_step = (max_action - min_action) / (self.num_of_action - 1)
        action_value = min_action + action * action_step

        return torch.tensor([[action_value]], dtype=torch.float32)
        # ====================================== #
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here ========= #
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon
        # ====================================== #

    def extract_policy_state(self, obs):
        """
        Extract policy state from dict to numpy array and normalize.

        Args:
            obs (dict): State observation.
        
        Returns:
            np.ndarray: Normalized policy state.
        """
        policy = obs['policy']
        state = np.array(policy[:, :4].tolist(), dtype=np.float32)
        
        # Define bounds as arrays
        bound = np.array([ 3,  np.deg2rad(24),  5,  5], dtype=np.float32)
        
        # Clip to bounds
        state = np.clip(state, -1*bound, bound)
        
        return state / bound


