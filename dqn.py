"""Deep Q-Network components for Lunar Lander (PyTorch + Gymnasium).

This module contains three small building blocks:

* :class:`QNetwork` - the neural network that estimates action values Q(s, a).
* :class:`ReplayBuffer` - a circular buffer of past transitions used for
  experience replay, which breaks the correlation between consecutive samples.
* :class:`DQN` - the agent that selects actions, stores transitions, and
  optimizes the network with the DQN loss from Mnih et al. (2013).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    """Feed-forward network mapping an observation to one Q value per action."""

    def __init__(self, n_observations: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ReplayBuffer:
    """Fixed-size circular buffer of (state, action, reward, next_state, done)."""

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        device: torch.device,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.pos
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        self.pos = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from a buffer of size {self.size}"
            )

        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]).to(self.device),
            torch.from_numpy(self.actions[indices]).to(self.device),
            torch.from_numpy(self.rewards[indices]).to(self.device),
            torch.from_numpy(self.next_states[indices]).to(self.device),
            torch.from_numpy(self.dones[indices]).to(self.device),
        )


@dataclass
class DQNConfig:
    """Hyperparameters for the DQN training procedure."""

    gamma: float = 0.99
    learning_rate: float = 1e-4
    buffer_size: int = 50_000
    min_buffer_size: int = 1_000
    batch_size: int = 64
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995  # multiplicative decay per episode
    hidden_size: int = 128


class DQN:
    """An agent that learns an action-value function with Deep Q-Learning."""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        config: DQNConfig | None = None,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        if self.config.min_buffer_size > self.config.buffer_size:
            raise ValueError("min_buffer_size cannot exceed buffer_size")
        if self.config.batch_size > self.config.min_buffer_size:
            raise ValueError("batch_size cannot exceed min_buffer_size")

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_observations = n_observations
        self.n_actions = n_actions

        if seed is not None:
            torch.manual_seed(seed)

        self.policy_net = QNetwork(
            n_observations, n_actions, self.config.hidden_size
        ).to(self.device)
        self.target_net = QNetwork(
            n_observations, n_actions, self.config.hidden_size
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target values are never differentiated

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.config.learning_rate
        )
        self.replay_buffer = ReplayBuffer(
            self.config.buffer_size,
            (n_observations,),
            self.device,
            seed=seed,
        )
        self.action_rng = np.random.default_rng(seed)

    def epsilon_for_episode(self, episode: int) -> float:
        """Return epsilon for a zero-based episode index."""
        return max(
            self.config.epsilon_end,
            self.config.epsilon_start * self.config.epsilon_decay**episode,
        )

    @torch.no_grad()
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection; epsilon=0 gives the greedy policy."""
        if self.action_rng.random() < epsilon:
            return int(self.action_rng.integers(self.n_actions))

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return int(self.policy_net(state_tensor).argmax(dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """Perform one gradient step; return the loss or None if still warming up."""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        # Q(s, a) for the actions actually taken.
        q_values = self.policy_net(states).gather(1, actions)

        # y = r + gamma * max_a' Q_target(s', a'), except at terminal states.
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1).values.unsqueeze(1)
            expected_q_values = rewards + self.config.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Softly chase the policy network so bootstrapped targets stay stable.
        with torch.no_grad():
            for target_param, param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.lerp_(param, self.config.tau)

        return float(loss.item())

    def save(
        self,
        path: str | Path,
        *,
        episode: int | None = None,
        mean_reward: float | None = None,
        env_id: str | None = None,
    ) -> Path:
        """Save the learned weights and training metadata to a PyTorch file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.policy_net.state_dict(),
            "n_observations": self.n_observations,
            "n_actions": self.n_actions,
            "config": asdict(self.config),
            "episode": episode,
            "mean_reward": mean_reward,
            "env_id": env_id,
        }
        torch.save(checkpoint, path)
        return path

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device | str | None = None
    ) -> tuple["DQN", dict[str, Any]]:
        """Load an agent and its checkpoint metadata from disk."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no saved model at {path}; train one first")

        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # PyTorch versions without the weights_only argument
            checkpoint = torch.load(path, map_location=device)

        agent = cls(
            n_observations=checkpoint["n_observations"],
            n_actions=checkpoint["n_actions"],
            config=DQNConfig(**checkpoint["config"]),
            device=device,
        )
        agent.policy_net.load_state_dict(checkpoint["model_state_dict"])
        agent.target_net.load_state_dict(checkpoint["model_state_dict"])
        return agent, checkpoint
