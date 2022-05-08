from concurrent.futures import process
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import numpy as np

from hlrl.core.experience_replay import ExperienceReplay
from hlrl.core.logger import Logger
from hlrl.core.algos import RLAlgo

class TorchRLAlgo(RLAlgo, nn.Module):
    """
    An abstract reinforcement learning algorithm.
    """
    def __init__(self, device: str = "cpu", logger: Optional[Logger] = None):
        """
        Creates the reinforcement learning algorithm.

        Args:
            device: The device of the tensors in the module.
            logger: The logger to log results while training and evaluating.
        """
        RLAlgo.__init__(self, logger)
        nn.Module.__init__(self)

        self.device = torch.device(device)

    def process_batch(
            self,
            rollouts: Dict[str, Union[torch.Tensor, np.ndarray]]
        ) -> Dict[str, torch.Tensor]:
        """
        Processes a batch to make it suitable for training.

        Args:
            rollouts: The training batch to process.

        Returns:
            The processed training batch.
        """
        for key, value in rollouts.items():
            if isinstance(value, torch.Tensor):
                rollouts[key] = value.to(self.device)
            elif isinstance(value, np.ndarray):
                rollouts[key] = torch.tensor(value, device=self.device)

        return rollouts

    def save_dict(self) -> Dict[str, Any]:
        """
        Saves in the current state of the algorithm in a dictionary.

        Returns:
            A dictionary of values to save this algorithm.
        """
        state_dict = self.state_dict()

        cpu_state_dict = {
            key: state_dict[key].to("cpu") for key in state_dict
        }

        # Save all the dicts
        state = {
            "state_dict": cpu_state_dict,
            "env_episodes": self.env_episodes,
            "training_steps": self.training_steps,
            "env_steps": self.env_steps
        }

        return state

    def save(self, save_path):
        model_name = "/model-" + str(self.training_steps) + ".zip"
        torch.save(self.save_dict(), save_path + model_name)

    def load_dict(self, load_path):
        return torch.load(load_path)

    def load(self, load_path: str = "", load_dict=None):
        if load_dict is None:
            load_dict = self.load_dict(load_path)

        self.load_state_dict(load_dict["state_dict"])
        self.env_episodes = load_dict["env_episodes"]
        self.training_steps = load_dict["training_steps"]
        self.env_steps = load_dict["env_steps"]

class TorchOffPolicyAlgo(TorchRLAlgo):
    """
    The base class of an off-policy torch algorithm.
    """
    def __init__(self, device: str = "cpu", logger: Optional[Logger] = None):
        """
        Creates the off-policy algorithm.

        Args:
            device: The device of the tensors in the module.
            logger: The logger to log results while training and evaluating.
        """
        super().__init__(device, logger)

    def process_batch(
            self,
            rollouts: Dict[str, Union[torch.Tensor, np.ndarray]],
            is_weights: Union[int, torch.Tensor, np.ndarray] = 1
        ) -> Dict[str, torch.Tensor]:
        """
        Processes a batch to make it suitable for training.

        Args:
            rollouts: The training batch to process.
            is_weights: The importance sample weights of the batch to process.

        Returns:
            The processed training batch.
        """
        rollouts = super().process_batch(rollouts)

        if isinstance(is_weights, torch.Tensor):
            is_weights = is_weights.to(self.device)
        elif isinstance(is_weights, np.ndarray):
            is_weights = torch.tensor(is_weights, device=self.device)

        return rollouts, is_weights

    def train_from_buffer(
            self,
            experience_replay: ExperienceReplay,
            batch_size: int,
            start_size: int,
            save_path: Optional[str] = None,
            save_interval: int = 10000
        ) -> None:
        """
        Starts training the network.

        Args:
            experience_replay: The experience replay buffer  to sample
                experiences from.
            batch_size: The batch size of the experiences to train on.
            start_size: The amount of expreiences in the buffer before training
                is started.
            save_path: The path to save the model to.
            save_interval: The number of batches between saves.
        """
        if(batch_size <= len(experience_replay)
           and start_size <= len(experience_replay)):
            sample = experience_replay.sample(batch_size)
            rollouts, ids, is_weights = sample

            new_q, new_q_targ = self.train_batch(rollouts, is_weights)

            experience_replay.calculate_and_update_priorities(
                ids, new_q, new_q_targ
            )

            if(save_path is not None
               and self.training_steps % save_interval == 0):
                self.save(save_path)
