import torch
import numpy as np

from typing import Any, Dict, List, OrderedDict

from hlrl.core.agents import RLAgent
from hlrl.core.common.wrappers import MethodWrapper

class TorchRLAgent(MethodWrapper):
    """
    A torch agent that wraps its experiences as torch tensors.
    """
    def __init__(self,
                 agent: RLAgent,
                 batch_state: bool = True):
        """
        Creates torch agent that can wrap experiences as tensors.

        Args:
            agent: The agent to wrap.
            batch_state: If the state should be batched with a batch size of 1
                when transformed.
        """
        super().__init__(agent)
        
        self.batch_state = batch_state

    def make_tensor(self, data, dtype=torch.float32):
        """
        Creates a float tensor of the data of batch size 1.
        """
        if self.batch_state:
            data = [data]

        return torch.tensor(
            np.array(data), dtype=dtype, device=self.algo.device
        )

    def transform_state(self, state):
        state_dict = self.om.transform_state(state)
        state_dict["state"] = self.make_tensor(state_dict["state"])

        return state_dict

    def transform_reward(
            self,
            state: Any,
            algo_step: OrderedDict[str, Any],
            reward: Any,
            terminal: Any,
            next_state: Any
        ) -> Any:
        """
        Creates a tensor from the reward.

        Args:
            state: The state of the environment.
            algo_step: The transformed algorithm step of the state.
            reward: The reward from the environment.
            terminal: If the next state is a terminal state.
            next_state: The new state of the environment.

        Returns:
            The reward as a tensor.
        """
        reward = self.om.transform_reward(
            state, algo_step, reward, terminal, next_state
        )

        if self.batch_state:
            reward = [reward]

        return self.make_tensor(reward)

    def transform_terminal(self, terminal: Any, info: Any) -> Any:
        """
        Transforms the terminal of an environment step.

        Args:
            terminal: The terminal value to transform.
            info: Additional environment information for the step.

        Returns:
            The transformed terminal.
        """
        terminal = self.om.transform_terminal(terminal, info)

        if self.batch_state:
            terminal = [terminal]

        return self.make_tensor(terminal, torch.long)

    def transform_action(self, action):
        return self.om.transform_action(action).squeeze().cpu().numpy()

    def reward_to_float(self, reward: torch.Tensor) -> float:
        """
        Converts the reward to a single float value.

        Args:
            reward: The reward to turn into a float.

        Returns:
            The float value of the reward tensor.
        """
        reward = reward[0].detach().cpu()
        reward = reward.item()

        return reward

    def create_batch(
            self,
            ready_experiences: Dict[str, List[Any]],
        ) -> Dict[str, torch.Tensor]:
        """
        Creates a batch of experiences to be trained on from the ready
        experiences.

        Args:
            ready_experiences: The experiences to be trained on.
        
        Returns:
            A dictionary of each field necessary for training.
        """
        batch = {
            key: torch.cat(ready_experiences[key]) for key in ready_experiences
        }

        return self.om.create_batch(batch)
