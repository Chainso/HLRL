import torch

from typing import Any, Dict, List, OrderedDict, Tuple

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

    def __reduce__(self) -> Tuple[type, Tuple[Any, ...]]:
        """
        Reduces the inputs used to serialize and recreate the torch agent.

        Returns:
            A tuple of the class and input arguments.
        """
        return (type(self), (self.obj, self.batch_state))

    def make_tensor(self, data):
        """
        Creates a float tensor of the data of batch size 1.
        """
        return torch.FloatTensor(data).to(self.algo.device)

    def transform_state(self, state):
        state_dict = self.om.transform_state(state)

        if self.batch_state:
            state_dict["state"] = [state_dict["state"]]

        state_dict["state"] = self.make_tensor(state_dict["state"])

        return state_dict

    def transform_reward(self,
                         state: Any,
                         algo_step: OrderedDict[str, Any],
                         reward: Any,
                         terminal: Any,
                         next_state: Any) -> Any:
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
        return self.make_tensor([[self.om.transform_reward(
            state, algo_step, reward, terminal, next_state
        )]])

    def transform_terminal(self, terminal):
        return self.make_tensor([[self.om.transform_terminal(terminal)]])

    def transform_action(self, action):
        return self.om.transform_action(action).squeeze().cpu().numpy()

    def reward_to_float(self,
                        reward: torch.Tensor) -> float:
        """
        Converts the reward to a single float value.

        Args:
            reward: The reward to turn into a float.

        Returns:
            The float value of the reward tensor.
        """
        return reward.detach().item()

    def create_batch(
            self,
            ready_experiences: Dict[str, List[Any]],
        ) -> Dict[str, torch.FloatTensor]:
        """
        Creates a batch of experiences to be trained on from the ready
        experiences.

        Args:
            ready_experiences: The experiences to be trained on.
        
        Returns:
            A dictionary of each field necessary for training.
        """
        for key in ready_experiences:
            ready_experiences[key] = torch.cat(ready_experiences[key])

        ready_experiences = self.om.create_batch(ready_experiences)

        return ready_experiences
