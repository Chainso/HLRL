import torch

from hlrl.core.agents import RLAgent
from hlrl.core.common.wrappers import MethodWrapper

class TorchRLAgent(MethodWrapper):
    """
    An agent that collects (state, action, reward, next state) tuple
    observations
    """
    def __init__(self,
        agent: RLAgent,
        batch_state: bool = True,
        device: str ="cpu"):
        """
        Creates an agent that interacts with the given environment using the
        algorithm given.

        Args:
            agent (RLAgent): The agent to wrap.
            batch_state (bool): If the state should be batched with a batch size
                of 1 when transformed.
            device (str): The device for the agent to run on.
        """
        super().__init__(agent)
        self.batch_state = batch_state
        self.device = torch.device(device)

    def make_tensor(self, data):
        """
        Creates a float tensor of the data of batch size 1.
        """
        return torch.FloatTensor(data).to(self.device)

    def transform_state(self, state):
        state_dict = self.om.transform_state(state)

        if self.batch_state:
            state_dict["state"] = [state_dict["state"]]

        state_dict["state"] = self.make_tensor(state_dict["state"])

        return state_dict

    def transform_reward(self, state, algo_step, reward, next_state):
        return self.make_tensor([[
            self.om.transform_reward(state, algo_step, reward, next_state)
        ]])

    def transform_terminal(self, terminal):
        return self.make_tensor([[self.om.transform_terminal(terminal)]])

    def transform_action(self, action):
        return self.om.transform_action(action).squeeze().cpu().numpy()

    def reward_to_float(self, reward):
        return reward.detach().item()