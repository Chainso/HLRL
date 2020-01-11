import torch

from hlrl.core.agents import RLAgent

class TorchRLAgent(RLAgent):
    """
    An agent that collects (state, action, reward, next state) tuple
    observations
    """
    def __init__(self, env, algo, render=False, logger=None, device="cpu"):
        """
        Creates an agent that interacts with the given environment using the
        algorithm given.

        Args:
            env (Env): The environment the agent will explore in.

            algo (TorchRLAlgo): The algorithm the agent will use the explore the
                                environment.
            render (bool): If the environment is to be rendered (if applicable)
            logger (Logger, optional): The logger to log results while
                                       interacting with the environment.
            device (str): The device for the agent to run on.
        """
        super().__init__(env, algo, render, logger)
        self.device = torch.device(device)

    def make_tensor(self, data):
        """
        Creates a float tensor of the data of batch size 1.
        """
        return torch.FloatTensor([data]).to(self.device)

    def transform_state(self, state):
        return self.make_tensor(state)

    def transform_reward(self, reward):
        return self.make_tensor([reward])

    def transform_terminal(self, terminal):
        return self.make_tensor([terminal])

    def transform_action(self, action):
        return action[0].cpu().numpy()