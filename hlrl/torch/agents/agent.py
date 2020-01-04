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
        self.device = device

    def _make_input_from_state(self, state):
        """
        Creates the algorithm input from the env state (does nothing by default)
        """
        return torch.FloatTensor([state]).to(self.device)