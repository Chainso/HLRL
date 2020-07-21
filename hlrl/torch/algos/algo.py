import torch
import torch.nn as nn

from hlrl.core.algos import RLAlgo

class TorchRLAlgo(RLAlgo, nn.Module):
    """
    An abstract reinforcement learning algorithm.
    """
    def __init__(self, logger=None):
        """
        Creates the reinforcement learning algorithm.

        Args:
            logger (Logger, optional): The logger to log results while training
                                       and evaluating.
        """
        RLAlgo.__init__(self, logger)
        nn.Module.__init__(self)

    def save_dict(self):
        # Save all the dicts
        state = {
            "state_dict": self.state_dict(),
            "env_episodes": self.env_episodes,
            "training_steps": self.training_steps,
            "env_steps": self.env_steps
        }

        return state

    def save(self, save_path):
        model_name = "/model-" + str(self.training_steps) + ".pt"
        torch.save(self.save_dict(), save_path + model_name)

    def load(self, load_path):
        state = torch.load(load_path)

        self.load_state_dict(state["state_dict"])
        self.env_episodes = state["env_episodes"]
        self.training_steps = state["training_steps"]
        self.env_steps = state["env_steps"]

class TorchOffPolicyAlgo(TorchRLAlgo):
    """
    The base class of an off-policy torch algorithm.
    """
    def __init__(self, logger=None):
        """
        Creates the off-policy algorithm.

        Args:
            logger (Logger, optional): The logger to log results while training
                                       and evaluating.
        """
        super().__init__(logger)


    def train_from_buffer(self, experience_replay, batch_size, start_size,
                          save_path=None, save_interval=10000):
        """
        Starts training the network.

        Args:
            experience_replay (ExperienceReplay): The experience replay buffer
                                                  to sample experiences from.
            batch_size (int): The batch size of the experiences to train on.
            start_size (int): The amount of expreiences in the buffer before
                              training is started.
            save_path (Optional, str): The path to save the model to.
            save_interval (int): The number of batches between saves.
        """
        if(batch_size <= len(experience_replay)
           and start_size <= len(experience_replay)):
            sample = experience_replay.sample(batch_size)
            rollouts, idxs, is_weights = sample

            new_q, new_q_targ = self.train_batch(rollouts, is_weights)
            experience_replay.update_priorities(idxs, new_q, new_q_targ)

            if(save_path is not None
               and self.training_steps % save_interval == 0):
                self.save(save_path)
