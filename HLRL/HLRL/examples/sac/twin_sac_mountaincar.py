import torch.nn as nn

class QFunc(nn.Module):
    def __init__(self):
        super(self).__init__()

    def forward(self):
        pass

class Policy(nn.Module):
    def __init__(self):
        super(self).__init__()

    def forward(self):
        pass

class Value(nn.Module):
    def __init__(self, state_space, hidden_size, num_hidden):
        assert state_space > 0
        assert hidden_size > 0
        assert num_hidden > 0

        super(self).__init__()

        self.linear = nn.Sequential(
            self._lin_block(state_space, hidden_size),
            *[self._lin_block(hidden_size, hidden_size)
              for _ in range(num_hidden)],
            nn.Linear(hidden_size, 1)
        )

    def _lin_block(self, num_in, num_out):
        return nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU()
        )

    def forward(self, inp):
        return self.linear(inp)

if(__name__ == "__main__"):
    from argparse import ArgumentParser

    from HLRL.core.logger import Logger
    from HLRL.torch.algos.sac.sac import SAC

    parser = ArgumentParser(description = "Twin Q-Function SAC example on "
                            + "the MountainCarContinuous-v0 environment.")
    # The logger
    logger = Logger