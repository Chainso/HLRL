from hlrl.core.experience_replay.per import PER

class TorchPER(PER):
    """
    A Prioritized Experience Replay implementation using torch tensors
    https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha, beta, beta_increment, epsilon,
                 device = "cpu"):
        """
        Creates a new PER buffer with the given parameters

        capacity : The capacity of the replay buffer
        alpha : The alpha value for the prioritization, between 0 and 1
                inclusive
        beta : The beta value for the importance sampling, between 0 and 1
               inclusive
        beta_increment : The value to increment the beta by
        epsilon : The value of epsilon to add to the priority
        """