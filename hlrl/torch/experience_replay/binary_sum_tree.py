import torch

from hlrl.core.experience_replay.binary_sum_tree import BinarySumTree

class TorchBinarySumTree(BinarySumTree):
    """
    A binary sum tree using torch instead of numpy
    """
    def __init__(self, num_leaves, device):
        """
        Creates a binary sum tree with the given number of leaves

        num_leaves : The number of leaves in the tree
        """
        super().__init__(num_leaves)
        self.device = torch.device(device)
        self.tree = torch.from_numpy(self.tree).to(device)