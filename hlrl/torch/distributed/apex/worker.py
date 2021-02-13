from torch import Tensor

from hlrl.core.distributed import ApexWorker

class TorchApexWorker(ApexWorker):
    """
    A worker for an off-policy algorithm, used to separate training and the
    agent.

    Based on Ape-X:
    https://arxiv.org/pdf/1803.00933.pdf
    """
    def on_receive_experiences(self, experiences, priorities):
        """
        A function to call whenever a batch of experiences is received.

        Args:
            experiences: The receieved experience.
            priorities: The priority of the experiences.

        Returns:
            The received experiences and priorities of the batch.
        """
        experiences, priorities = super().on_receive_experiences(
            experiences, priorities
        )

        for experience in experiences:
            for key in experience:
                if isinstance(experience[key], Tensor):
                    experience[key] = experience[key].clone()

        return experiences, priorities
