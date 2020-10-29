import queue

from multiprocessing import Queue
from typing import Any, Callable, Dict, Iterable, Optional


from hlrl.core.common.wrappers import MethodWrapper

class QueueAgent(MethodWrapper):
    """
    An agent that sends its experiences into a queue 1 at a time rather than
    training directly.
    """
    def train_step(self,
                   ready_experiences: Iterable[Dict[str, Any]],
                   experience_queue: Queue) -> None:
        """
        Trains on the ready experiences.

        Args:
            ready_experiences: The buffer of experiences that can be trained on.
            experience_queue: The queue to send experiences to.
        """
        try:
            for experience in ready_experiences:
                experience_queue.put_nowait(experience)
        except queue.Full:
            pass

    def train(self,
              num_episodes: int,
              batch_size: int,
              decay: float,
              n_steps: int,
              experience_queue: Queue,
              exit_condition: Optional[Callable[[], bool]] = None) -> None:
        """
        Trains the algorithm for the number of episodes specified on the
        environment.

        Args:
            num_episodes: The number of episodes to train for.
            batch_size: The number of ready experiences to train on at a time.
            decay: The decay of the next.
            n_steps: The number of steps.
            experience_queue: The queue to send experiences to.
            exit_condition: An alternative exit condition to num episodes which
                will be used if given.
        """
        self.om.train(
            num_episodes, batch_size, decay, n_steps, experience_queue,
            exit_condition=exit_condition
        )
        
        while not experience_queue.empty():
            pass