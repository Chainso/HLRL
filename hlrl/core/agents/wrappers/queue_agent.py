import queue

from multiprocessing import Barrier, Queue
from typing import Any, Callable, Dict, Iterable, Optional


from hlrl.core.common.wrappers import MethodWrapper

class QueueAgent(MethodWrapper):
    """
    An agent that sends its experiences into a queue 1 at a time rather than
    training directly.
    """
    def train_step(self,
                   ready_experiences: Dict[str, Iterable[Any]],
                   batch_size: int,
                   experience_queue: Queue) -> bool:
        """
        Trains on the ready experiences.

        Args:
            ready_experiences: The buffer of experiences that can be trained on.
            batch_size: The batch size for training.
            experience_queue: The queue to send experiences to.

        Returns:
            True, if ready experiences were used, False if the batch was too
            small.
        """
        added = False

        # Get length of a random key
        keys = list(ready_experiences)
        if len(keys) > 0:
            key = keys[0]
            
            # Going to use >= here because in the case the queue is full, the
            # number of ready experiences may be greater than the batch size
            if len(ready_experiences[key]) >= batch_size:
                batch = self.create_batch(ready_experiences)

                try:
                    for experience in batch:
                        experience_queue.put_nowait(experience)
                        added = True
                except queue.Full:
                    pass

        return added

    def train(self,
              num_episodes: int,
              batch_size: int,
              decay: float,
              n_steps: int,
              experience_queue: Queue,
              queue_barrier: Barrier,
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
            queue_barrier: A barrier to use when all queue tasks are complete on
                all processes.
            exit_condition: An alternative exit condition to num episodes which
                will be used if given.
        """
        self.om.train(
            num_episodes, batch_size, decay, n_steps, experience_queue,
            exit_condition=exit_condition
        )
        
        # Wait for all processes to finish using queues
        queue_barrier.wait()

        while not experience_queue.empty():
            pass
