
import queue

from multiprocessing import Barrier, Queue, Event
from time import time

from hlrl.core.algos import RLAlgo

class ApexLearner():
    """
    A learner for a distributed algorithm using experience replay.

    Based on Ape-X:
    https://arxiv.org/pdf/1803.00933.pdf
    """
    def train(
            self,
            algo: RLAlgo,
            done_event: Event,
            queue_barrier: Barrier,
            training_steps: int,
            sample_queue: Queue,
            priority_queue: Queue,
            save_path: str = None,
            save_interval: int = 10000
        ) -> None:
        """
        Trains the algorithm until all agent processes have ended.

        Args:
            algo: The algorithm to train.
            done_event: The event to set to allow the other processes to exit.
            queue_barrier: A barrier to use when all queue tasks are complete on
                all processes.
            training_steps: The number of steps to train for.
            sample_queue: The queue to receive buffer samples from.
            priority_queue: The queue to send updated values to.
            save_path: The directory to save the model to.
            save_interval: The number of training steps in-between model saves.
        """
        training_step = 0
        train_start = 0

        while training_step < training_steps and not done_event.is_set():
            if algo.logger is not None:
                sample_start = time()

            sample = sample_queue.get()
            rollouts, idxs, is_weights = sample

            if algo.logger is not None:
                if train_start == 0:
                    train_start = time()

                train_step_start = time()

                algo.logger["Train/Samples per Second"] = (
                    1 / (train_step_start - sample_start), algo.training_steps
                )

            new_qs, new_q_targs = algo.train_batch(rollouts, is_weights)

            priority_queue.put((idxs, new_qs, new_q_targs))

            if algo.logger is not None:
                train_end = time()

                algo.logger["Train/Training Steps per Second"] = (
                    1 / (train_end - train_step_start), algo.training_steps
                )

                algo.logger["Train/Training Steps + Samples per Second"] = (
                    training_step / (train_end - train_start),
                    algo.training_steps
                )

            if(save_path is not None
                and algo.training_steps % save_interval == 0):
                algo.save(save_path)

            training_step += 1

        # Signal exit
        done_event.set()

        # Wait for all processes to finish using queues
        queue_barrier.wait()

        # Clear queues
        try:
            while not sample_queue.empty():
                sample_queue.get_nowait()
        except queue.Empty:
            pass

        while not priority_queue.empty():
            pass
