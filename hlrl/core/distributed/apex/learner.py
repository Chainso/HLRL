import queue
from typing import Any, Callable, Optional, Tuple

from multiprocessing import Barrier, Event, Pipe, Queue
from time import time

from hlrl.core.algos import RLAlgo
from hlrl.core.experience_replay import PER

class ApexLearner():
    """
    A learner for a distributed algorithm using experience replay.

    Based on Ape-X:
    https://arxiv.org/pdf/1803.00933.pdf
    """
    def __init__(
            self,
            experience_replay: PER,
            prestart_func: Optional[Callable[[RLAlgo], Any]] = None
        ):
        """
        Creates the learner for a distributed RL algorithm.

        Args:
            experience_replay: The PER object responsible for computing the
                errors and priorites of training batch.
            prestart_func: A function to run when training start.
        """
        self.experience_replay = experience_replay
        self.prestart_func = prestart_func

    def train(
            self,
            algo: RLAlgo,
            done_event: Event,
            queue_barrier: Barrier,
            training_steps: int,
            sample_queue: Queue,
            priority_queue: Queue,
            param_pipes: Tuple[Pipe, ...] = tuple(),
            param_send_interval: int = 0,
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
            param_pipes: A tuple of pipes to send the model state to
                periodically.
            param_send_interval: The number of training steps in between
                parameter sends, 0 for never.
            save_path: The directory to save the model to.
            save_interval: The number of training steps in-between model saves.
        """
        if self.prestart_func is not None:
            self.prestart_func(algo)

        if param_send_interval > 0:
            for pipe in param_pipes:
                pipe.send(algo.save_dict())

        training_step = 0
        train_start = 0

        while training_step < training_steps and not done_event.is_set():
            if algo.logger is not None:
                sample_start = time()

            sample = sample_queue.get()
            rollouts, ids, is_weights = sample

            if algo.logger is not None:
                if train_start == 0:
                    train_start = time()

                train_step_start = time()

                algo.logger["Train/Samples per Second"] = (
                    1 / (train_step_start - sample_start), algo.training_steps
                )

            new_qs, new_q_targs = algo.train_batch(rollouts, is_weights)

            errors = self.experience_replay.get_error(new_qs, new_q_targs)
            priorities = self.experience_replay.get_priority(errors)

            priority_queue.put((ids, priorities))

            if algo.logger is not None:
                train_end = time()

                algo.logger["Train/Training Steps per Second"] = (
                    1 / (train_end - train_step_start), algo.training_steps
                )

                algo.logger["Train/Training Steps + Samples per Second"] = (
                    training_step / (train_end - train_start),
                    algo.training_steps
                )

            training_step += 1

            if(save_path is not None
                and algo.training_steps % save_interval == 0):
                algo.save(save_path)

            if (param_send_interval > 0
                and training_step % param_send_interval == 0):
                
                for pipe in param_pipes:
                    pipe.send(algo.save_dict())

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
