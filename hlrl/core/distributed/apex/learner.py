
import queue
from typing import Optional
from time import time

from torch import cuda
from torch.multiprocessing import Barrier, Event, Pipe, Queue

from hlrl.core.algos import RLAlgo
from hlrl.core.experience_replay import ExperienceReplay


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
            experience_replay: ExperienceReplay,
            experience_queue: Queue,
            param_pipes: Tuple[Pipe],
            param_send_interval: int,
            save_path: Optional[str] = None,
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
            experience_queue: The queue to experiences from.
            param_pipes: A tuple of pipes to send parameters to periodically.
            param_send_interval: The number of training steps in between each
                send of the model paramters.
            save_path: The directory to save the model to.
            save_interval: The number of training steps in-between model saves.
        """
        training_step = 0
        
        # Training vars
        train_ret = None

        if algo.logger is not None:
            train_start = 0

        query_func = lambda: str(algo.device) != "cuda" or cuda.query()

        while training_step < training_steps and not done_event.is_set():
            # Update samples once all computation is complete
            # (since gpu is asynchronous)
            if (query_func() and train_ret is not None):

                experience_replay.calculate_and_update_priorities(*sample)

                train_ret = None
                training_step += 1

                if algo.logger is not None:
                    train_time = time() - train_start
                    train_speed = train_time / training_step

                    algo.logger["Train/Training Steps per Second"] = (
                        train_speed, training_step
                    )

            # Start training a sample
            if (training_step < training_steps
                and len(experience_replay) >= batch_size
                and len(experience_replay) >= start_size
                and query_func() and train_ret is None):
                
                if algo.logger is not None:
                    train_start = time()

                sample = experience_replay.sample(batch_size)
                rollouts, idxs, is_weights = sample

                train_ret = algo.train_batch(rollouts, is_weights)

            # Add all new experiences to the queue
            try:
                for _ in range(experience_queue.qsize()):
                    experience = experience_queue.get_nowait()
                    experience_replay.add(experience)
            except queue.Empty:
                pass

            if(save_path is not None
                and algo.training_steps % save_interval == 0):

                algo.save(save_path)

            if algo.training_steps % param_send_interval == 0:
                for pipe in param_pipes:
                    pipe.send(algo.save_dict())

        # Signal exit
        done_event.set()

        # Wait for all processes to finish using queues
        queue_barrier.wait()

        # Clear queues
        try:
            while not experience_queue.empty():
                sample_queue.get_nowait()
        except queue.Empty:
            pass
