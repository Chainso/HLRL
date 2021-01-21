
import queue
from typing import Optional, Tuple
from time import time

from multiprocessing import Barrier, Event, Pipe, Queue

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
            batch_size: int,
            start_size: int,
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
            batch_size: The size of the training batch.
            start_size: The number of samples in the buffer to start training.
            experience_replay: The replay buffer to add experiences into.
            experience_queue: The queue to experiences from.
            param_pipes: A tuple of pipes to send parameters to periodically.
            param_send_interval: The number of training steps in between each
                send of the model paramters.
            save_path: The directory to save the model to.
            save_interval: The number of training steps in-between model saves.
        """
        training_step = 0

        if algo.logger is not None:
            train_start = 0

        while training_step < training_steps and not done_event.is_set():
            # Start training a sample
            if (len(experience_replay) >= batch_size
                and len(experience_replay) >= start_size):
                
                if algo.logger is not None and train_start == 0:
                    train_start = time()

                sample = experience_replay.sample(batch_size)
                rollouts, sample_ids, is_weights = sample

                train_ret = algo.train_batch(rollouts, is_weights)

                experience_replay.update_priorities(sample_ids, *train_ret)

                training_step += 1

                if algo.logger is not None:
                    train_time = time() - train_start
                    train_speed = training_step / train_time

                    algo.logger["Train/Training Steps per Second"] = (
                        train_speed, training_step
                    )

                if(save_path is not None
                    and algo.training_steps % save_interval == 0):

                    algo.save(save_path)

                if algo.training_steps % param_send_interval == 0:
                    for pipe in param_pipes:
                        pipe.send(algo.save_dict())

            # Add all new experiences to the queue
            try:
                for _ in range(experience_queue.qsize()):
                    experience = experience_queue.get_nowait()
                    experience_replay.add(experience)
            except queue.Empty:
                pass

        # Signal exit
        done_event.set()

        # Wait for all processes to finish using queues
        queue_barrier.wait()

        # Clear queues
        try:
            while not experience_queue.empty():
                experience_queue.get_nowait()
        except queue.Empty:
            pass
