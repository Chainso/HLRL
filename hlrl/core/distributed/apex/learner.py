
from multiprocessing import Queue, Event
from time import time

from hlrl.core.algos import RLAlgo

class ApexLearner():
    """
    A learner for a distributed algorithm using experience replay.

    Based on Ape-X:
    https://arxiv.org/pdf/1803.00933.pdf
    """
    def train(self, algo: RLAlgo, done_event: Event, sample_queue: Queue,
        priority_queue: Queue, save_path: str = None,
        save_interval: int = 10000):
        """
        Trains the algorithm until all agent processes have ended.

        Args:
            algo (RLAlgo): The algorithm to train.
            done_event (multiprocessing.Event): The event to set to allow the
                process to exit.
            sample_queue (multiprocessing.Queue): The queue to receive
                buffer samples from.
            priority_queue (multiprocessing.Queue): The queue to send updated
                values to.
            save_path (str): The directory to save the model to.
            save_interval (int): The number of training steps in-between
                model saves.
        """
        while not done_event.is_set():
            if algo.logger is not None:
                sample_start = time()

            sample = sample_queue.get()
            rollouts, idxs, is_weights = sample

            if algo.logger is not None:
                train_start = time()

                algo.logger["Train/Samples per Second"] = (
                    1 / (train_start - sample_start), algo.training_steps
                )

            new_qs, new_q_targs = algo.train_batch(rollouts, is_weights)

            priority_queue.put((idxs, new_qs, new_q_targs))

            if algo.logger is not None:
                train_end = time()

                algo.logger["Train/Training Steps per Second"] = (
                    1 / (train_end - train_start), algo.training_steps
                )

                algo.logger["Train/Training Steps + Samples per Second"] = (
                    1 / (train_end - sample_start), algo.training_steps
                )

            if(save_path is not None
                and algo.training_steps % save_interval == 0):
                algo.save(save_path)

        done_event.wait()
