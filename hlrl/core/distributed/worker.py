import queue

from multiprocessing import Queue, Event

from hlrl.core.experience_replay import ExperienceReplay

class Worker():
    """
    A worker for an off-policy algorithm, used to separate training and the
    agent.

    Based on Ape-X:
    https://arxiv.org/pdf/1803.00933.pdf
    """
    def train(self, experience_replay: ExperienceReplay, done_event: Event,
        agent_queue: Queue, sample_queue: Queue, priority_queue: Queue,
        batch_size: int, start_size: int):
        """
        Trains the algorithm until all agent processes have ended.

        Args:
            experience_replay (ExperienceReplay): The replay buffer to add
                experiences into.
            done_event (multiprocessing.Event): The event to set to awaken
                the agents to exit.
            agent_queue (multiprocessing.Queue): The queue of experiences to
                receive from agents.
            sample_queue (multiprocessing.Queue): The queue to send batches
                to the learner.
            priority_queue (multiprocessing.Queue): The queue to receive updated
                values for priority calculation from the learner.
            batch_size (int): The size of the training batch.
            start_size (int): The number of samples in the buffer to start
                training.
        """
        while not done_event.is_set():
            # Add all new experiences to the queue
            try:
                for _ in range(agent_queue.qsize()):
                    experience = agent_queue.get_nowait()
                    experience_replay.add(experience)
            except queue.Empty:
                pass

            # Receive new Q-values and targets to update
            try:
                for _ in range(sample_queue.qsize()):
                    idxs, new_qs, new_q_targs = priority_queue.get_nowait()
                    experience_replay.update_priorities(
                        idxs, new_qs, new_q_targs
                    )
            except queue.Empty:
                pass

            # Send a sample to the learner to train on
            if (len(experience_replay) >= batch_size
                and len(experience_replay) >= start_size):
                try:
                    sample = experience_replay.sample(batch_size)
                    sample_queue.put_nowait(sample)
                except queue.Full:
                    pass