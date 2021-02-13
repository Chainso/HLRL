import queue

from multiprocessing import Barrier, Queue, Event

from hlrl.core.experience_replay import ExperienceReplay

class ApexWorker():
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
        return experiences, priorities

    def train(
            self,
            experience_replay: ExperienceReplay,
            done_event: Event,
            queue_barrier: Barrier,
            agent_queue: Queue,
            sample_queue: Queue,
            priority_queue: Queue,
            batch_size: int,
            start_size: int
        ) -> None:
        """
        Trains the algorithm until all agent processes have ended.

        Args:
            experience_replay: The replay buffer to add experiences into.
            done_event: The event to set to allow the agents to exit.
            queue_barrier: A barrier to use when all queue tasks are complete on
                all processes.
            agent_queue: The queue of experiences to receive from agents.
            sample_queue: The queue to send batches to the learner.
            priority_queue: The queue to receive updated values for priority
                calculation from the learner.
            batch_size: The size of the training batch.
            start_size: The number of samples in the buffer to start training.
        """
        while not done_event.is_set():
            # Add all new experiences to the queue
            try:
                experiences, priorities = agent_queue.get_nowait()
                experiences, priorities = self.on_receive_experiences(
                    experiences, priorities
                )
                
                for experience, priority in zip(experiences, priorities):
                    experience_replay.add(experience, priority)
            except queue.Empty:
                pass

            # Receive new Q-values and targets to update
            try:
                for _ in range(priority_queue.qsize()):
                    ids, priorities = priority_queue.get_nowait()
                    experience_replay.update_priorities(ids, priorities)
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

        # Wait for all processes to finish using queues
        queue_barrier.wait()

        # Clear queues
        try:
            while not agent_queue.empty():
                agent_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while not priority_queue.empty():
                priority_queue.get_nowait()
        except queue.Empty:
            pass
