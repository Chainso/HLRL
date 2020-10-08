from multiprocessing import Queue, Event

class Worker():
    """
    A worker for an off-policy algorithm, used to separate training and the
    agent.

    Based on Ape-X:
    https://arxiv.org/pdf/1803.00933.pdf
    """
    def __init__(self, algo, experience_replay, experience_queue):
        """
        Creates worker for the algorithm, and with an experience queue to be fed
        experiences from.

        Args:
            algo (torch.nn.Module): The algorithm to train.
            experience_replay (ExperienceReplay): The replay buffer to add
                experiences into.
            experience_queue (torch.multiprocessing.Queue): The queue to receive
                experiences from.
        """
        self.algo = algo
        self.experience_replay = experience_replay
        self.experience_queue = experience_queue

    def _send_rollouts_thread(self, learner_queue: Queue):
        """
        Sends samples from a replay buffer to a queue, then recieves the updated
        Q-value and Q-target to update the experience priorities in the replay
        buffer.

        Args:
            learner_queue (multiprocessing.Queue): The queue to send experiences
                to and received updated values and targets from.
        """
        running = True

        while self.running:
            pass
        
    def train(self, agent_procs, mp_event, batch_size, start_size, save_path,
        save_interval):
        """
        Trains the algorithm until all agent processes have ended.

        Args:
            agent_procs (List[torch.multiprocessing.Process]): A list of agent
                processes.
            mp_event (torch.multiprocessing.Event): The event to set to awaken
                the agents to exit.
            batch_size (int): The size of the training batch.
            start_size (int): The number of samples in the buffer to start
                training.
            save_path (str): The directory to save the model to.
            save_interval (int): The number of training steps in-between
                model saves.
        """
        done_count = 0
        while any(proc.is_alive() for proc in agent_procs):
            if done_count == len(agent_procs):
                # Wait on the train processes
                mp_event.set()
                for proc in agent_procs:
                    proc.join()
            else:
                experience = self.experience_queue.get()

                if experience is None:
                    done_count += 1
                else:
                    self.experience_replay.add(experience)
                    self.algo.train_from_buffer(
                        self.experience_replay, batch_size, start_size,
                        save_path, save_interval
                    )