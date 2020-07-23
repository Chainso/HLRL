class Worker():
    """
    A worker for an off-policy algorithm, used to separate training and the
    agent.
    """
    def __init__(self, algo, experience_replay, experience_queue):
        """
        Creates worker for the algorithm, and with an experience queue to be fed
        experiences from.

        Args:
            algo (torch.nn.Module): The algorithm to train.
            experience_replay (ExperienceReplay):   The replay buffer to add
                                                    experiences into.
            experience_queue (torch.multiprocessing.Queue): The queue to receive
                                                            experiences from.
        """
        self.algo = algo
        self.experience_replay = experience_replay
        self.experience_queue = experience_queue


    def train(self, agent_procs, batch_size, start_size, save_path,
        save_interval):
        """
        Trains the algorithm until all agent processes have ended.

        Args:
            agent_procs ([torch.multiprocess.Process]): A list of agent
                                                        processes.
            batch_size (int):   The size of the training batch.
            start_size (int):   The number of samples in the buffer to start
                                training.
            save_path (str):    The directory to save the model to.
            save_interval (int):    The number of training steps in-between
                                    model saves.
        """
        done_count = 0
        while any(proc.is_alive() for proc in agent_procs):
            if done_count == len(agent_procs):
                # Wait on the train processes
                for proc in agent_procs:
                    proc.join()
            else:
                experience = self.experience_queue.get()

                if experience is None:
                    print("Done!")
                    done_count += 1
                else:
                    self.experience_replay.add(experience)
                    self.algo.train_from_buffer(
                        self.experience_replay, batch_size, start_size,
                        save_path, save_interval
                    )