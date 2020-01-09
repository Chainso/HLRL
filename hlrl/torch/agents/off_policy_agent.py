import torch

from collections import deque

from .agent import TorchRLAgent

class OffPolicyAgent(TorchRLAgent):
    """
    An agent that collects (state, action, reward, next state) tuple
    observations
    """
    def __init__(self, env, algo, experience_replay, render=False, logger=None,
                 device="cpu"):
        """
        Creates an agent that interacts with the given environment using the
        algorithm given.

        Args:
            env (Env): The environment the agent will explore in.
            algo (TorchRLAlgo): The algorithm the agent will use the explore the
                                environment.
            experience_replay (ExperienceReplay): The experience replay to store
                                                  (state, action, reward,
                                                   next state) tuples in.
            render (bool): If the environment is to be rendered (if applicable).
            logger (Logger, optional) : The logger to log results while
                                        interacting with the environment.
            device (str): The device for the agent to run on.
        """
        super().__init__(env, algo, render, logger, device)
        self.experience_replay = experience_replay

    def _n_step_decay(self, experiences, decay):
        """
        Perform n-step decay on experiences of ((s, a, r, ...), ...) tuples
        """
        reward = 0
        for experience in list(experiences)[::-1]:
            reward += experience[0][2] + decay * reward

        return reward

    def _get_buffer_experience(self, experiences, decay):
        """
        Perpares the experience to add to the buffer.
        """
        reward = self._n_step_decay(experiences, decay)
        experiences[0][0][2] = reward

        q_val = experiences[0][1][0]
        next_q_val = experiences[0][-1][0]

        target_q_val = reward + decay * next_q_val

        return experiences

    def add_to_buffer(self, experiences, decay):
        """
        Adds the experience to the replay buffer.
        """
        experiences = self._get_buffer_experience(experiences, decay)
        self.experience_replay.add(experiences[0][0], *experiences[0][1],
                                   *experiences[0][-1])
 
    def put_in_queue(self, experiences, decay, queue):
        """
        Puts the experience in multiprocessing queue.
        """
        experiences = self._get_buffer_experience(experiences, decay)
        queue.put((experiences[0][0], *experiences[0][1], *experiences[0][-1]))

    def train(self, num_episodes, decay, n_steps, queue=None):
        """
        Trains the algorithm for the number of episodes specified on the
        environment.

        Args:
            num_episodes (int): The number of episodes to train for.
            decay (float): The decay of the next.
            n_steps (int): The number of steps.
            queue (mp.Queue): A queue to put replays if using multiprocessing.
        """
        for episode in range(1, num_episodes + 1):
            self.env.reset()
            ep_reward = 0
            experiences = deque(maxlen = n_steps)
            while(not self.env.terminal):
                (state, action, reward, next_state, terminal, info,
                add_algo_ret) = self.step()

                next_algo_ret = self.algo.step(next_state)[1:]
                ep_reward += reward

                # Convert the reward and terminal into a tensor for storage
                reward = torch.FloatTensor([[reward]]).to(self.device)
                terminal = torch.FloatTensor([[terminal]]).to(self.device)

                experiences.append([[state, action, reward, next_state,
                                     terminal], *add_algo_ret, *next_algo_ret])

                self.algo.env_steps += 1

                if (len(experiences) == n_steps):
                    # Do n-step decay and add to the buffer
                    if queue is not None:
                        self.put_in_queue(experiences, decay, queue)
                    else:
                        self.add_to_buffer(experiences, decay)

            if queue is not None:
                self.put_in_queue(experiences, decay, queue)

                # Make sure the queue finishes adding the experiences before
                # garbage collecting the experience buffer
                # No queue.join() in multiprocessing.Queue
                # Unsure of reliability of this method
                while queue.empty():
                    pass
            else:
                self.add_to_buffer(experiences, decay)

            if(self.logger is not None):
                self.logger["Train/Episode Reward"] = ep_reward, episode
            print("Episode", str(episode) + ":", ep_reward)
            self.algo.env_episodes += 1
