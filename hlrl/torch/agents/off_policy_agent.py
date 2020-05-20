import torch

from collections import deque
from hlrl.core.logger import TensorboardLogger

from .agent import TorchRLAgent


class OffPolicyAgent(TorchRLAgent):
    """
    An agent that collects (state, action, reward, next state) tuple
    observations
    """
    def _n_step_decay(self, experiences, decay):
        """
        Perform n-step decay on experiences of ((s, a, r, ...), ...) tuples
        """
        reward = 0
        for experience in list(experiences)[::-1]:
            reward = experience[0][2] + decay * reward

        return reward

    def _get_buffer_experience(self, experiences, decay):
        """
        Perpares the experience to add to the buffer.
        """
        reward = self._n_step_decay(experiences, decay)

        experience = experiences.pop()

        experience, algo_extras, next_algo_extras = experience
        q_val = algo_extras[0]
        next_q = next_algo_extras[0]

        experience[2] = reward

        target_q_val = reward + decay * next_q

        buffer_experience = (experience, q_val, target_q_val, *algo_extras[1:],
                             *next_algo_extras[1:])

        return buffer_experience

    def add_to_buffer(self, experience_queue, experiences, decay):
        """
        Adds the experience to the replay buffer.
        """
        experience = self._get_buffer_experience(experiences, decay)
        experience_queue.put(experience)

    def train(self, num_episodes, experience_queue, decay, n_steps):
        """
        Trains the algorithm for the number of episodes specified on the
        environment.

        Args:
            num_episodes (int): The number of episodes to train for.
            experience_queue (Queue): The queue to store experiences in.
            decay (float): The decay of the next.
            n_steps (int): The number of steps.
        """
        for episode in range(self.algo.env_episodes + 1, num_episodes + 1):
            self.env.reset()
            ep_reward = 0
            experiences = deque(maxlen=n_steps)
            while(not self.env.terminal):
                (state, action, reward, next_state, terminal, info, inp_extras,
                 algo_extras, next_inp_extras) = self.step()

                next_algo_step = self.algo.step(next_state, *next_inp_extras)
                next_algo_step = self.transform_algo_step(next_algo_step)
                next_actions, next_algo_extras = (next_algo_step[0],
                                                  next_algo_step[1:])

                ep_reward += reward

                experiences.append([[state, action, reward, next_state,
                                     terminal, *inp_extras], algo_extras,
                                    next_algo_extras])

                self.algo.env_steps += 1

                if (len(experiences) == n_steps):
                    # Do n-step decay and add to the buffer
                    self.add_to_buffer(experience_queue, experiences, decay)

            # Add the rest to the buffer
            while len(experiences) > 0:
                self.add_to_buffer(experience_queue, experiences, decay)

            if(self.logger is not None):
                self.logger["Train/Episode Reward"] = (ep_reward, episode)

            print("Episode", str(episode) + ":", ep_reward)
            self.algo.env_episodes += 1

        experience_queue.put(None)
