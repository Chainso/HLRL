from collections import deque

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
        if len(experiences) > 0:
            reward = experiences[-1]["reward"]
            for experience in list(experiences)[:-1:-1]:
                reward = experience["reward"] + decay * reward

        return reward

    def _get_buffer_experience(self, experiences, decay):
        """
        Perpares the experience to add to the buffer.
        """
        decayed_reward = self._n_step_decay(experiences, decay)
        experience = experiences.pop()
        experience["reward"] = decayed_reward

        next_q_val = experience.pop("next_q_val")
        target_q_val = experience["reward"] + decay * next_q_val

        # Update experience with target q value
        experience["target_q_val"] = target_q_val

        return experience

    def transform_algo_step(self, algo_step):
        """
        Transforms the algorithm step on the observation to a dictionary.
        """
        # Action then q val
        transed_algo_step = super().transform_algo_step(algo_step)
        transed_algo_step["q_val"] = algo_step[1]

        return transed_algo_step

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
            self.reset()
            self.env.reset()

            ep_reward = 0
            experiences = deque(maxlen=n_steps)

            while(not self.env.terminal):
                experience = self.step(True)

                ep_reward += experience["reward"].item()
                
                experiences.append(experience)

                self.algo.env_steps += 1

                if (len(experiences) == n_steps):
                    # Do n-step decay and add to the buffer
                    self.add_to_buffer(experience_queue, experiences, decay)

            # Add the rest to the buffer
            while len(experiences) > 0:
                self.add_to_buffer(experience_queue, experiences, decay)

            if(self.logger is not None):
                self.logger["Train/Episode Reward"] = (ep_reward, episode)

            print("Episode {0} Step {1} Reward: {2}".format(
                self.algo.env_episodes, self.algo.env_steps, ep_reward
            ))

            self.algo.env_episodes += 1

        experience_queue.put(None)
