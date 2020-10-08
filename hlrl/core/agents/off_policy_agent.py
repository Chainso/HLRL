from collections import deque
from time import time

from hlrl.core.agents import RLAgent

class OffPolicyAgent(RLAgent):
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

    def get_buffer_experience(self, experiences, decay):
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
        transed_algo_step = {
            **super().transform_algo_step(algo_step[:-1]),
            "q_val": algo_step[-1]
        }

        return transed_algo_step

    def add_to_buffer(self, experiences, decay, experience_queue):
        """
        Adds the experience to the replay buffer.
        """
        experience = self.get_buffer_experience(experiences, decay)
        experience_queue.put(experience)

    def train_algo(self, experiences, decay, experience_replay, algo,
        *algo_args, **algo_kwargs):
        """
        Puts an experience in the buffer and trains the algorithm with a single
        batch from the buffer.

        Args:
            experiences: The experiences to add to the buffer.
            decay (float): The decay value for n_step decay.
            experience_replay (ExperienceReplay): The experience play buffer to
                store experiences in.
            algo (RLAlgo): The off policy algirithm to train.
            *algo_args (Tuple[Any, ...]): Any positional arguments for the
                algorithm training.
            **algo_kwargs (Dict[str, ...]): Any keyword arguments for the
                algorithm training. 
        """
        experience = self.get_buffer_experience(experiences, decay)
        experience_replay.add(experience)

        algo.train_from_buffer(
            experience_replay, *algo_args, **algo_kwargs
        )

    def train_process(self, num_episodes, decay, n_steps, experience_queue,
        done_event):
        """
        Trains the algorithm for the number of episodes specified on the
        environment, to be called in a separate process.

        Args:
            num_episodes (int): The number of episodes to train for.
            decay (float): The decay of the next.
            n_steps (int): The number of steps.
            experience_queue (Queue): The queue to store experiences in.
            done_event (Event): The event to wait on before exiting the process.
        """
        for episode in range(1, num_episodes + 1):
            self.reset()
            self.env.reset()

            ep_reward = 0
            experiences = deque(maxlen=n_steps)

            while(not self.env.terminal):
                experience = self.step(True)

                ep_reward += self.reward_to_float(experience["reward"])
                
                experiences.append(experience)

                self.algo.env_steps += 1

                if (len(experiences) == n_steps):
                    # Do n-step decay and add to the buffer
                    self.add_to_buffer(experiences, decay, experience_queue)

            # Add the rest to the buffer
            while len(experiences) > 0:
                self.add_to_buffer(experiences, decay, experience_queue)

            self.algo.env_episodes += 1

            if self.logger is not None:
                self.logger["Train/Episode Reward"] = (
                    ep_reward, self.algo.env_episodes
                )

            if not self.silent:
                print("Episode {0} Step {1} Reward: {2}".format(
                    self.algo.env_episodes, self.algo.env_steps, ep_reward
                ))

        done_event.wait()

    def train(self, num_episodes, decay, n_steps, experience_replay, algo,
        *algo_args, **algo_kwargs):
        """
        Trains the algorithm for the number of episodes specified on the
        environment, to be called in a separate process.

        Args:
            num_episodes (int): The number of episodes to train for.
            decay (float): The decay of the next.
            n_steps (int): The number of steps.
            experience_replay (ExperienceReplay): The experience play buffer to
                store experiences in.
            algo (RLAlgo): The off policy algirithm to train.
            *algo_args (Tuple[Any, ...]): Any positional arguments for the
                algorithm training.
            **algo_kwargs (Dict[str, ...]): Any keyword arguments for the
                algorithm training. 
        """
        for episode in range(1, num_episodes + 1):
            self.reset()
            self.env.reset()

            ep_reward = 0
            experiences = deque(maxlen=n_steps)

            while(not self.env.terminal):
                if self.logger is not None:
                    step_time = time()

                experience = self.step(True)

                ep_reward += self.reward_to_float(experience["reward"])
                
                experiences.append(experience)

                self.algo.env_steps += 1

                if (len(experiences) == n_steps):
                    # Do n-step decay and add to the buffer
                    self.train_algo(
                        experiences, decay, experience_replay, algo, *algo_args,
                        **algo_kwargs
                    )

                if self.logger is not None:
                    self.logger["Train/Environment Step Time (s)"] = (
                        time() - step_time, self.algo._env_steps
                    )

            # Add the rest to the buffer
            while len(experiences) > 0:
                self.train_algo(
                    experiences, decay, experience_replay, algo, *algo_args,
                    **algo_kwargs
                )

            self.algo.env_episodes += 1

            if self.logger is not None:
                self.logger["Train/Episode Reward"] = (
                    ep_reward, self.algo.env_episodes
                )

            if not self.silent:
                print("Episode {0} Step {1} Reward: {2}".format(
                    self.algo.env_episodes, self.algo.env_steps, ep_reward
                ))