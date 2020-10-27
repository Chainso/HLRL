import torch.multiprocessing as mp
import os

from abc import abstractmethod
from collections import OrderedDict, deque
from typing import Any, Dict, List, NoReturn, Tuple
from time import time

class RLAgent():
    """
    An agent that collects (state, action, reward, next state) tuple
    observations
    """
    def __init__(self, env, algo, render=False, silent=False, logger=None):
        """
        Creates an agent that interacts with the given environment using the
        algorithm given.

        Args:
            env (Env): The environment the agent will explore in.

            algo (RLAlgo): The algorithm the agent will use the explore the
                           environment.
            render (bool): If the environment is to be rendered (if applicable)
            silent (bool): If true, does not output to standard output.
            logger (Logger, optional) : The logger to log results while
                                        interacting with the environment.
        """
        self.env = env
        self.algo = algo
        self.render = render
        self.silent = silent
        self.logger = logger

    def _add_prefix(self, map, prefix):
        """
        Returns a copy of map with the keys having a prefix.
        """
        return {prefix + key : value for key, value in map.items()}

    def transform_state(self, state):
        """
        Creates the dictionary of algorithm inputs from the env state
        """
        return OrderedDict({
            "state": state
        })

    def transform_next_state(self, next_state):
        """
        Transforms the next observation of an environment to a dictionary.
        """
        transed_ns = self.transform_state(next_state)
        transed_ns = self._add_prefix(transed_ns, "next_")

        return transed_ns

    def transform_algo_step(self, algo_step):
        """
        Transforms the algorithm step on the observation to a dictionary.
        """
        return OrderedDict({
            "action": algo_step[0]
        })

    def transform_next_algo_step(self, next_algo_step):
        """
        Transforms the next algorithm step on the observation to a dictionary.
        """
        transed_nas = self.transform_algo_step(next_algo_step)
        transed_nas = self._add_prefix(transed_nas, "next_")

        return transed_nas

    def transform_reward(self, state, algo_step, reward, next_state):
        """
        Transforms the reward of an environment step.

        Args:
            state (Any): The state of the environment.
            action (Any): The last action taken in the environment.
            reward (Any): The reward from the environment.
            next_state (Any): The new state of the environment.
        """
        return reward

    def transform_terminal(self, terminal):
        """
        Transforms the terminal of an environment step.
        """
        return terminal

    def transform_action(self, action):
        """
        Transforms the action of the algorithm output to be usable with the
        environment.
        """
        return action

    def reward_to_float(self, reward):
        """
        Converts the reward to a single float value.

        Args:
            reward (Any): The reward to turn into a float.
        """
        return reward

    def reset(self):
        """
        Resets the agent.
        """
        # The simplest agent doesn't require anything to reset
        pass

    def n_step_decay(self,
                     experiences: Tuple[Dict[str, Any], ...],
                     decay: float) -> Any:
        """
        Perform n-step decay on experiences of ((s, a, r, ...), ...) tuples

        Args:
            experiences: The experiences containing rewards.
            decay: The decay constant.

        Returns:
            The decayed reward.
        """
        reward = experiences[-1]["reward"]
        for experience in list(experiences)[:-1:-1]:
            reward = experience["reward"] + decay * reward

        return reward

    def get_buffer_experience(self,
                              experiences: Tuple[Dict[str, Any], ...],
                              decay: float) -> Any:
        """
        Perpares the experience to add to the buffer.

        Args:
            experiences: The experiences containing rewards.
            decay: The decay constant.

        Returns:
            The oldest stored experience.
        """
        decayed_reward = self.n_step_decay(experiences, decay)
        experience = experiences.pop()
        experience["reward"] = decayed_reward

        return experience

    def add_to_buffer(self,
                      ready_experiences: List[Dict[str, Any], ...],
                      experiences: Tuple[Dict[str, Any], ...],
                      decay: float) -> None:
        """
        Prepares the oldest experiences from experiences and transfers it to
        ready experiences.

        Args:
            ready_experiences: The buffer of experiences that can be trained on.
            experiences: The experiences containing rewards.
            decay: The decay constant.
        """
        ready_experiences.append(self.get_buffer_experience(experiences, decay))

    def train_experiences(
        self,
        ready_experiences: List[Dict[str, Any], ...],
        trainer: Any) -> NoReturn:
        """
        Trains on the ready experiences if the batch size is met.

        Args:
            ready_experiences: The buffer of experiences that can be trained on.
            trainer: Any object responsible for the training of the algorithm.
        """
        raise NotImplementedError

    def step(self, with_next_step: bool = False) -> None:
        """
        Takes 1 step in the agent's environment. Returns the experience
        dictionary. Resets the environment if the current state is a
        terminal.

        Args:
            with_next_step: If true, runs the next state through the model as
                well.
        """
        if self.env.terminal:
            self.env.reset()

        if self.render:
            self.env.render()

        state = self.env.state
        algo_inp = self.transform_state(state)
        state = algo_inp.pop("state")

        algo_step = self.algo.step(state, *algo_inp.values())
        algo_step = self.transform_algo_step(algo_step)

        env_action = self.transform_action(algo_step["action"])
        next_state, reward, terminal, _ = self.env.step(env_action)

        next_algo_inp = self.transform_next_state(next_state)
        next_state = next_algo_inp.pop("next_state")
        reward = self.transform_reward(state, algo_step, reward, next_state)
        terminal = self.transform_terminal(terminal)
    
        experience = OrderedDict({
            "state": state,
            **algo_inp,
            **algo_step,
            "reward": reward,
            "next_state": next_state,
            **next_algo_inp,
            "terminal": terminal
        })

        if with_next_step:
            next_algo_step = self.algo.step(next_state, *next_algo_inp.values())
            next_algo_step = self.transform_next_algo_step(next_algo_step)

            experience.update(next_algo_step)

        return experience

    def play(self, num_episodes):
        """
        Resets and plays the environment with the algorithm. Returns the average
        reward per episode.

        Args:
            num_episodes (int): The numbers of episodes to play for.
        """
        avg_reward = 0

        for episode in range(1, num_episodes + 1):
            self.reset()
            self.env.reset()
            ep_reward = 0
            
            while(not self.env.terminal):
                reward = self.step()["reward"]
                reward = self.reward_to_float(reward)

                ep_reward += reward

            if self.logger is not None:
                self.logger["Play/Episode Reward"] = ep_reward, episode

            if not self.silent:
                print("Episode {0} | Reward: {1}".format(episode, ep_reward))

            avg_reward += ep_reward / num_episodes

        if self.logger is not None:
            self.logger["Play/Average Reward"] = avg_reward

        if not self.silent:
            print("-------------------")
            print("Average Reward:", avg_reward)

        return avg_reward

    @abstractmethod
    def train(self,
              num_episodes: int,
              batch_size: int,
              decay: float,
              n_steps: int,
              trainer: Any) -> None:
        """
        Trains the algorithm for the number of episodes specified on the
        environment.

        Args:
            num_episodes: The number of episodes to train for.
            batch_size: The number of ready experiences to train on at a time.
            decay: The decay of the next.
            n_steps: The number of steps.
            trainer: Any object responsible for the training of the algorithm.
        """
        if self.logger is not None:
            agent_train_start_time = time()

        ready_experiences = []

        for _ in range(1, num_episodes + 1):
            if self.logger is not None:
                episode_time = time()

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

                if len(experiences) == n_steps:
                    # Do n-step decay and add to the buffer
                    self.add_to_buffer(ready_experiences, experiences, decay)

                if len(ready_experiences) == batch_size:
                    self.train_experiences(ready_experiences, trainer)

                if self.logger is not None:
                    self.logger["Train/Agent Steps per Second"] = (
                        1 / (time() - step_time), self.algo.env_steps
                    )

            # Add the rest to the buffer
            while len(experiences) > 0:
                self.add_to_buffer(ready_experiences, experiences, decay)

                if len(ready_experiences) == batch_size:
                    self.train_experiences(ready_experiences, trainer)

            self.algo.env_episodes += 1

            if self.logger is not None:
                self.logger["Train/Episode Reward"] = (
                    ep_reward, self.algo.env_episodes
                )

                self.logger["Train/Episode Reward over Wall Time (s)"] = (
                    ep_reward, time() - agent_train_start_time
                )

                self.logger["Train/Agent Episodes per Second"] = (
                    1 / (time() - episode_time), self.algo.env_episodes
                )

            if not self.silent:
                print("Episode {0} | Step {1} | Reward: {2}".format(
                    self.algo.env_episodes, self.algo.env_steps, ep_reward
                ))

class AgentPool():
    """
    An asynchronous pool of agents.
    """
    def __init__(self, agents):
        """
        Properties:
            agents ([RLAgent]): A list of agents in the pool.
        """
        self.agents = agents

    def train_process(self, *agent_train_args):
        """
        Trains each agent using the arguments given.

        Args:
            agent_train_args (*[object]): The list of arguments for each agent.
        """
        procs = [mp.Process(target = agent.train_process, args=agent_train_args)
                 for agent in self.agents]

        for proc in procs:
            proc.start()

        return procs