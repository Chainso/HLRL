# Fixes self-type reference (RLAgent) for typing annotations
from __future__ import annotations
from abc import abstractmethod
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, OrderedDict
from time import time

from hlrl.core.envs import Env
from hlrl.core.algos import RLAlgo

class RLAgent():
    """
    An agent that collects (state, action, reward, next state) tuple
    observations.
    """
    def __init__(
            self,
            env: Env,
            algo: RLAlgo,
            render: bool = False,
            silent: bool = False,
            logger: Optional[str] = None
        ):
        """
        Creates an agent that interacts with the given environment using the
        algorithm given.

        Args:
            env: The environment the agent will explore in.
            algo: The algorithm the agent will use the explore the environment.
            render: If the environment is to be rendered (if applicable)
            silent: If true, does not output to standard output.
            logger: The logger to log results while interacting with the
                environment.
        """
        self.env = env
        self.algo = algo
        self.render = render
        self.silent = silent
        self.logger = logger

    @staticmethod
    def train_from_builder(
            agent_builder: Callable[[Env], RLAgent],
            env_builder: Env,
            *train_args: Any,
            **train_kwargs: Any
        ) -> None:
        """
        Creates an agent and starts training when given a builder function to
        create the agent and a builder function to create the environment to
        train in.

        Args:
            agent_builder: A function that creates an agent when given the
                environment.
            env_builder: A builder function to create an environment for an
                agent.
        """
        agent = agent_builder(env=env_builder())
        agent.train(*train_args, **train_kwargs)

    def _add_prefix(
            self,
            val_dict: Dict[str, Any],
            prefix: str
        ) -> OrderedDict[str, Any]:
        """
        Returns a copy of map with the keys having a prefix.

        Args:
            map: The map of key-value pairs.
            prefix: The prefix to prepend to each key.

        Returns:
            The new ordered dictionary with the prefix prepended to each key.
        """
        return OrderedDict(
            (prefix + key, value) for key, value in val_dict.items()
        )

    def transform_state(self,
                        state: Any) -> OrderedDict[str, Any]:
        """
        Creates the dictionary of algorithm inputs from the env state.

        Args:
            state: The state to add to the dictionary.

        Returns:
            An ordered dictionary with a key "state" mapping to the state.
        """
        return OrderedDict((
            ("state", state),
        ))

    def transform_next_state(self,
                             next_state: Any) -> OrderedDict[str, Any]:
        """
        Transforms the next observation of an environment to a dictionary.
        
        Args:
            next_state: The next state to add to the dictionary.

        Returns:
            An ordered dictionary with a key "next_state" mapping to the next
            state.
        """
        transed_ns = self.transform_state(next_state)
        transed_ns = self._add_prefix(transed_ns, "next_")

        return transed_ns

    def transform_algo_step(
        self,
        algo_step: Tuple[Any, ...]) -> OrderedDict[str, Any]:
        """
        Transforms the algorithm step on the observation to a dictionary.
        
        Args:
            algo_step: The outputs of the algorithm on the input state.

        Returns:
            An ordered dictionary of the algorithm "action" -> action.
        """
        return OrderedDict((
            ("action", algo_step[0]),
        ))

    def transform_next_algo_step(
        self,
        next_algo_step: Tuple[Any, ...]) -> OrderedDict[str, Any]:
        """
        Transforms the next algorithm step on the observation to a dictionary.
        
        Args:
            next_algo_step: The outputs of the algorithm on the next input
                state.

        Returns:
            An ordered dictionary of the next algorithm outputs, prefixed with
            "next_"
        """
        transed_nas = self.transform_algo_step(next_algo_step)
        transed_nas = self._add_prefix(transed_nas, "next_")

        return transed_nas

    def transform_reward(self,
                         state: Any,
                         algo_step: OrderedDict[str, Any],
                         reward: Any,
                         terminal: Any,
                         next_state: Any) -> Any:
        """
        Transforms the reward of an environment step.

        Args:
            state: The state of the environment.
            algo_step: The transformed algorithm step of the state.
            reward: The reward from the environment.
            terminal: If the next state is a terminal state.
            next_state: The new state of the environment.

        Returns:
            The transformed reward.
        """
        return reward

    def transform_terminal(self,
                           terminal: Any) -> Any:
        """
        Transforms the terminal of an environment step.

        Args:
            terminal: The terminal value to transform.

        Returns:
            The transformed terminal.
        """
        return terminal

    def transform_action(self,
                         action: Any) -> Any:
        """
        Transforms the action of the algorithm output to be usable with the
        environment.

        Args:
            action: The action to transform to environment-usable

        Returns:
            The action to be taken in the environment.
        """
        return action

    def reward_to_float(self,
                        reward: float) -> float:
        """
        Converts the reward to a single float value.

        Args:
            reward: The reward to turn into a float.

        Returns:
            The float value of the reward.
        """
        return reward

    def reset(self):
        """
        Resets the agent.
        """
        # The simplest agent doesn't require anything to reset
        pass

    def create_batch(
            self,
            ready_experiences: Dict[str, List[Any]],
        ) -> Dict[str, List[Any]]:
        """
        Creates a batch of experiences to be trained on from the ready
        experiences.

        Args:
            ready_experiences: The experiences to be trained on.
        
        Returns:
            A dictionary of each field necessary for training.
        """
        # The simplest agent doesn't need to do anything
        return ready_experiences

    def n_step_decay(
            self,
            experiences: Tuple[Dict[str, Any], ...],
            decay: float
        ) -> Any:
        """
        Perform n-step decay on experiences of ((s, a, r, ...), ...) tuples

        Args:
            experiences: The experiences containing rewards.
            decay: The decay constant.

        Returns:
            The decayed reward.
        """
        # Want to check for terminals as well since envs may auto reset
        decayed_experience = experiences.popleft()

        reward = decayed_experience["reward"]
        terminal_mask = 1 - decayed_experience["terminal"]

        for i, experience in enumerate(experiences, start=1):
            reward += terminal_mask * (decay ** i) * experience["reward"]
            terminal_mask *= 1 - experience["terminal"]

        experiences.appendleft(decayed_experience)

        return reward

    def get_buffer_experience(
            self,
            experiences: List[Dict[str, Any]],
            decay: float
        ) -> Any:
        """
        Perpares the experience to add to the buffer.

        Args:
            experiences: The experiences containing rewards.
            decay: The decay constant.

        Returns:
            The oldest stored experience.
        """
        decayed_reward = self.n_step_decay(experiences, decay)

        experience = experiences.popleft()
        experience["reward"] = decayed_reward

        # Correct for n_step
        if len(experiences) > 0:
            for key in experience:
                if key.startswith("next_"):
                    experience[key] = experiences[-1][key]

            experience["terminal"] = experiences[-1]["terminal"]

        return experience

    def add_to_buffer(
            self,
            ready_experiences: Dict[str, List[Any]],
            experiences: List[Dict[str, Any]],
            decay: float
        ) -> None:
        """
        Prepares the oldest experiences from experiences and transfers it to
        ready experiences.

        Args:
            ready_experiences: The buffer of experiences that can be trained on.
            experiences: The experiences containing rewards.
            decay: The decay constant.
        """
        experience = self.get_buffer_experience(experiences, decay)

        for key in experience:
            if key not in ready_experiences:
                ready_experiences[key] = []

            ready_experiences[key].append(experience[key])

    def train_step(
            self,   
            ready_experiences: Dict[str, List[Any]],
            batch_size: int,
            learner: Any,
            *learner_args: Any,
            **learner_kwargs: Any
        ) -> Dict[str, List[Any]]:
        """
        Trains on the ready experiences.

        Args:
            ready_experiences: The buffer of experiences that can be trained on.
            batch_size: The batch size for training.
            learner: Any object responsible for the training of the algorithm.
            learner_args: Any positional arguments for the learner.
            learner_kwargs: Any keyword arguments for the learner.

        Returns:
            The unused ready experiences.
        """
        # Get length of a random key
        keys = list(ready_experiences)
        if len(keys) > 0:
            key = keys[0]
            if len(ready_experiences[key]) == batch_size:
                learn_batch = self.create_batch(ready_experiences)
                learner.train_batch(
                    learn_batch, *learner_args, **learner_kwargs
                )

                ready_experiences = {}

        return ready_experiences

    def step(self,
             with_next_step: bool = False) -> None:
        """
        Takes 1 step in the agent's environment. Returns the experience
        dictionary. Resets the environment if the current state is a
        terminal.

        Args:
            with_next_step: If true, runs the next state through the model as
                well.
        """
        if self.env.terminal:
            self.reset()
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
        reward = self.transform_reward(
            state, algo_step, reward, terminal, next_state
        )
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

    def play(self,
             num_episodes: int = 0,
             num_steps: int = 0) -> None:
        """
        Resets and plays the environment with the algorithm. Returns the average
        reward per episode. Will prioritize the inputted number of episodes to
        the number of steps if both are given.

        Args:
            num_episodes: The number of episodes to play for.
            num_steps: The number of steps to play for.
        """
        if num_episodes <= 0 and num_steps <= 0:
            raise ValueError(
                "One of num_episodes and num_steps must be a positive integer"
            )
        elif num_episodes > 0:
            using_episodes = True
        else:
            using_episodes = False

        avg_reward = 0
        episode = 0
        step = 0

        while ((using_episodes and episode < num_episodes)
               or (not using_episodes and step < num_steps)):
            self.reset()
            self.env.reset()
            ep_reward = 0
            
            while (not self.env.terminal
                   and (using_episodes or step < num_steps)):
                reward = self.step()["reward"]
                reward = self.reward_to_float(reward)

                ep_reward += reward
                step += 1

            episode += 1

            if self.logger is not None:
                self.logger["Play/Episode Reward"] = (ep_reward, episode)

            if not self.silent:
                print("Episode {0}\t|\tStep {1}\t|\tReward: {2}".format(
                    episode, step, ep_reward
                ))

            avg_reward += ep_reward

        avg_reward /= episode

        if self.logger is not None:
            self.logger["Play/Average Reward"] = avg_reward

        if not self.silent:
            print("--------------------------------")
            print("Average Reward:", avg_reward)

    def train(
            self,
            num_episodes: int,
            batch_size: int,
            decay: float,
            n_steps: int,
            learner: Any,
            *learner_args: Any,
            exit_condition: Optional[Callable[[], bool]] = None,
            **learner_kwargs: Any
        ) -> None:
        """
        Trains the algorithm for the number of episodes specified on the
        environment.

        Args:
            num_episodes: The number of episodes to train for.
            batch_size: The number of ready experiences to train on at a time.
            decay: The decay of the next.
            n_steps: The number of steps.
            learner: Any object responsible for the training of the algorithm.
            *learner_args: Any positional arguments for the learner.
            exit_condition: An alternative exit condition to num episodes which
                will be used if given.
            **learner_kwargs: Any keyword arguments for the learner.
        """
        if self.logger is not None:
            agent_train_start_time = time()

        ready_experiences = {}

        episode = 0

        while (episode < num_episodes
               or (exit_condition is not None and not exit_condition())):
            if self.logger is not None:
                episode_time = time()

            self.reset()
            self.env.reset()

            ep_reward = 0
            experiences = deque(maxlen=n_steps)

            while (not self.env.terminal
                   and (exit_condition is None or not exit_condition())):
                if self.logger is not None:
                    step_time = time()

                experience = self.step(True)
                ep_reward += self.reward_to_float(experience["reward"])

                experiences.append(experience)
 
                self.algo.env_steps += 1

                if len(experiences) == n_steps:
                    # Do n-step decay and add to the buffer
                    self.add_to_buffer(ready_experiences, experiences, decay)

                ready_experiences = self.train_step(
                    ready_experiences, batch_size, learner, *learner_args,
                    **learner_kwargs
                )

                if self.logger is not None:
                    self.logger["Train/Agent Steps per Second"] = (
                        1 / (time() - step_time), self.algo.env_steps
                    )

            episode += 1

            # Add the rest to the buffer
            while len(experiences) > 0:
                self.add_to_buffer(ready_experiences, experiences, decay)

                ready_experiences = self.train_step(
                    ready_experiences, batch_size, learner, *learner_args,
                    **learner_kwargs
                )

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
                print("Episode {0}\t|\tStep {1}\t|\tReward: {2}".format(
                    self.algo.env_episodes, self.algo.env_steps, ep_reward
                ))
