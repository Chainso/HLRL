import queue

from collections import deque
from time import time
from typing import Any, Dict, List, Tuple, OrderedDict

from hlrl.core.agents import RLAgent
from hlrl.core.experience_replay import ExperienceReplay

class OffPolicyAgent(RLAgent):
    """
    An agent that collects (state, action, reward, next state) tuple
    observations
    """
    def transform_algo_step(
        self,
        algo_step: Tuple[Any, ...]) -> OrderedDict[str, Any]:
        """
        Transforms the algorithm step on the observation to a dictionary.
        
        Args:
            algo_step: The outputs of the algorithm on the input state.

        Returns:
            An ordered dictionary of the algorithm "action" -> action and
            "q_val" -> Q-value.
        """
        transed_algo_step = super().transform_algo_step(algo_step[:-1])
        transed_algo_step["q_val"] = algo_step[-1]
        
        return transed_algo_step

    def create_batch(
            self,
            ready_experiences: Dict[str, List[Any]],
        ) -> Tuple[Dict[str, Any], ...]:
        """
        Creates a batch of experiences to be trained on from the ready
        experiences.

        Args:
            ready_experiences: The experiences to be trained on.
        
        Returns:
            A dictionary of each field necessary for training.
        """
        # Convert to list of dicts to send to replay buffer 1 by 1
        batch = tuple(
            dict(zip(ready_experiences, experience))
            for experience in zip(*ready_experiences.values())
        )

        return batch

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
        experience = super().get_buffer_experience(experiences, decay)

        next_q_val = experience.pop("next_q_val")

        # Correct for n_step
        if len(experiences) > 0:
            next_q_val = experiences[-1]["next_q_val"]

        target_q_val = experience["reward"] + decay * next_q_val

        # Update experience with target q value
        experience["target_q_val"] = target_q_val

        return experience

    def train_step(self,
                   ready_experiences: Dict[str, List[Any]],
                   batch_size: int,
                   experience_replay: ExperienceReplay,
                   *train_args: Any,
                   **train_kwargs: Any) -> bool:
        """
        Trains on the ready experiences if the batch size is met.

        Args:
            ready_experiences: The buffer of experiences that can be trained on.
            batch_size: The batch size for training.
            experience_replay: An experience replay buffer to add experiences
                to.
            train_args: Any positional arguments for the algorithm training.
            train_kwargs: Any keyword arguments for the algorithm training.

        Returns:
            True, if ready experiences were used, False if the batch was too
            small.
        """
        added = False

        # Get length of a random key
        keys = list(ready_experiences)
        if len(keys) > 0:
            key = keys[0]
            if len(ready_experiences[key]) == batch_size:
                experiences_to_add = self.create_batch(ready_experiences)

                for experience in experiences_to_add:
                    experience_replay.add(experience)

                added = True

        self.algo.train_from_buffer(
            experience_replay, *train_args, **train_kwargs
        )

        return added