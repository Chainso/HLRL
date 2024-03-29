# Fixes self-type reference (RLAgent) for typing annotations
from __future__ import annotations
from typing import Any, Dict, Tuple, OrderedDict

from hlrl.core.agents import RLAgent
from hlrl.core.common.wrappers import MethodWrapper

class NStepAgent(MethodWrapper):
    """
    An agent that calculates N-step returns.
    """
    def __init__(
            self,
            agent: RLAgent,
            decay: float
        ):
        """
        Creates the N-step agent.

        Args:
            agent: The underlying agent to wrap.
            decay: The decay of each step in returns.
        """
        super().__init__(agent)

        self.decay = decay

    def transform_algo_step(
            self,
            algo_step: Tuple[Any, ...]
        ) -> OrderedDict[str, Any]:
        """
        Transforms the algorithm step on the observation to a dictionary.
        
        Args:
            algo_step: The outputs of the algorithm on the input state.

        Returns:
            An ordered dictionary of the algorithm step with the value added.
        """
        transed_algo_step = self.om.transform_algo_step(algo_step[:-1])
        transed_algo_step["value"] = algo_step[-1]
        
        return transed_algo_step

    def after_step(
            self,
            experience: Dict[str, Any],
            next_algo_inp: OrderedDict[str, Any]
        ) -> None:
        """
        Adds the bootstrap value in the case where the agent has not terminated
        but the environment has, or sets to 0 otherwise.

        Args:
            experience: The experience generated by the step.
            next_algo_inp: The inputs to the algorithm to process the next
                state.
        """
        self.om.after_step(experience, next_algo_inp)

        if experience["truncated"]:
            algo_step = self.algo.step(*next_algo_inp.values())
            algo_step = self.transform_algo_step(algo_step)

            experience["next_value"] = algo_step["value"]
        else:
            # Sets 0 for multiple types instead of just int
            experience["next_value"] = experience["value"] * 0

    def prepare_experiences(
            self,
            experiences: Tuple[Dict[str, Any], ...]
        ) -> Any:
        """
        Performs N-step decay on the experiences.

        Args:
            experiences: The experiences to add.

        Returns:
            The prepared experiences to add to the replay buffer.
        """
        experiences = self.om.prepare_experiences(experiences)
        next_return = 0

        for t in reversed(range(len(experiences))):
            # "next_value" supplies the bootstrap if the environment terminated
            # early but the agent didn't
            non_terminal = 1 - experiences[t]["terminal"]

            discounted_term = self.decay * (
                experiences[t]["next_value"]
                + non_terminal * next_return
            )

            experiences[t]["reward"] = next_return = (
                experiences[t]["reward"] + discounted_term
            )

        return experiences

    def clean_experiences(
            self,
        experiences: Tuple[Dict[str, Any], ...]
        ) -> Any:
        """
        Removes the next value from the experiences.

        Args:
            experiences: The prepared experiences to clean.

        Returns:
            The prepared experiences without the next values.
        """
        for experience in experiences:
            del experience["next_value"]

        return self.om.clean_experiences(experiences)
