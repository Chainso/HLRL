from typing import Any, Dict, List, OrderedDict, Tuple

from hlrl.core.agents.agent import RLAgent
from hlrl.core.common.wrappers import MethodWrapper


class UDRLAgent(MethodWrapper):
    """
    An Upside-down Reinforcement Learning agent that takes commands.
    https://arxiv.org/pdf/1912.02875.pdf
    """
    def __init__(self, agent: RLAgent):
        """
        Turns the agent into one that receives commands.
        """
        super().__init__(agent)

        self.command = None

    def set_command(self, command: Any) -> None:
        """
        Sets the hidden state to the given one.

        Args:
            command: The new value of the command.
        """
        self.command = command

    def transform_state(self, state: Any) -> OrderedDict[str, Any]:
        """
        Appends the command to the state.

        Args:
            state: The raw state to transform.

        Returns:
            The transformed state with the command in it.
        """
        transed_state = self.om.transform_state(state)
        transed_state["command"] = self.command

        return transed_state

    def reset(self) -> None:
        """
        Resets the agent's command.
        """
        self.set_command(self.obj.algo.sample_command())


class RewardHorizonAgent(UDRLAgent):
    """
    An UDLR agent that uses a target return and time horizon as its command.
    """
    def n_step_decay(
            self,
            experiences: Tuple[Dict[str, Any], ...],
            decay: float
        ) -> Any:
        """
        Sums the reward on experiences of ((s, a, r, ...), ...) tuples.

        Args:
            experiences: The experiences containing rewards.
            decay: The decay constant that will be ignored.

        Returns:
            The horizon return.
        """
        return self.om.n_step_decay(experiences, 1)

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
        single_step_reward = experiences[0]["reward"]
        experience = self.om.get_buffer_experience(experiences, decay)
        experience["single_reward"] = single_step_reward

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
        _, time_horizon = self.command

        if time_horizon == 0:
            while experiences:
                experience = self.get_buffer_experience(experiences, decay)
                 
                # Only keep time horizon for experience
                experience["command"] = experience["command"][1:]

                for key in experience:
                    if key not in ready_experiences:
                        ready_experiences[key] = []

                    ready_experiences[key].append(experience[key])

            self.set_command(self.om.obj.choose_command())
