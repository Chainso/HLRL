from typing import Any, Deque, Dict, List, Tuple

from hlrl.core.agents.wrappers import NStepAgent
from hlrl.core.experience_replay import ExperienceReplay

class OffPolicyAgent(NStepAgent):
    """
    An agent that collects (state, action, reward, next state) tuple
    observations.
    """
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
            A tuple of the dictionary of each field necessary for training,
            the Q-values and target Q-values of the experiences.
        """
        # Convert to list of dicts to send to replay buffer 1 by 1
        vals = ready_experiences.pop("value")
        target_vals = ready_experiences.pop("target_value")

        batch = tuple(
            dict(zip(ready_experiences, experience))
            for experience in zip(*ready_experiences.values())
        )

        return batch, vals, target_vals

    def prepare_experiences(
            self,
            experiences: Deque[Dict[str, Any]],
        ) -> Any:
        """
        Prepares the experience to add to the buffer.

        Args:
            experiences: The experiences containing rewards.

        Returns:
            The oldest stored experience.
        """
        # Use the last experience for the next Q-value to calculate the target
        # Q-values
        next_return = 0
        n_step_idx = len(experiences) - 1

        for t in reversed(range(len(experiences) - 1)):
            # "next_value" supplies the bootstrap if the environment terminated
            # early but the agent didn't
            non_terminal = (
                (1 - experiences[t]["env_terminal"])
                * (1 - experiences[t]["terminal"])
            )

            discounted_term = self.decay * non_terminal * next_return
            next_return = experiences[t]["reward"] + discounted_term

            n_step_idx = (
                n_step_idx * non_terminal + t * (1 - non_terminal)
            )

        experience = experiences[0]

        experience["reward"] = next_return
        experience["next_state"] = experiences[n_step_idx]["next_state"]
        experience["terminal"] = experiences[n_step_idx]["terminal"]
        experience["n_steps"] = n_step_idx + 1

        last_is_nonterminal = (
            (n_step_idx == len(experiences) - 1)
            * (1 - experiences[n_step_idx]["env_terminal"])
            * (1 - experiences[n_step_idx]["terminal"])
        )
        bootstrap_value = (
            experiences[-1]["value"] * last_is_nonterminal
            + experiences[n_step_idx]["next_value"] * (1 - last_is_nonterminal)
        )

        experience["target_value"] = (
            experience["reward"] + self.decay * bootstrap_value
        )

        experiences.popleft()

        return [experience]

    def train_step(
            self,
            ready_experiences: Dict[str, List[Any]],
            batch_size: int,
            experience_replay: ExperienceReplay,
            *train_args: Any,
            **train_kwargs: Any
        ) -> Dict[str, List[Any]]:
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
            The ready experiences that were not added to the replay buffer.
        """
        # Get length of a random key
        keys = list(ready_experiences)
        if len(keys) > 0:
            key = keys[0]
            if len(ready_experiences[key]) == batch_size:
                experiences_to_add, q_vals, target_q_vals = self.create_batch(
                    ready_experiences
                )

                errors = experience_replay.get_error(q_vals, target_q_vals)
                priorities = experience_replay.get_priority(errors)

                for i in range(len(experiences_to_add)):
                    experience = experiences_to_add[i]
                    priority = priorities[i]

                    experience["id"] = (self.algo.env_steps, i)

                    experience_replay.add(experience, priority.item())

                ready_experiences = {}

        self.algo.train_from_buffer(
            experience_replay, *train_args, **train_kwargs
        )

        return ready_experiences
