from typing import Any, Dict

import torch

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.torch.algos.algo import TorchRLAlgo

class TorchRecurrentAlgo(MethodWrapper):
    """
    A wrapper to burn in hidden states when training a recurrent algorithm.
    """
    def __init__(self, algo: TorchRLAlgo, burn_in_length: int = 0):
        """
        Creates the recurrent algorithm wrapper on the underlying algorithm.

        Args:
            algo: The algorithm to wrap.
            burn_in_length: The number of states to burn in the hidden state.
        """
        super().__init__(algo)

        self.burn_in_length = burn_in_length

    def __reduce__(self) -> Tuple[type, Tuple[Any, ...]]:
        """
        Reduces the inputs used to serialize and recreate the recurrent agent.

        Returns:
            A tuple of the class and input arguments.
        """
        return (type(self), (self.obj, self.burn_in_length))

    def burn_in_hidden_states(self,
            rollouts: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
        """
        Burns in the hidden states of the rollouts.

        Args:
            rollouts: The training batch to burn in hidden states for.

        Returns:
            The training batch with the hidden states burned in.
        """
        rollouts = {
            key: value.to(self.device) for key, value in rollouts.items()
        }

        # Switch from (batch size, 2, num layers, hidden size) to
        # (2, num layers, batch size, hidden size)
        rollouts["hidden_state"] = (
            rollouts["hidden_state"].permute(1, 2, 0, 3).contiguous()
        )

        states = rollouts["state"]
        actions = rollouts["action"]
        rewards = rollouts["reward"]
        next_states = rollouts["next_state"]
        terminals = rollouts["terminal"]
        hidden_states = rollouts["hidden_state"]

        burn_in_states = states
        burn_in_next_states = next_states
        new_hiddens = hidden_states

        if self.burn_in_length > 0:    
            with torch.no_grad():
                burn_in_states = states[:, :self.burn_in_length].contiguous()

                *_, new_hiddens = self.forward(
                    burn_in_states, hidden_states
                )


        rollouts["hidden_state"] = [nh for nh in new_hiddens]

        rollouts["state"] = states[:, self.burn_in_length:].contiguous()
        rollouts["action"] = actions[:, self.burn_in_length:].contiguous()
        rollouts["reward"] = rewards[:, self.burn_in_length:].contiguous()
        rollouts["next_state"] = next_states[
            :, self.burn_in_length:
        ].contiguous()
        rollouts["terminal"] = terminals[:, self.burn_in_length:].contiguous()

        with torch.no_grad():
            first_burned_in = states[:, :1]
            *_, next_hiddens = self.forward(
                first_burned_in, new_hiddens
            )

        rollouts["next_hidden_state"] = [nh for nh in next_hiddens]

        return rollouts

    def train_batch(self,
                    rollouts: Dict[str, torch.Tensor],
                    *training_args: Any) -> Any:
        """
        Burns in the hidden states before training the batch.

        Args:
            rollouts: The batch to train on.
            training_args: The arguments to pass into the algorithm train batch.
        
        Returns:
            The train batch return of the wrapped algorithm.
        """
        rollouts = self.burn_in_hidden_states(rollouts)

        return self.om.train_batch(rollouts, *training_args)
