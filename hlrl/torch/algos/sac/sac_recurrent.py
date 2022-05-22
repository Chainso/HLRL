from typing import Any, Dict, Tuple, Union

import torch

from .sac import SAC

class SACRecurrent(SAC):
    """
    Soft-Actor-Critic with a recurrent network.
    """
    def forward(
            self,
            observation: torch.Tensor,
            hidden_state: Tuple[torch.Tensor, torch.Tensor],
            *recurrent_args: Any,
            **recurrent_kwargs: Any
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the model output for a batch of observations

        Args:
            observation: A batch of observations from the environment.
            hidden_state: The hidden state.
            recurrent_args: Any addition positional arguments for the recurrent
                network.
            recurrent_kwargs: Any additional keyword arguments for the recurrent
                network.

        Returns:
            The action, Q-val, and new hidden state.
        """
        # Only going to update the hidden state using the policy hidden state
        action, _, _, new_hidden = self.policy(
            observation, hidden_state, *recurrent_args, **recurrent_kwargs
        )

        q_val, _ = self.q_func1(
            observation, action, hidden_state, *recurrent_args,
            **recurrent_kwargs
        )

        return action, q_val, new_hidden

    def step(
            self,
            observation: torch.Tensor,
            hidden_state: Tuple[torch.Tensor, torch.Tensor],
            *recurrent_args: Any,
            **recurrent_kwargs: Any
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the model action for a single observation of the environment.

        Args:
            observation: A single observation from the environment.
            hidden_state: The hidden state.
            recurrent_args: Any addition positional arguments for the recurrent
                network.
            recurrent_kwargs: Any additional keyword arguments for the recurrent
                network.

        Returns:
            The action, Q-value of the action and hidden state.
        """
        with torch.no_grad():
            action, q_val, new_hidden = self(
                observation, hidden_state, *recurrent_args, **recurrent_kwargs
            )

        new_hidden = [nh for nh in new_hidden]

        return action, q_val, new_hidden

    def reset_hidden_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resets the hidden state for the network.

        Returns:
            The default hidden state of the network.
        """
        return [
            tens.to(self.device)
            for tens in self.policy.reset_hidden_state()
        ]

    def get_critic_loss(
            self,
            rollouts: Dict[str, torch.Tensor]
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Calculates the loss for the Q-function/functions.

        Args:
            rollouts: The (s, a, r, s', t) of training data for the network.

        Returns:
            The batch-wise loss for the Q-function/functions.
        """
        states = rollouts["state"]
        actions = rollouts["action"]
        rewards = rollouts["reward"]
        next_states = rollouts["next_state"]
        terminals = rollouts["terminal"]
        n_steps = rollouts["n_steps"]
        hidden_states = rollouts["hidden_state"]
        next_hiddens = rollouts["next_hidden_state"]
        sequence_lengths = rollouts["sequence_length"]
        sequence_mask = rollouts["sequence_mask"]

        with torch.no_grad():
            next_actions, next_log_probs, _, _ = self.policy(
                next_states, next_hiddens, lengths=sequence_lengths
            )
            next_log_probs = next_log_probs.sum(-1, keepdim=True)

            q_targ_pred, _ = self.q_func_targ1(
                next_states, next_actions, next_hiddens,
                lengths=sequence_lengths
            )

            if self.twin:
                q_targ_pred2, _ = self.q_func_targ2(
                    next_states, next_actions, next_hiddens,
                    lengths=sequence_lengths
                )
                q_targ_pred = torch.min(q_targ_pred, q_targ_pred2)

            q_targ = q_targ_pred - self.temperature * next_log_probs
            q_next = (
                rewards + (1 - terminals) * (self._discount ** n_steps) * q_targ
            )

        q_pred, _ = self.q_func1(
            states, actions, hidden_states, lengths=sequence_lengths
        )

        q_loss = self.q_loss_func(q_pred, q_next)
        q_loss = q_loss * sequence_mask

        if self.twin:
            q_pred2, _ = self.q_func2(
                states, actions, hidden_states, lengths=sequence_lengths
            )

            q_loss2 = self.q_loss_func(q_pred2, q_next)
            q_loss = q_loss * sequence_mask

            q_loss = (q_loss, q_loss2)
    
        return q_loss
        
    def get_actor_loss(
            self,
            rollouts: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the loss for the actor/policy.

        Args:
            rollouts: The (s, a, r, s', t) of training data for the network.

        Returns:
            The batch-wise loss for the actor/policy and the log probability of
            a sampled action on the current policy.
        """
        states = rollouts["state"]
        hidden_states = rollouts["hidden_state"]
        sequence_lengths = rollouts["sequence_length"]
        sequence_mask = rollouts["sequence_mask"]

        pred_actions, pred_log_probs, _, _ = self.policy(
            states, hidden_states, lengths=sequence_lengths
        )

        pred_log_probs = pred_log_probs.sum(-1, keepdim=True)
        pred_log_probs = pred_log_probs * sequence_mask
        
        p_q_pred, _ = self.q_func1(
            states, pred_actions, hidden_states, lengths=sequence_lengths
        )

        if self.twin:
            p_q_pred2, _ = self.q_func2(
                states, pred_actions, hidden_states, lengths=sequence_lengths
            )
            p_q_pred = torch.min(p_q_pred, p_q_pred2)

        p_loss = self.temperature * pred_log_probs - p_q_pred
        p_loss = p_loss * sequence_mask

        return p_loss, pred_log_probs
