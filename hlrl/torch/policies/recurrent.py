from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .linear import LinearPolicy
from .distribution import GaussianPolicy, TanhGaussianPolicy

class LSTMPolicy(nn.Module):
    """
    A LSTM policy.
    """
    def __init__(self,
        inp_n: int,
        out_n: int,
        lin_before_hidden_size: int,
        lin_before_num_layers: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        lin_after_hidden_size: int,
        lin_after_num_layers: int,
        activation_fn: nn.Module):
        """
        Creates the LSTM policy, with a linear policy before and after it. The
        LSTM module is batch major.

        Args:
            inp_n: The number of input units.
            out_n: The number of output units.
            lin_before_hidden_size: The number of hidden units in the linear
                network before the LSTM.
            lin_before_num_layers: The number of hidden layers before the LSTM.
            lstm_hidden_size: The number of hidden units in the LSTM.
            lstm_num_layers: The number of hidden layers in the LSTM.
            lin_after_hidden_size: The number of hidden units in the linear
                network after the LSTM.
            lin_after_num_layers: The number of hidden layers after the LSTM.
            activation_fn: The activation function for each layer.
        """
        super().__init__()

        self.out_n = out_n
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        lstm_in_n = (
            lin_before_hidden_size if lin_before_num_layers > 0 else inp_n
        )

        self.lstm_out_n = (
            lin_after_hidden_size if lin_after_num_layers > 0 else out_n
        )

        self.lin_before = LinearPolicy(
            inp_n, lstm_hidden_size, lin_before_hidden_size,
            lin_before_num_layers, activation_fn
        )

        if lin_before_num_layers > 0:
            self.lin_before = nn.Sequential(self.lin_before, activation_fn())

        self.lstm = nn.LSTM(
            lstm_in_n, self.lstm_out_n, lstm_num_layers, batch_first=True
        )

        self.lin_after = LinearPolicy(
            lstm_hidden_size, out_n, lin_after_hidden_size,
            lin_after_num_layers, activation_fn
        )

    def forward(
            self,
            states: torch.Tensor,
            hidden_states: Tuple[torch.Tensor, torch.Tensor],
            lengths: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Takes in the input with the batch dimension being first and returns the
        output along with the new hidden state.

        Args:
            states: The states for the network input.
            hidden_states: The hidden states of the LSTM.
            lengths: The lengths of each sequence for a padded sequence.
        """
        # Input size is (batch size, sequence length, ...)
        # For hidden states its (batch size, ...) since going 1 step at a time
        batch_size, sequence_length = states.shape[:2]

        states = states.view(
            batch_size * sequence_length, *states.shape[2:]
        )
        lin_before = self.lin_before(states)
  
        lstm_in = lin_before.view(
            batch_size, sequence_length, *lin_before.shape[1:]
        )
        hidden_states = [hs for hs in hidden_states]

        if lengths is None:
            lstm_out, new_hiddens = self.lstm(lstm_in, hidden_states)
        else:
            packed = pack_padded_sequence(
                lstm_in, lengths, batch_first=True, enforce_sorted=False
            )

            lstm_out, new_hiddens = self.lstm(packed, hidden_states)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
    
        lstm_out = lstm_out.contiguous().view(
            batch_size * sequence_length, *lstm_out.shape[2:]
        )

        lin_after = self.lin_after(lstm_out)
        lin_after = lin_after.view(
            batch_size, sequence_length, *lin_after.shape[1:]
        )
    
        return lin_after, new_hiddens

    def reset_hidden_state(
            self,
            batch_size: int = 1
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a reset hidden state of the LSTM.

        Args:
            batch_size: The batch size of the reset hidden state.
        """
        zero_state = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_out_n
        )
        reset_hidden = (zero_state, zero_state.clone())

        return reset_hidden


class LSTMCatPolicy(LSTMPolicy):
    """
    A LSTM policy that takes multiple inputs to be concatenated.
    """
    def __init__(
            self,
            input_sizes: Tuple[int, ...],
            out_n: int,
            lin_before_hidden_size: int,
            lin_before_num_layers: int,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            lin_after_hidden_size: int,
            lin_after_num_layers: int,
            activation_fn: nn.Module
        ):
        """
        Creates the LSTM policy using multiple inputs, with a linear policy
        before and after it. The LSTM is batch major.

        Args:
            input_sizes: The sizes of the network inputs to concatenate.
            out_n: The number of output units.
            lin_before_hidden_size: The number of hidden units in the linear
                network before the LSTM.
            lin_before_num_layers: The number of hidden layers before the LSTM.
            lstm_hidden_size: The number of hidden units in the LSTM.
            lstm_num_layers: The number of hidden layers in the LSTM.
            lin_after_hidden_size: The number of hidden units in the linear
                network after the LSTM.
            lin_after_num_layers: The number of hidden layers after the LSTM.
            activation_fn: The activation function for each layer.
        """
        super().__init__(
            sum(input_sizes), out_n, lin_before_hidden_size,
            lin_before_num_layers, lstm_hidden_size, lstm_num_layers,
            lin_after_hidden_size, lin_after_num_layers, activation_fn
        )

    def forward(
            self,
            *inputs: Tuple[Union[torch.Tensor,
                                 Tuple[torch.Tensor, torch.Tensor]]],
            lengths: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Returns the output along with the new hidden states.

        Args:
            inputs: The inputs to concatenate and the hidden states of the LSTM.
            lengths: The lengths of each sequence for a padded sequence.

        Returns:
            The LSTM output on the inputs.
        """
        *inputs, hidden_states = inputs
        lin_in = torch.cat(inputs, dim=-1)
        return super().forward(lin_in, hidden_states, lengths)

class LSTMGaussianPolicy(LSTMPolicy):
    """
    A LSTM policy wth a gaussian head.
    """
    def __init__(self,
        inp_n: int,
        out_n: int,
        lin_before_hidden_size: int,
        lin_before_num_layers: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        lin_after_hidden_size: int,
        lin_after_num_layers: int,
        activation_fn: nn.Module,
        squished: Optional[bool] = True):
        """
        Creates the LSTM policy with a guassian head, with a linear policy
        before and after it. The LSTM is batch major.

        Args:
            inp_n: The number of input units.
            out_n: The number of output units.
            lin_before_hidden_size: The number of hidden units in the linear
                network before the LSTM.
            lin_before_num_layers: The number of hidden layers before the LSTM.
            lstm_hidden_size: The number of hidden units in the LSTM.
            lstm_num_layers: The number of hidden layers in the LSTM.
            lin_after_hidden_size: The number of hidden units in the linear
                network after the LSTM.
            lin_after_num_layers: The number of hidden layers after the LSTM.
            activation_fn: The activation function for each layer.
            squished: If true, uses a tanh gaussian over the regular.
        """
        super().__init__(
            inp_n, lin_after_hidden_size, lin_before_hidden_size,
            lin_before_num_layers, lstm_hidden_size, lstm_num_layers,
            lin_after_hidden_size, lin_after_num_layers, activation_fn
        )

        # Reinitializing lstm to always have the output be the hidden size
        # to feed into the gaussian network
        lstm_in_n = (
            lin_before_hidden_size if lin_before_num_layers > 0 else inp_n
        )

        self.lstm = nn.LSTM(
            lstm_in_n, lin_after_hidden_size, lstm_num_layers, batch_first=True
        )

        gaussian_args = (
            lin_after_hidden_size, out_n, lin_after_hidden_size, 1,
            activation_fn
        )

        if squished:
            self.gaussian = TanhGaussianPolicy(*gaussian_args)
        else:
            self.gaussian = GaussianPolicy(*gaussian_args)

    def forward(
            self,
            states: torch.Tensor,
            hidden_states: Tuple[torch.Tensor, torch.Tensor],
            lengths: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Returns the output along with the new hidden states.

        Args:
            states: The states for the network input.
            hidden_states: The hidden states of the LSTM.
            lengths: The lengths of each sequence for a padded sequence.
        """
        batch_size, sequence_length = states.shape[:2]
        gauss_in, new_hidden = super().forward(states, hidden_states, lengths)

        gauss_in = gauss_in.view(
            batch_size * sequence_length, *gauss_in.shape[2:]
        )
        action, log_prob, mean = self.gaussian(gauss_in)

        action = action.view(
            batch_size, sequence_length, *action.shape[1:]
        )
        log_prob = log_prob.view(
            batch_size, sequence_length, *log_prob.shape[1:]
        )
        mean = mean.view(
            batch_size, sequence_length, *mean.shape[1:]
        )

        return action, log_prob, mean, new_hidden
