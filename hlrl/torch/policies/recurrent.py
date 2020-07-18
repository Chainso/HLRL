import torch
import torch.nn as nn

from .linear import LinearPolicy
from .gaussian import GaussianPolicy, TanhGaussianPolicy

class LSTMPolicy(nn.Module):
    """
    A LSTM policy.
    """
    def __init__(self, input_size, output_size, b_hidden_size, b_num_hidden,
                 l_hidden_size, l_num_hidden, a_hidden_size, a_num_hidden,
                 activation_fn):
        """
        Args:
            input_size (int): The number of input units.
            output_size (int): The number of output units.
            b_hidden_size (int): The number of hidden units in the linear
                                 network before the LSTM.
            b_num_hidden (int): The number of hidden layers before the LSTM.
            b_hidden_size (int): The number of hidden units in the LSTM.
            b_num_hidden (int): The number of hidden layers in the LSTM.
            a_hidden_size (int): The number of hidden units in the linear
                                 network after the LSTM.
            a_num_hidden (int): The number of hidden layers after the LSTM.
            activation_fn (callable): The activation function for each layer.
        """
        super().__init__()

        self.lstm_layers = l_num_hidden

        if b_num_hidden == 0:
            self.lstm_inp = input_size 
        else:
            self.lstm_inp = b_hidden_size

        if a_num_hidden == 0:
            self.lstm_out = output_size
        else:
            self.lstm_out = l_hidden_size

        self.lstm = nn.LSTM(self.lstm_inp, self.lstm_out, l_num_hidden)

        self.lin_before = LinearPolicy(input_size, b_hidden_size, b_hidden_size,
                                       b_num_hidden - 1, activation_fn)

        if b_num_hidden > 0:
            self.lin_before = nn.Sequential(
                self.lin_before,
                activation_fn()
            )

        self.lin_after = LinearPolicy(l_hidden_size, output_size, a_hidden_size,
                                      a_num_hidden - 1, activation_fn)

    def forward(self, states, hidden_states):
        """
        Returns the output along with the new hidden state.
        """
        # Input size is (batch size, sequence length, ...)
        # For hidden states its (batch size, ...) since going 1 step at a time
        batch_size, sequence_length = states.shape[:2]

        states = states.contiguous().view(batch_size * sequence_length,
                                          *states.shape[2:])

        lin_before = self.lin_before(states)

        lstm_in = lstm_in.view(
            sequence_length, batch_size, *lin_before.shape[1:]
        )

        # Switch from (batch size, num layers, hidden size) to
        # (num layers, batch size, hidden size)
        hidden_states = [tens.transpose(1, 0) for tens in hidden_states]

        lstm_out, new_hiddens = self.lstm(lstm_in, hidden_states)

        # Back to batch major
        new_hiddens = [tens.transpose(1, 0) for tens in hidden_states]

        lin_after_in = lstm_out.view(batch_size * sequence_length,
                                     *lstm_out.shape[2:])

        lin_after = self.lin_after(lin_after_in)
        lin_after = lin_after.view(batch_size, sequence_length,
                                   *lin_after.shape[1:])
    
        return lin_after, new_hiddens

    def reset_hidden_state(self, batch_size=1, batch_major=True):
        """
        Returns a reset hidden state of the LSTM.
        """
        if batch_major:
            zero_state = torch.zeros(batch_size, self.lstm_layers,
                                     self.lstm_out)
        else:
            zero_state = torch.zeros(self.lstm_layers, batch_size,
                                     self.lstm_out)

        reset_hidden = (zero_state, zero_state)
        return reset_hidden

    def forward(self, states, current_actions, hidden_states):
        """
        Returns the output along with the new hidden states.
        """
        lin_in = torch.cat([states, current_actions], dim=-1)
        return super().forward(lin_in, hidden_states)

class LSTMGaussianPolicy(LSTMPolicy):
    """
    A LSTM Gaussian policy (same as LSTM policy but with a Gaussian head on top)
    """
    def __init__(self, input_size, output_size, b_hidden_size,
                 b_num_hidden, l_hidden_size, l_num_hidden, a_hidden_size,
                 a_num_hidden, activation_fn, squished=False):
        """
        Args:
            input_size (int): The number of input units.
            output_size (int): The number of output units.
            b_hidden_size (int): The number of hidden units in the linear
                                 network before the LSTM.
            b_num_hidden (int): The number of hidden layers before the LSTM.
            b_hidden_size (int): The number of hidden units in the LSTM.
            b_num_hidden (int): The number of hidden layers in the LSTM.
            a_hidden_size (int): The number of hidden units in the linear
                                 network after the LSTM.
            a_num_hidden (int): The number of hidden layers after the LSTM.
            activation_fn (callable): The activation function for each layer.
            squished (bool): If the Gaussian policy should be a squished
                             Gaussian (to [-1, 1] by tanh).
        """
        if a_num_hidden == 0:
            gauss_in = l_hidden_size
        else:
            gauss_in = a_hidden_size

        super().__init__(input_size, gauss_in, b_hidden_size, b_num_hidden,
                         l_hidden_size, l_num_hidden, a_hidden_size,
                         a_num_hidden, activation_fn)

        if squished:
            self.gaussian = TanhGaussianPolicy(a_hidden_size, output_size, 0, 0,
                                               activation_fn)
        else:
            self.gaussian = GaussianPolicy(gauss_in, output_size, 0, 0,
                                           activation_fn)

    def forward(self, states, hidden_states):
        """
        Returns the mean and log standard deviation along with the new hidden
        states.
        """
        gauss_in, new_hidden = super().forward(states, hidden_states)
        mean, log_std = self.gaussian(gauss_in)

        return mean, log_std, new_hidden

    def sample(self, states, hidden_states, epsilon=1e-4):
        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample and the new hidden states.
        """
        batch_size, sequence_length = states.shape[:2]

        gauss_in, new_hidden = super().forward(states, lhidden_states)

        gauss_in = gauss_in.contiguous().view(batch_size * sequence_length,
                                              *gauss_in.shape[2:])
        action, log_prob, mean = self.gaussian.sample(gauss_in, epsilon)

        action = action.contiguous().view(batch_size, sequence_length,
                                          *action.shape[1:])
        log_prob = log_prob.contiguous().view(batch_size, sequence_length,
                                              *log_prob.shape[1:])
        mean = mean.contiguous().view(batch_size, sequence_length,
                                      *mean.shape[1:])

        return action, log_prob, mean, new_hidden