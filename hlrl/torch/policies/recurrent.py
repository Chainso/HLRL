import torch.nn as nn

from .linear import LinearPolicy
from .gaussian import GaussianPolicy

class LSTMPolicy(nn.Module):
    """
    A LSTM policy.
    """
    def __init__(self, input_size, action_n, output_size, b_hidden_size,
                 b_num_hidden, l_hidden_size, l_num_hidden, a_hidden_size,
                 a_num_hidden, activation_fn):
        """
        Args:
            input_size (int): The number of input units.
            action_n (int): The number of actions (appended to LSTM input).
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

        if b_num_hidden == 0:
            if a_num_hidden == 0:
                self.lstm = nn.LSTM(input_size + action_n, output_size)
            else:
                self.lstm = nn.LSTM(input_size + action_n, l_hidden_size)
        else:
            if a_num_hidden == 0:
                self.lstm = nn.LSTM(b_hidden_size + action_n, output_size)
            else:
                self.lstm = nn.LSTM(b_hidden_size + action_n, l_hidden_size)

        self.lin_before = LinearPolicy(input_size, b_hidden_size, b_hidden_size,
                                       b_num_hidden - 1, activation_fn)

        if b_num_hidden > 0:
            self.lin_before = nn.Sequential(
                *self.lin_before,
                activation_fn()
            )

        self.lin_after = LinearPolicy(l_hidden_size, output_size, a_hidden_size,
                                      a_num_hidden - 1, activation_fn)

    def forward(self, states, last_actions, hidden_states):
        """
        Returns the output along with the new hidden states.
        """
        lin_before = self.lin_before(states)

        # (sequence length, batch_size, input size)
        lstm_in = torch.cat([lin_before, last_actions], dim=-1)
        lstm_in = lin_before.permute(1, 0, 2)

        lstm_out, new_hidden = self.lstm(lstm_in, hidden_states)

        after_in = lstm_out.permute(1, 0, 2)
        out = self.lin_after(after_in)

        return out, new_hidden

class LSTMSAPolicy(nn.Module):
    """
    A LSTM policy that takes state-action inputs.
    """
    def __init__(self, input_size, action_n, output_size, b_hidden_size,
                 b_num_hidden, l_hidden_size, l_num_hidden, a_hidden_size,
                 a_num_hidden, activation_fn):
        """
        Args:
            input_size (int): The number of input units.
            action_n (int): The number of actions (appended to LSTM input).
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
        super().__init__(input_size, action_n, output_size, b_hidden_size,
                         b_num_hidden, l_hidden_size, l_num_hidden,
                         a_hidden_size, a_num_hidden, activation_fn)

    def forward(self, states, current_actions, last_actions, hidden_states):
        """
        Returns the output along with the new hidden states.
        """
        lin_in = torch.cat([state, current_actions], dim=-1)
        return super().forward(lin_in, last_actions, hidden_states)

class LSTMGaussianPolicy(LSTMPolicy):
    """
    A LSTM Gaussian policy (same as LSTM policy but with a Gaussian head on top)
    """
    def __init__(self, input_size, action_n, output_size, b_hidden_size,
                 b_num_hidden, l_hidden_size, l_num_hidden, a_hidden_size,
                 a_num_hidden, activation_fn, squished=False):
        """
        Args:
            input_size (int): The number of input units.
            action_n (int): The number of actions (appended to LSTM input).
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
        super().__init__(input_size, action_n, output_size, b_hidden_size,
                         b_num_hidden, l_hidden_size, l_num_hidden,
                         a_hidden_size, a_num_hidden, activation_fn)

        if squished:
            self.gaussian = TanhGaussianPolicy(l_hidden_size, output_size, 0, 0,
                                               activation_fn)
        else:
            self.gaussian = GaussianPolicy(l_hidden_size, output_size, 0, 0,
                                        activation_fn)

    def forward(self, states, last_actions, hidden_states):
        """
        Returns the mean and log standard deviation along with the new hidden
        states.
        """
        gauss_in, new_hidden = super().forward(states, last_actions,
                                               hidden_states)
        mean, log_std = self.gaussian(gauss_in)

        return mean, log_std, new_hidden

    def sample(self, states, epsilon=1e-4):
        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample and the new hidden states.
        """
        gauss_in, new_hidden = super().forward(states, last_actions,
                                               hidden_states)
        action, log_prob, mean = self.gaussian.sample(gauss_in)

        return action, log_prob, mean, new_hidden