import yaml
from functools import partial
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

from hlrl.core.logger import TensorboardLogger
from hlrl.core.agents import OffPolicyAgent
from hlrl.torch.algos import (
    RainbowIQN, RainbowIQNRecurrent, TorchRecurrentAlgo, RND
)
from hlrl.torch.agents import (
    TorchRLAgent, SequenceInputAgent, ExperienceSequenceAgent,
    TorchRecurrentAgent
)
from hlrl.torch.experience_replay import TorchPER, TorchPSER, TorchR2D2
from hlrl.torch.policies import LinearPolicy, LSTMPolicy

def setup_test(args, env):
    # The logs path
    logs_path = None
    save_path = None

    if args.experiment_path is not None:
        logs_path = Path(args.experiment_path, "logs")
        logs_path.mkdir(parents=True, exist_ok=True)
        logs_path = str(logs_path)

        save_path = Path(args.experiment_path, "models")
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = str(save_path)

        with open(Path(args.experiment_path, "config.yml"), "w") as config_file:
            yaml.dump(vars(args), config_file)
    
    # The algorithm logger
    algo_logger = (
        None if logs_path is None else TensorboardLogger(logs_path + "/algo")
    )

    # Initialize IQN
    activation_fn = nn.ReLU
    optim = partial(torch.optim.Adam, lr=args.lr)

    # Setup networks
    if args.num_layers > 1:
        qfunc_num_layers = 1
        autoencoder_out_n = args.hidden_size
    else:
        qfunc_num_layers = 0
        autoencoder_out_n = env.action_space[0]

    qfunc = LinearPolicy(
        args.hidden_size, env.action_space[0], args.hidden_size,
        qfunc_num_layers, activation_fn
    )

    if args.recurrent:
        num_lin_before = 1 if args.num_layers > 2 else 0

        autoencoder = LSTMPolicy(
            env.state_space[0], autoencoder_out_n, args.hidden_size,
            num_lin_before, args.hidden_size, max(args.num_layers - 2, 1), 0, 0,
            activation_fn
        )

        algo = RainbowIQNRecurrent(
            autoencoder_out_n, autoencoder, qfunc,
            args.discount, args.polyak, args.n_quantiles, args.embedding_dim,
            args.huber_threshold, args.target_update_interval, optim, optim,
            args.device, algo_logger
        )

        algo = TorchRecurrentAlgo(algo, args.burn_in_length)
    else:
        autoencoder = LinearPolicy(
            env.state_space[0], autoencoder_out_n, args.hidden_size,
            min(args.num_layers - 1, 1), activation_fn
        )

        if args.num_layers > 1:
            autoencoder = nn.Sequential(autoencoder, activation_fn())

        algo = RainbowIQN(
            args.hidden_size, autoencoder, qfunc, args.discount, args.polyak,
            args.n_quantiles, args.embedding_dim, args.huber_threshold,
            args.target_update_interval, optim, optim, args.device, algo_logger
        )

    if args.normalize_return:
        algo = NormalizeReturnAlgo(algo)

    if args.load_path is not None:
        algo.load(args.load_path)

    return algo
