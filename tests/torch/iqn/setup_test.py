import torch
import torch.nn as nn

from functools import partial

from hlrl.core.logger import TensorboardLogger
from hlrl.core.agents import AgentPool, OffPolicyAgent
from hlrl.torch.algos import RainbowIQN
from hlrl.torch.agents import (
    TorchRLAgent, SequenceInputAgent, ExperienceSequenceAgent,
    TorchRecurrentAgent
)
from hlrl.torch.experience_replay import TorchPER, TorchPSER, TorchR2D2
from hlrl.torch.policies import LinearPolicy, LSTMPolicy

def setup_test(args, env):
    # The logger
    logger = args.logs_path
    logger = None if logger is None else TensorboardLogger(logger)
    
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
            num_lin_before, args.hidden_size, max(args.num_layers - 2, 1),
            0, 0, activation_fn
        )
        # TODO CREATE RECURRENT RAINBOW IQN INSTANCE HERE ONCE IMPLEMENTED
    else:
        autoencoder = LinearPolicy(
            env.state_space[0], autoencoder_out_n, args.hidden_size,
            min(args.num_layers - 1, 1), activation_fn
        )

        if args.num_layers > 1:
            autoencoder = nn.Sequential(autoencoder, activation_fn())

        algo = RainbowIQN(
            args.hidden_size, autoencoder, qfunc, args.discount,
            args.polyak, args.n_quantiles, args.embedding_dim,
            args.huber_threshold, args.target_update_interval, optim,
            optim, logger
        )

    algo = algo.to(torch.device(args.device))

    if args.load_path is not None:
        algo.load(args.load_path)

    # Create agent class
    agent = OffPolicyAgent(
        env, algo, args.render, silent=not args.verbose, logger=logger
    )

    if args.recurrent:
        agent = SequenceInputAgent(agent, device=args.device)
        agent = TorchRecurrentAgent(agent)
    else:
        agent = TorchRLAgent(agent, batch_state=False, device=args.device)

    algo.create_optimizers()

    algo.train()
    algo.share_memory()
    # Experience replay
    if args.recurrent:
        experience_replay = TorchR2D2(
            args.er_capacity, args.er_alpha, args.er_beta,
            args.er_beta_increment, args.er_epsilon,
            args.max_factor
        )
        agent = ExperienceSequenceAgent(
            agent, args.burn_in_length + args.sequence_length,
            args.burn_in_length
        )
    else:
        experience_replay = TorchPER(
            args.er_capacity, args.er_alpha, args.er_beta,
            args.er_beta_increment, args.er_epsilon
        )

    agents = [agent]

    agent_pool = AgentPool(agents)

    return algo, agent_pool, env, experience_replay