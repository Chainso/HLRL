

if(__name__ == "__main__"):
    import torch
    import torch.nn as nn
    import torch.multiprocessing as mp
    import gym
    
    from gym.wrappers import RescaleAction
    from argparse import ArgumentParser
    from functools import partial

    from hlrl.core.logger import TensorboardLogger
    from hlrl.core.common.functional import compose
    from hlrl.core.distributed import ApexLearner, ApexWorker
    from hlrl.core.envs.gym import GymEnv
    from hlrl.core.agents import AgentPool, OffPolicyAgent, IntrinsicRewardAgent
    from hlrl.torch.algos import SAC, SACRecurrent, RND
    from hlrl.torch.agents import (
        TorchRLAgent, SequenceInputAgent, ExperienceSequenceAgent,
        TorchRecurrentAgent
    )
    from hlrl.torch.policies import (
        LinearPolicy, LinearSAPolicy, TanhGaussianPolicy, LSTMPolicy,
        LSTMSAPolicy, LSTMGaussianPolicy
    )
    from hlrl.torch.experience_replay import TorchPER, TorchPSER, TorchR2D2

    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")

    # The hyperparameters as command line arguments
    parser = ArgumentParser(
        description="Twin Q-Function SAC example on the Pendulum-v0 "
            + "environment."
    )

    # Logging
    parser.add_argument(
        "-l, --logs_path ", dest="logs_path", type=str,
        help="log training data to tensorboard using the path"
    )

    # Env args
    parser.add_argument(
        "-r, --render", dest="render", action="store_true",
        help="render the environment"
    )
    parser.add_argument(
        "-e,", "--env", dest="env", default="Pendulum-v0",
        help="the gym environment to train on"
    )

    # Model args
    parser.add_argument(
        "--hidden_size", type=int, default=256,
        help="the size of each hidden layer"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3,
        help="the number of layers before the output layers"
    )

    # Algo args
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="the device (cpu/gpu) to train and play on"
    )
    parser.add_argument(
        "--recurrent", action="store_true",
        help="make the network recurrent (using LSTM)"
    )
    parser.add_argument(
        "--exploration", default=None, choices=["rnd"],
        help="The type of exploration to use [rnd]"
    )
    parser.add_argument(
        "--discount", type=float, default=0.99,
        help="the next state reward discount factor"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="the coefficient of temperature of the entropy for SAC"
    )
    parser.add_argument(
        "--polyak", type=float, default=5e-3,
        help="the polyak constant for the target network updates"
    )
    parser.add_argument(
        "--target_update_interval", type=int, default=1,
        help="the number of training steps inbetween target network updates"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="the learning rate"
    )
    parser.add_argument(
        "--twin", type=bool, default=True,
        help="true if SAC should use twin Q-networks"
    )

    # Training/Playing args
    parser.add_argument(
        "-p", "--play", dest="play", action="store_true",
        help="runs the environment using the model instead of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="the batch size of the training set"
    )
    parser.add_argument(
        "--start_size", type=int, default=512,
        help="the size of the replay buffer before training"
    )
    parser.add_argument(
        "--save_path", type=str, default=None,
        help="the path to save the model to")
    parser.add_argument(
        "--load_path", type=str, default=None,
        help="the path of the saved model to load"
    )
    parser.add_argument(
        "--save_interval", type=int, default=500,
        help="the number of batches in between saves"
    )

    # Agent args
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="the number of episodes to train for"
    )
    parser.add_argument(
        "--decay", type=float, default=0.99,
        help="the gamma decay for the target Q-values"
    )
    parser.add_argument(
        "--n_steps", type=int, default=1, help="the number of decay steps"
    )
    parser.add_argument(
        "--num_agents", type=int, default=1,
        help="the number of agents to run concurrently"
    )

    # Experience Replay args
    parser.add_argument(
        "--er_capacity", type=float, default=50000,
        help="the alpha value for PER"
    )
    parser.add_argument(
        "--er_alpha", type=float, default=0.6, help="the alpha value for PER"
    )
    parser.add_argument(
        "--er_beta", type=float, default=0.4, help="the alpha value for PER"
    )
    parser.add_argument(
        "--er_beta_increment", type=float, default=1e-3,
        help="the alpha value for PER"
    )
    parser.add_argument(
        "--er_epsilon", type=float, default=1e-2,
        help="the epsilon value for PER"
    )
    parser.add_argument(
        "--burn_in_length", type=int, default=5,
        help="if recurrent, the number of burn in samples for R2D2"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=5,
        help="if recurrent, the length of the sequence to train on"
    )
    parser.add_argument(
        "--max_factor", type=int, default=0.9,
        help="if recurrent, factor of max priority to mean priority for R2D2"
    )

    args = parser.parse_args()

    # Initialize the environment, and rescale for Tanh policy
    gym_env = gym.make(args.env)
    gym_env = RescaleAction(gym_env, -1, 1)
    env = GymEnv(gym_env)

    # The logger
    logger = args.logs_path
    logger = None if logger is None else TensorboardLogger(logger)
    
    # Initialize SAC
    activation_fn = nn.ReLU
    optim = partial(torch.optim.Adam, lr=args.lr)

    # Setup networks
    if args.recurrent:
        num_lin_before = 1 if args.num_layers > 2 else 0
        num_lin_after = 1 if args.num_layers > 1 else 0

        qfunc = LSTMSAPolicy(
            env.state_space[0], env.action_space[0], 1, args.hidden_size,
            num_lin_before, args.hidden_size, max(args.num_layers - 2, 1),
            args.hidden_size, num_lin_after, activation_fn
        )

        policy = LSTMGaussianPolicy(
            env.state_space[0], env.action_space[0], args.hidden_size,
            num_lin_before, args.hidden_size, max(args.num_layers - 2, 1),
            args.hidden_size, num_lin_after, activation_fn
        )

        algo = SACRecurrent(
            env.action_space, qfunc, policy, args.discount, args.polyak,
            args.target_update_interval, optim, optim, optim, args.twin,
            args.burn_in_length, logger
        )
    else:
        qfunc = LinearSAPolicy(
            env.state_space[0], env.action_space[0], 1, args.hidden_size,
            args.num_layers, activation_fn
        )

        policy = TanhGaussianPolicy(
            env.state_space[0], env.action_space[0], args.hidden_size,
            args.num_layers, activation_fn
        )

        algo = SAC(
            env.action_space, qfunc, policy, args.discount, args.polyak,
            args.target_update_interval, optim, optim, optim, args.twin,
            logger
        )

    if args.exploration == "rnd":
        rnd_network = LinearPolicy(
            env.state_space[0], args.hidden_size, args.hidden_size,
            args.num_layers, activation_fn
        )

        rnd_target = LinearPolicy(
            env.state_space[0], args.hidden_size, args.hidden_size,
            args.num_layers + 2, activation_fn
        )

        algo = RND(algo, rnd_network, rnd_target, optim)

    algo = algo.to(torch.device(args.device))

    if args.load_path is not None:
        algo.load(args.load_path)

    # Create agent class
    agent_builder = partial(
        OffPolicyAgent, env, algo, args.render, logger=logger
    )

    if args.recurrent:
        agent_builder = compose([
            agent_builder, partial(SequenceInputAgent, device=args.device),
            TorchRecurrentAgent
        ])
    else:
        agent_builder = compose([
            agent_builder, partial(TorchRLAgent, device=args.device)
        ])

    if args.play:
        algo.eval()

        agent = agent_builder()
        agent.play(args.episodes)
    else:
        if args.exploration == "rnd":
            agent_builder = compose([agent_builder, IntrinsicRewardAgent])

        algo.create_optimizers()

        algo.train()
        algo.share_memory()

        # Experience replay
        if args.recurrent:
            experience_replay = TorchR2D2(
                args.er_capacity, args.er_alpha, args.er_beta,
                args.er_beta_increment, args.er_epsilon, args.max_factor
            )

            agent_builder = compose([
                agent_builder,
                partial(
                    ExperienceSequenceAgent,
                    sequence_length=args.burn_in_length + args.sequence_length,
                    keep_length=args.burn_in_length
                )
            ])
        else:
            experience_replay = TorchPER(
                args.er_capacity, args.er_alpha, args.er_beta,
                args.er_beta_increment, args.er_epsilon
            )

        done_event = mp.Event()

        agent_queue = mp.Queue()
        sample_queue = mp.Queue()
        priority_queue = mp.Queue()

        # Start the learner
        learner = ApexLearner()

        learner_proc = mp.Process(
            target=learner.train,
            args=(
                algo, done_event, sample_queue, priority_queue, args.save_path,
                args.save_interval
            )
        )

        # Start the worker for the model
        worker = ApexWorker()
        worker_proc = mp.Process(
            target=worker.train,
            args=(
                experience_replay, done_event, agent_queue, sample_queue,
                priority_queue, args.batch_size, args.start_size,
            )
        )

        # Start the agents
        agents = [agent_builder() for _ in range(args.num_agents)]

        # Start all processes
        learner_proc.start()
        worker_proc.start()

        agent_pool = AgentPool(agents)
        agent_procs = agent_pool.train_process(
            args.episodes, args.decay, args.n_steps, agent_queue, done_event
        )

        # Wait for processes to end
        # TODO MAKE SURE TO SET DONE_EVENT IN ONE OF THE PROCESS PROPERLY
        learner_proc.join()
        worker_proc.join()

        for agent_proc in agent_procs:
            agent_proc.join()