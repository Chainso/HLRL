if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.multiprocessing as mp
    import gym

    from argparse import ArgumentParser
    from functools import partial
    from pathlib import Path

    from hlrl.core.logger import TensorboardLogger
    from hlrl.core.common.functional import compose
    from hlrl.torch.trainers import OffPolicyTrainer
    from hlrl.core.envs.gym import GymEnv
    from hlrl.torch.algos import DQN, RND
    from hlrl.torch.policies import LinearPolicy, LSTMPolicy

    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")

    # The hyperparameters as command line arguments
    parser = ArgumentParser(
        description="DQN example on the CartPole-v1 environment."
    )

    # File args
    parser.add_argument(
        "-x", "--experiment_path", type=str,
        help="the path to save experiment results and models"
    )
    parser.add_argument(
        "--load_path", type=str,
        help="the path of the saved model to load"
    )

    # Env args
    parser.add_argument(
        "-r", "--render", dest="render", action="store_true",
        help="render the environment"
    )
    parser.add_argument(
        "-e,", "--env", dest="env", default="CartPole-v1",
        help="the gym environment to train on"
    )

    # Model args
    parser.add_argument(
		"--device", type=torch.device, default="cpu",
		help="the device (cpu/gpu) to train and play on"
	)
    parser.add_argument(
        "--hidden_size", type=int, default=256,
        help="the size of each hidden layer"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2,
        help="the number of layers before the output layers"
    )

    # Algo args
    parser.add_argument(
		"--recurrent", action="store_true",
		help="make the network recurrent (using LSTM)"
	)
    parser.add_argument(
        "--exploration", choices=["rnd", "munchausen"],
        help="The type of exploration to use"
    )
    parser.add_argument(
		"--discount", type=float, default=0.99,
		help="the next state reward discount factor"
	)
    parser.add_argument(
        "--polyak", type=float, default=5e-3,
        help="the polyak constant for the target network updates"
    )
    parser.add_argument(
        "--target_update_interval", type=int, default=1,
        help="the number of training steps in-between target network updates"
    )
    parser.add_argument(
		"--lr", type=float, default=1e-3,
		help="the learning rate"
	)

    # Training/Playing args
    parser.add_argument(
        "-p", "--play", dest="play", action="store_true",
        help="runs the environment using the model instead of training"
    )
    parser.add_argument(
		"--batch_size", type=int, default=128,
		help="the batch size of the training set"
	)
    parser.add_argument(
		"--start_size", type=int, default=256,
		help="the size of the replay buffer before training"
	)
    parser.add_argument(
		"--save_interval", type=int, default=5000,
		help="the number of batches in between saves"
	)
    parser.add_argument(
		"--episodes", type=int, default=10000,
		help="the number of episodes to play for if playing"
	)
    parser.add_argument(
        "--training_steps", type=int, default=50000,
        help="the number of training steps to train for"
    )

    # Agent args
    parser.add_argument(
		"--decay", type=float, default=0.99,
		help="the gamma decay for the target Q-values"
	)
    parser.add_argument(
		"--n_steps", type=int, default=5,
		help="the number of decay steps"
	)
    parser.add_argument(
        "--num_agents", type=int, default=0,
        help="the number of agents to run concurrently"
    )
    parser.add_argument(
        "--silent", action="store_true",
        help="will run without standard output from agents"
    )

    # Experience Replay args
    parser.add_argument(
		"--er_capacity", type=int, default=50000,
		help="the maximum amount of episodes in the replay buffer"
	)
    parser.add_argument(
		"--er_alpha", type=float, default=0.6,
		help="the alpha value for PER"
	)
    parser.add_argument(
		"--er_beta", type=float, default=0.4,
		help="the beta value for PER"
	)
    parser.add_argument(
		"--er_beta_increment", type=float, default=1e-3,
		help="the increment of the beta value on each sample for PER"
	)
    parser.add_argument(
		"--er_epsilon", type=float, default=1e-3,
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

    logs_path = None
    save_path = None

    if args.experiment_path is not None:
        logs_path = Path(args.experiment_path, "logs")
        logs_path.mkdir(parents=True, exist_ok=True)
        logs_path = str(logs_path)

        save_path = Path(args.experiment_path, "models")
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = str(save_path)

    # Initialize the environment
    args.vectorized = False
    env_builder = partial(gym.make, args.env)
    env_builder = compose(env_builder, GymEnv)
    env = env_builder()

    # The algorithm logger
    algo_logger = (
        None if logs_path is None else TensorboardLogger(logs_path + "/algo")
    )

    # Initialize IQN
    activation_fn = nn.ReLU
    optim = partial(torch.optim.Adam, lr=args.lr)

    if args.recurrent:
        num_lin_before = 1 if args.num_layers > 2 else 0
        num_lin_after = max(args.num_layers - 2, 1)

        qfunc = LSTMPolicy(
            env.state_space[0], env.action_space[0], args.hidden_size,
            num_lin_before, args.hidden_size, 1, args.hidden_size,
            num_lin_after, activation_fn
        )

        algo = DQNRecurrent(
            qfunc, args.discount, args.polyak, args.target_update_interval,
            optim, args.device, algo_logger
        )

        algo = TorchRecurrentAlgo(algo, args.burn_in_length)
    else:
        qfunc = LinearPolicy(
            env.state_space[0], env.action_space[0], args.hidden_size,
            args.num_layers, activation_fn
        )

        algo = DQN(
            qfunc, args.discount, args.polyak, args.target_update_interval,
            optim, args.device, algo_logger
        )

    if args.exploration == "rnd":
        rnd_network = LinearPolicy(
            env.state_space[0], args.hidden_size, args.hidden_size,
            args.num_layers + 2, activation_fn
        )

        rnd_target = LinearPolicy(
            env.state_space[0], args.hidden_size, args.hidden_size,
            args.num_layers, activation_fn
        )

        algo = RND(algo, rnd_network, rnd_target, optim)

    algo = algo.to(torch.device(args.device))

    if args.load_path is not None:
        algo.load(args.load_path)

    off_policy_trainer = OffPolicyTrainer()
    off_policy_trainer.train(args, env_builder, algo)