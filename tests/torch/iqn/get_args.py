from argparse import ArgumentParser

def get_args():
    # The hyperparameters as command line arguments
    parser = ArgumentParser(
        description="Rainbow-IQN tests."
    )

    # Logging
    parser.add_argument(
        "-l, --logs_path ", dest="logs_path", type=str, default=None,
        help="log training data to tensorboard using the path"
    )

    # Env args
    parser.add_argument(
        "-r", "--render", dest="render", action="store_true",
        help="render the environment"
    )

    # Model args
    parser.add_argument(
        "--hidden_size", type=int, default=512,
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
		"--discount", type=float, default=0.99,
		help="the next state reward discount factor"
	)
    parser.add_argument(
        "--polyak", type=float, default=5e-3,
        help="the polyak constant for the target network updates"
    )
    parser.add_argument(
		"--n_quantiles", type=float, default=64,
		help="the number of quantile samples for IQN"
	)
    parser.add_argument(
		"--embedding_dim", type=float, default=64,
		help="the dimension of the quantile distribution for IQN"
	)
    parser.add_argument(
		"--huber_threshold", type=float, default=1,
		help="the threshhold of the huber loss (kappa) for IQN"
	)
    parser.add_argument(
        "--target_update_interval", type=int, default=1,
        help="the number of training steps in-between target network updates"
    )
    parser.add_argument(
		"--lr", type=float, default=3e-4,
		help="the learning rate"
	)

    # Training/Playing args
    parser.add_argument(
		"--batch_size", type=int, default=32,
		help="the batch size of the training set"
	)
    parser.add_argument(
		"--start_size", type=int, default=64,
		help="the size of the replay buffer before training"
	)
    parser.add_argument(
		"--save_path", type=str, default=None,
		help="the path to save the model to"
	)
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
		"--episodes", type=int, default=10000,
		help="the number of episodes to train for"
	)
    parser.add_argument(
		"--decay", type=float, default=0.99,
		help="the gamma decay for the target Q-values"
	)
    parser.add_argument(
		"--n_steps", type=int, default=5,
		help="the number of decay steps"
	)
    parser.add_argument(
        "--verbose", action="store_true",
        help="outputs episode information to standard output"
    )

    # Experience Replay args
    parser.add_argument(
		"--er_capacity", type=float, default=50000,
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

    return args