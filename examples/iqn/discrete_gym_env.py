import torch
import torch.nn as nn

from hlrl.torch.policies import LinearPolicy, LSTMPolicy

def train(args, algo, experience_replay, experience_queue, agent_procs):
    done_count = 0
    while any(proc.is_alive() for proc in agent_procs):
        if done_count == len(agent_procs):
            # Wait on the train processes
            for proc in agent_procs:
                proc.join()
        else:
            experience = experience_queue.get()

            if experience is None:
                print("Done!")
                done_count += 1
            else:
                experience_replay.add(experience)
                algo.train_from_buffer(
                    experience_replay, args["batch_size"], args["start_size"],
                    args["save_path"], args["save_interval"]
                )

def play(args, agent):
    agent.play(args["episodes"])

if(__name__ == "__main__"):
    import torch.nn as nn
    import torch.multiprocessing as mp
    import gym
    
    from argparse import ArgumentParser

    from hlrl.core.logger import TensorboardLogger
    from hlrl.torch.algos import RainbowIQN
    from hlrl.core.envs import GymEnv
    from hlrl.core.agents import AgentPool, RecurrentAgent
    from hlrl.torch.agents import OffPolicyAgent, SequenceInputAgent, \
                                  ExperienceSequenceAgent
    from hlrl.torch.experience_replay import TorchPER, TorchPSER, TorchR2D2
    from hlrl.core.trainers import Worker

    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")

    # The hyperparameters as command line arguments
    parser = ArgumentParser(
        description="Rainbow-IQN example on the CartPole-v1 environment."
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
    parser.add_argument(
        "-e,", "--env", dest="env", default="CartPole-v1",
        help="the gym environment to train on"
    )

    # Model args
    parser.add_argument(
        "--hidden_size", type=int, default=512,
        help="the size of each hidden layer"
    )
    parser.add_argument(
        "--num_hidden", type=int, default=2,
        help="the number of hidden layers before the output layers"
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
        "--target_update_interval", type=float, default=500,
        help="the number of training steps in-between target network updates"
    )
    parser.add_argument(
		"--lr", type=float, default=3e-4,
		help="the learning rate"
	)

    # Training/Playing args
    parser.add_argument(
        "-p", "--play", dest="play", action="store_true",
        help="runs the environment using the model instead of training"
    )
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

    args = vars(parser.parse_args())

    # Initialize the environment, and rescale for Tanh policy
    gym_env = gym.make(args["env"])
    env = GymEnv(gym_env)

    # The logger
    logger = args["logs_path"]
    logger = None if logger is None else TensorboardLogger(logger)
    
    # Initialize SAC
    activation_fn = nn.ReLU
    optim = lambda params: torch.optim.Adam(params, lr=args["lr"])

    # Setup networks
    b_num_hidden = 1 if args["num_hidden"] > 0 else 0

    qfunc = LinearPolicy(
        args["hidden_size"], env.action_space[0], args["hidden_size"], 1,
        activation_fn
    )

    if args["recurrent"]:
        autoencoder = LSTMPolicy(
            env.state_space[0], args["hidden_size"], 1, args["hidden_size"],
            b_num_hidden, args["hidden_size"], 1, args["hidden_size"],
            args["num_hidden"] - 1, activation_fn
        )
        """ Commented out until rainbow iqn recurrent implemented
        algo = SACRecurrent(env.action_space, qfunc, policy, args["discount"],
                            args["polyak"], args["target_update_interval"],
                            optim, optim, optim, args["twin"],
                            args["burn_in_length"],
                            logger).to(torch.device(args["device"]))"""
    else:
        autoencoder = nn.Sequential(
            LinearPolicy(
                env.state_space[0], args["hidden_size"], args["hidden_size"],
                args["num_hidden"] - 1, activation_fn
            ),
            nn.ReLU()
        )

        algo = RainbowIQN(
            args["hidden_size"], autoencoder, qfunc, args["discount"],
            args["polyak"], args["embedding_dim"], args["n_quantiles"],
            args["huber_threshold"], args["target_update_interval"], optim,
            optim, logger
        )

    algo = algo.to(torch.device(args["device"]))

    if args["load_path"] is not None:
        algo.load(args["load_path"])

    # Create agent class
    agent = OffPolicyAgent(
        env, algo, args["render"], logger=logger, device=args["device"]
    )

    if args["recurrent"]:
        agent = SequenceInputAgent(agent)
        agent = RecurrentAgent(agent)

    if args["play"]:
        algo.eval()
        play(args, agent)
    else:
        algo.train()
        algo.share_memory()

        experience_queue = mp.Queue()

        # Experience replay
        if args["recurrent"]:
            experience_replay = TorchR2D2(
                args["er_capacity"], args["er_alpha"], args["er_beta"],
                args["er_beta_increment"], args["er_epsilon"],
                args["max_factor"]
            )
            agent = ExperienceSequenceAgent(
                agent, args["burn_in_length"] + args["sequence_length"],
                args["burn_in_length"]
            )
        else:
            experience_replay = TorchPER(
                args["er_capacity"], args["er_alpha"], args["er_beta"],
                args["er_beta_increment"], args["er_epsilon"]
            )

        agents = [agent]

        agent_pool = AgentPool(agents)
        agent_procs = agent_pool.train(
            args["episodes"], experience_queue, args["decay"], args["n_steps"]
        )

        # Start the worker for the model
        worker = Worker(algo, experience_replay, experience_queue)
        worker.train(
            agent_procs, args["batch_size"], args["start_size"],
            args["save_path"], args["save_interval"]
        )
