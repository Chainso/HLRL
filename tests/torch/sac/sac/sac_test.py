from time import time
from queue import deque

import torch

def make_state(state, recurrent):
    state = torch.FloatTensor(state).to(algo.device).unsqueeze(0)

    if recurrent:
        state = state.unsqueeze(0)

    return state

def train(
        env,
        algo,
        experience_replay,
        num_episodes: int,
        batch_size: int,
        decay: float,
        n_steps: int,
        start_size: int,
        save_path = None,
        save_interval = 5000,
        logger = None,
        recurrent: bool = False
    ) -> None: 
        if logger is not None:
            agent_train_start_time = time()

        algo.create_optimizers()

        ready_experiences = {}

        step = 0
        for episode in range(1, num_episodes + 1):
            env.reset()

            if logger is not None:
                episode_time = time()

            ep_reward = 0

            episode_step = 0
            while not env.terminal:
                if logger is not None:
                    step_time = time()

                state = make_state(env.state, recurrent)

                action, q_val = algo.step(state)
                env_action = action.squeeze().cpu().numpy()

                next_state, reward, terminal, info = env.step(env_action)

                algo.env_steps += 1
                step += 1
                episode_step += 1

                ep_reward += reward

                reward = make_state([reward], recurrent)

                terminal = terminal * (not info.get("TimeLimit.truncated"))
                terminal = make_state([terminal], recurrent)

                next_state = make_state(next_state, recurrent)

                next_q_val = algo.step(next_state)[1]
                target_q_val = reward + decay * next_q_val

                experience = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "terminal": terminal,
                    "q_val": q_val,
                    "target_q_val": target_q_val
                }

                experience_replay.calculate_and_add(experience)

                algo.train_from_buffer(
                    experience_replay, batch_size, start_size, save_path,
                    save_interval
                )

                if logger is not None:
                    logger["Train/Agent Steps per Second"] = (
                        1 / (time() - step_time), algo.env_steps
                    )

            algo.env_episodes += 1

            if logger is not None:
                logger["Train/Episode Reward"] = (
                    ep_reward, algo.env_episodes
                )

                logger["Train/Episode Reward over Wall Time (s)"] = (
                    ep_reward, time() - agent_train_start_time
                )

                logger["Train/Agent Episodes per Second"] = (
                    1 / (time() - episode_time), algo.env_episodes
                )

            print("Episode {0}\t|\tStep {1}\t|\tReward: {2}".format(
                algo.env_episodes, algo.env_steps, ep_reward    
            ))


if(__name__ == "__main__"):
    import torch.nn as nn
    import torch.multiprocessing as mp
    import gym
    import yaml

    from gym.wrappers import RescaleAction
    from argparse import ArgumentParser, Namespace
    from functools import partial
    from pathlib import Path

    from hlrl.core.logger import TensorboardLogger
    from hlrl.core.common.functional import compose
    from hlrl.torch.trainers import OffPolicyTrainer
    from hlrl.core.envs.gym import GymEnv
    from hlrl.torch.algos import SAC, SACRecurrent, RND, TorchRecurrentAlgo
    from hlrl.torch.policies import (
        LinearPolicy, LinearSAPolicy, TanhGaussianPolicy, LSTMPolicy,
        LSTMSAPolicy, LSTMGaussianPolicy
    )
    from hlrl.torch.experience_replay import TorchPER

    mp.set_start_method("spawn")

    # The hyperparameters as command line arguments
    parser = ArgumentParser(
        description="Twin Q-Function SAC example on the Pendulum-v0 "
            + "environment."
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
    parser.add_argument(
        "-c", "--config_file", type=str,
        help="the path of the yaml configuration file"
    )

    # Env args
    parser.add_argument(
        "-r", "--render", action="store_true",
        help="render the environment"
    )
    parser.add_argument(
        "-e", "--env", default="Pendulum-v0",
        help="the gym environment to train on"
    )

    # Model args
    parser.add_argument(
		"--device", type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
		help="the device (cpu/gpu) to train and play on"
	)
    parser.add_argument(
        "--hidden_size", type=int, default=256,
        help="the size of each hidden layer"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3,
        help="the number of layers"
    )

    # Algo args
    parser.add_argument(
        "--recurrent", action="store_true",
        help="make the network recurrent (using LSTM)"
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
        "-p", "--play", action="store_true",
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
        "--save_interval", type=int, default=5000,
        help="the number of batches in between saves"
    )
    parser.add_argument(
		"--episodes", type=int, default=500,
		help="the number of episodes to play for if playing"
	)
    parser.add_argument(
        "--training_steps", type=int, default=50000,
        help="the number of training steps to train for"
    )

    # Agent args
    parser.add_argument(
        "--n_steps", type=int, default=5, help="the number of decay steps"
    )
    parser.add_argument(
        "--num_agents", type=int, default=1,
        help="the number of agents to run concurrently, 0 is single process"
    )
    parser.add_argument(
        "--model_sync_interval", type=int, default=0,
        help="the number of training steps between agent model syncs, if 0, "
            + "all processes will share the same model",
    )
    parser.add_argument(
        "--num_prefetch_batches", type=int, default=16,
        help="the number of batches to prefetch to the learner in distributed "
            + "learning"
    )
    parser.add_argument(
        "--local_batch_size", type=int, default=64,
        help="the number of experiences the agent sends at once in distributed "
            + "learning"
    )
    parser.add_argument(
        "--silent", action="store_true",
        help="will run without standard output from agents"
    )
    parser.add_argument(
        "--action_mask", type=int, nargs="+", default=None,
        help="the mask over available actions to take"
    )
    parser.add_argument(
        "--default_action", type=float, default=None,
        help="the default action values if a masked is used"
    )

    # Experience Replay args
    parser.add_argument(
		"--er_capacity", type=int, default=50000,
		help="the maximum amount of experiences in the replay buffer"
	)
    parser.add_argument(
        "--er_alpha", type=float, default=0, help="the alpha value for PER"
    )
    parser.add_argument(
        "--er_beta", type=float, default=0.4, help="the alpha value for PER"
    )
    parser.add_argument(
        "--er_beta_increment", type=float, default=1e-3,
		help="the increment of the beta value on each sample for PER"
    )
    parser.add_argument(
        "--er_epsilon", type=float, default=1e-4,
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
        "--max_factor", type=float, default=0.9,
        help="if recurrent, factor of max priority to mean priority for R2D2"
    )

    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file, "r") as config_file:
            # Want to make sure to use config file args over defaults, but given
            # arguments over config file arguments
            args = vars(args)
            default = vars(parser.parse_args([]))
            given_args = {
                key: args[key] for key in args if args[key] != default[key]
            }

            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)

            args = {**args, **config_dict, **given_args}
            args = Namespace(**args)

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

    # Initialize the environment, and rescale for Tanh policy
    args.vectorized = False
    env_builder = partial(gym.make, args.env)
    env_builder = compose(env_builder, partial(RescaleAction, a=-1, b=1))
    env_builder = compose(env_builder, GymEnv)
    env = env_builder()

    # Action masking
    if args.action_mask and not args.default_action:
        args.default_action = (0,) * len(args.action_mask)
        args.masked = True

        # Resize the action space to accomdate for the mask
        env.action_space = (sum(args.action_mask),) +  env.action_space[1:]

    # The algorithm logger
    algo_logger = (
        None if logs_path is None else TensorboardLogger(logs_path + "/algo")
    )
    
    # Initialize SAC
    activation_fn = nn.ReLU
    optim = partial(torch.optim.Adam, lr=args.lr)

    # Setup networks
    if args.recurrent:
        num_lin_before = 1 if args.num_layers > 1 else 0
        num_lin_after = max(args.num_layers - 2, 1)

        qfunc = LSTMSAPolicy(
            env.state_space[0], env.action_space[0], 1, args.hidden_size,
            num_lin_before, args.hidden_size, 1, args.hidden_size,
            num_lin_after, activation_fn
        )

        policy = LSTMGaussianPolicy(
            env.state_space[0], env.action_space[0], args.hidden_size,
            num_lin_before, args.hidden_size, 1, args.hidden_size,
            num_lin_after, activation_fn, squished=True
        )

        algo = SACRecurrent(
            env.action_space, qfunc, policy, args.discount ** args.n_steps,
            args.polyak, args.target_update_interval, optim, optim, optim,
            args.twin, args.device, algo_logger
        )

        algo = TorchRecurrentAlgo(algo, args.burn_in_length, args.n_steps)
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
            env.action_space, qfunc, policy, args.discount ** args.n_steps,
            args.polyak, args.target_update_interval, optim, optim, optim,
            args.twin, args.device, algo_logger
        )

    if args.load_path is not None:
        algo.load(args.load_path)

    algo = algo.to(args.device)

    experience_replay = TorchPER(
        args.er_capacity, args.er_alpha, args.er_beta, args.er_beta_increment,
        args.er_epsilon
    )

    # The agent logger
    agent_logger = (
        None if logs_path is None else TensorboardLogger(logs_path + "/agent")
    )

    train(
        env,
        algo,
        experience_replay,
        args.episodes,
        args.batch_size,
        args.discount, 
        args.n_steps,
        args.start_size,
        save_path,
        args.save_interval,
        agent_logger,
        args.recurrent
    )