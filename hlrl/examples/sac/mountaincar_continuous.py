import torch
import torch.nn as nn

from hlrl.torch.policies import LinearPolicy, LinearSAPolicy


if(__name__ == "__main__"):
    from argparse import ArgumentParser

    from hlrl.core.logger import make_tensorboard_logger
    from hlrl.torch.algos.sac.sac import SAC
    from hlrl.core.envs.gym.gym_env import GymEnv
    from hlrl.torch.agents import OffPolicyAgent

    # The hyperparameters as command line arguments
    parser = ArgumentParser(description = "Twin Q-Function SAC example on "
                            + "the MountainCarContinuous-v0 environment.")

    # Logging
    parser.add_argument("-l, --logs_path ", dest="logs_path", type=str,
                        help="log training data to tensorboard using the path")

    # Model arg
    parser.add_argument("--hidden_size", type=int, default=16,
                        help="the size of each hidden layer")
    parser.add_argument("--num_hidden", type=int, default=16,
                        help="the number of hidden layers before the output "
                             + "layers")

    # Algo args
    parser.add_argument("--device", type=str, default="cpu",
                        help="the device (cpu/gpu) to train and play on")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="the next state reward discount factor")
    parser.add_argument("--entropy", type=float, default=0.5,
                        help="the coefficient of entropy for SAC")
    parser.add_argument("--polyak", type=float, default=0.5,
                        help="the polyak constant for the target network "
                             + "updates")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="the learning rate")
    parser.add_argument("--twin", type=bool, default=True,
                        help="true if SAC should use twin Q-networks")
    args = vars(parser.parse_args())
    print(args)
    # Initialize the environment
    env = GymEnv("MountainCarContinuous-v0")

    # The logger
    logger = args["logs_path"]
    logger = None if logger is None else make_tensorboard_logger(logger)

    # Initialize SAC
    activation_fn = nn.ReLU
    qfunc = LinearSAPolicy(env.state_space[0], env.action_space[0], 1,
                           args["hidden_size"], args["num_hidden"],
                           activation_fn)
    policy = LinearPolicy(env.state_space[0], env.action_space[0],
                          args["hidden_size"], args["num_hidden"],
                          activation_fn)
    value_func = LinearPolicy(env.state_space[0], 1, args["hidden_size"],
                              args["num_hidden"], activation_fn)

    optim = lambda params: torch.optim.Adam(params, lr=args["lr"])
    algo = SAC(qfunc, policy, value_func, args["discount"],
               args["entropy"], args["polyak"], optim, optim, optim,
               args["twin"], logger)

    # Initialize agent
    

