import torch
import torch.nn as nn

from hlrl.torch.policies import LinearPolicy, LinearSAPolicy, GaussianPolicy, TanhGaussianPolicy


if(__name__ == "__main__"):
    import torch.multiprocessing as mp

    from argparse import ArgumentParser

    from hlrl.core.logger import TensorboardLogger
    from hlrl.torch.algos.sac.sac import SAC
    from hlrl.core.envs import GymEnv
    from hlrl.core.agents import AgentPool
    from hlrl.torch.agents import OffPolicyAgent
    from hlrl.torch.experience_replay import TorchPER

    mp.set_start_method("spawn")

    # The hyperparameters as command line arguments
    parser = ArgumentParser(description="Twin Q-Function SAC example on "
                            + "the MountainCarContinuous-v0 environment.")

    # Logging
    parser.add_argument("-l, --logs_path ", dest="logs_path", type=str,
                        help="log training data to tensorboard using the path")

    # Env args
    parser.add_argument("-r, --render", dest="render", action="store_true",
                        help="render the environment")
    parser.add_argument("-e,", "--env", dest="env", default="Pendulum-v0",
                        help="the gym environment to train on")

    # Model args
    parser.add_argument("--hidden_size", type=int, default=32,
                        help="the size of each hidden layer")
    parser.add_argument("--num_hidden", type=int, default=1,
                        help="the number of hidden layers before the output "
                             + "layers")

    # Algo args
    parser.add_argument("--device", type=str, default="cpu",
                        help="the device (cpu/gpu) to train and play on")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="the next state reward discount factor")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="the coefficient of temperature of the entropy "
                             + "for SAC")
    parser.add_argument("--polyak", type=float, default=0.995,
                        help="the polyak constant for the target network "
                             + "updates")
    parser.add_argument("--target_update_interval", type=float, default=1,
                        help="the number of training steps inbetween target "
                             + "network updates")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="the learning rate")
    parser.add_argument("--twin", type=bool, default=True,
                        help="true if SAC should use twin Q-networks")

    # Training args
    parser.add_argument("--batch_size", type=int, default=32,
                        help="the batch size of the training set")
    parser.add_argument("--save_path", type=str, default=None,
                        help="the path to save the model to")
    parser.add_argument("--save_interval", type=int, default=10000,
                        help="the number of batches in between saves")

    # Agent args
    parser.add_argument("--episodes", type=int, default=1000,
                        help="the number of episodes to train for")
    parser.add_argument("--decay", type=float, default=0.99,
                        help="the gamma decay for the target Q-values")
    parser.add_argument("--n_steps", type=int, default=1,
                        help="the number of decay steps")

    # Experience Replay args
    parser.add_argument("--er_capacity", type=float, default=50000,
                        help="the alpha value for PER")
    parser.add_argument("--er_alpha", type=float, default=0.6,
                        help="the alpha value for PER")
    parser.add_argument("--er_beta", type=float, default=0.4,
                        help="the alpha value for PER")
    parser.add_argument("--er_beta_increment", type=float, default=1e-3,
                        help="the alpha value for PER")
    parser.add_argument("--er_epsilon", type=float, default=1e-2,
                        help="the epsilon value for PER")

    args = vars(parser.parse_args())

    # Initialize the environment
    env = GymEnv(args["env"])

    # The logger
    logger = args["logs_path"]
    logger = None if logger is None else TensorboardLogger(logger)

    # Initialize SAC
    activation_fn = nn.ReLU
    qfunc = LinearSAPolicy(env.state_space[0], env.action_space[0], 1,
                           args["hidden_size"], args["num_hidden"],
                           activation_fn)
    policy = TanhGaussianPolicy(env.state_space[0], env.action_space[0],
                                args["hidden_size"], args["num_hidden"],
                                activation_fn)

    optim = lambda params: torch.optim.Adam(params, lr=args["lr"])
    algo = SAC(env.action_space, qfunc, policy, args["discount"],
               args["temperature"], args["polyak"],
               args["target_update_interval"], optim, optim, optim,
               args["twin"], logger).to(torch.device(args["device"]))
    algo.train()
    algo.share_memory()

    # Experience replay
    experience_replay = TorchPER(args["er_capacity"], args["er_alpha"],
                                 args["er_beta"], args["er_beta_increment"],
                                 args["er_epsilon"])
    experience_queue = mp.JoinableQueue()

    # Initialize agent
    agents = [
        OffPolicyAgent(env, algo, experience_queue, args["render"],
                       logger=logger, device=args["device"])
    ]

    agent_pool = AgentPool(agents)
    agent_procs = agent_pool.train(args["episodes"], args["decay"],
                                   args["n_steps"])

    #### TEMPORARY LOGGER ####
    algo.logger = TensorboardLogger("./logs")

    while any(proc.is_alive() for proc in agent_procs):
        experience = experience_queue.get()
        experience_queue.task_done()

        if experience is None:
            # Wait on the train processes
            for proc in agent_procs:
                proc.join()
        else:
            experience_replay.add(*experience)
            algo.train_from_buffer(experience_replay, args["batch_size"],
                                   args["save_path"], args["save_interval"])
