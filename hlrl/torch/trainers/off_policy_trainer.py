from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Callable

import torch.multiprocessing as mp

from hlrl.core.envs import Env
from hlrl.core.algos import RLAlgo
from hlrl.core.logger import TensorboardLogger
from hlrl.core.common.functional import compose
from hlrl.core.distributed import ApexLearner, ApexRunner
from hlrl.core.agents import (
    OffPolicyAgent, IntrinsicRewardAgent, MunchausenAgent, QueueAgent
)
from hlrl.torch.agents import (
    TorchRLAgent, SequenceInputAgent, ExperienceSequenceAgent,
    TorchRecurrentAgent, TorchOffPolicyAgent
)
from hlrl.torch.experience_replay import TorchPER, TorchPSER, TorchR2D2

class OffPolicyTrainer():
    """
    A trainer for off-policy algorithms.
    """
    def train(self,
              args: Namespace,
              env_builder: Callable[[], Env],
              algo: RLAlgo):
        """
        Trains the algorithm on the environment given using the argument
        namespace as parameters.
        
        "args" must have the following attributes:
        {
            experiment_path (str): The path to save experiment results and
                models.
            render (bool): Render the environment.
            silent (bool): Will run without standard output from agents.
            decay (float): The gamma decay for the target Q-values.
            n_steps (int): The number of decay steps.
            num_agents (int): The number of agents to run concurrently, 0 is
                single process.
            vectorized (bool): If the environment is vectorized.
            recurrent (bool),Make the network recurrent (using LSTM)
            play (bool): Runs the environment using the model instead of
                training.
            exploration (str, ["rnd", "munchausen"]): The type of exploration to
                use.
		    episodes (int): The number of episodes to play for if playing.
            er_capacity (int): The alpha value for PER.
            batch_size (int): The batch size of the training set.
            training_steps (int): The number of training steps to train for.
            start_size (int): The size of the replay buffer before training.
            er_alpha (float): The alpha value for PER.
            er_beta (float): The alpha value for PER.
            er_beta_increment (float): The increment of the beta value on each
                sample for PER.
            er_epsilon (float): The epsilon value for PER.
            burn_in_length (int): If recurrent, the number of burn in samples
                for R2D2.
            sequence_length (int): If recurrent, the length of the sequence to
                train on.
            max_factor (int): If recurrent, factor of max priority to mean
                priority for R2D2.
        }

        Args:
            args: The namespace of arguments for training.
            env_builder: The nullary function to create the environment.
            algo: The algorithm to train.
        """
        logs_path = None
        save_path = None

        if args.experiment_path is not None:
            logs_path = Path(args.experiment_path, "logs")
            logs_path.mkdir(parents=True, exist_ok=True)
            logs_path = str(logs_path)

            save_path = Path(args.experiment_path, "models")
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = str(save_path)

        # Create agent class
        agent_builder = partial(
            OffPolicyAgent, algo=algo, render=args.render, silent=args.silent
        )

        if args.num_agents > 0:
            agent_builder = compose(agent_builder, QueueAgent)

        agent_builder = compose(
            agent_builder,
            partial(TorchRLAgent, batch_state=not args.vectorized)
        )
        agent_builder = compose(agent_builder, TorchOffPolicyAgent)
        
        if args.recurrent:
            agent_builder = compose(
                agent_builder, SequenceInputAgent, TorchRecurrentAgent
            )

        if args.play:
            algo.eval()

            agent_logger = (
                None if logs_path is None
                else TensorboardLogger(logs_path + "/play-agent")
            )

            agent = agent_builder(env=env_builder(), logger=agent_logger)
            agent.play(args.episodes)
        else:
            if args.exploration == "rnd":
                agent_builder = compose(agent_builder, IntrinsicRewardAgent)
            elif args.exploration == "munchausen":
                agent_builder = compose(
                    agent_builder, partial(MunchausenAgent, alpha=0.9)
                )

            algo.create_optimizers()

            algo.train()
            algo.share_memory()

            # Experience replay
            er_capacity = int(args.er_capacity)
            if args.recurrent:
                experience_replay_func = partial(
                    TorchR2D2, er_capacity, args.er_alpha, args.er_beta,
                    args.er_beta_increment, args.er_epsilon, args.max_factor
                )

                agent_builder = compose(
                    agent_builder,
                    partial(
                        ExperienceSequenceAgent,
                        sequence_length=(
                            args.burn_in_length + args.sequence_length
                        ),
                        overlap=args.burn_in_length
                    )
                )
            else:
                experience_replay_func = partial(
                    TorchPER, er_capacity, args.er_alpha, args.er_beta,
                    args.er_beta_increment, args.er_epsilon
                )

            experience_replay = experience_replay_func()

            base_agent_logs_path = None
            if logs_path is not None:
                base_agent_logs_path = logs_path + "/train-agent"

            # Single process
            if args.num_agents == 0:
                agent_logger = None
                if base_agent_logs_path is not None:
                    agent_logger = TensorboardLogger(base_agent_logs_path)

                agent = agent_builder(env=env_builder(), logger=agent_logger)

                agent.train(
                    args.episodes, 1, args.discount, args.n_steps,
                    experience_replay, args.batch_size, args.start_size,
                    save_path, args.save_interval
                )

            # Multiple processes
            else:
                done_event = mp.Event()

                # Number of agents + worker + learner
                queue_barrier = mp.Barrier(args.num_agents + 2)

                max_queue_size = 64
                experience_queue = mp.Queue(maxsize=max_queue_size)
                param_pipes = []
                param_send_interval = 400

                learner = ApexLearner(
                    done_event, queue_barrier, experience_replay
                )

                learner_args = (
                    algo, args.training_steps, args.batch_size, args.start_size,
                    experience_queue, param_pipes, param_send_interval,
                    save_path, args.save_interval
                )

                agents = []
                agent_train_args = []
                agent_train_kwargs = []

                for i in range(args.num_agents):
                    agent_logger = None
                    if base_agent_logs_path is not None:
                        agent_logs_path = (
                            base_agent_logs_path + "-" + str(i + 1)
                        )
                        agent_logger = TensorboardLogger(agent_logs_path)

                    agents.append(
                        agent_builder(env=env_builder(), logger=agent_logger)
                    )

                    agent_train_args.append((
                        1, 1, args.discount, args.n_steps, experience_queue,
                        queue_barrier
                    ))
                    agent_train_kwargs.append({
                        "exit_condition": done_event.is_set
                    })

                runner = ApexRunner(done_event)
                runner.start(
                    learner_args, agents, agent_train_args, agent_train_kwargs
                )
