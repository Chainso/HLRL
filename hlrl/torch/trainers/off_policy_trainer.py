from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
import torch.multiprocessing as mp

from hlrl.core.envs import Env
from hlrl.core.algos import RLAlgo
from hlrl.core.logger import TensorboardLogger
from hlrl.core.common.functional import compose, partial_iterator
from hlrl.core.distributed import ApexRunner
from hlrl.core.agents import (
    RLAgent, OffPolicyAgent, IntrinsicRewardAgent, MunchausenAgent, QueueAgent,
    TimeLimitAgent
)
from hlrl.torch.algos import TorchRLAlgo
from hlrl.torch.agents import (
    TorchRLAgent, SequenceInputAgent, ExperienceSequenceAgent,
    TorchRecurrentAgent, TorchOffPolicyAgent, UnmaskedActionAgent
)
from hlrl.torch.experience_replay import TorchPER, TorchPSER, TorchR2D2
from hlrl.torch.distributed import TorchApexWorker

class OffPolicyTrainer():
    """
    A trainer for off-policy algorithms.
    """
    def train(
            self,
            args: Namespace,
            env_builder: Callable[[], Env],
            algo: RLAlgo
        ) -> None:
        """
        Trains the algorithm on the environment given using the argument
        namespace as parameters.
        
        "args" must have the following attributes:
        {
            experiment_path (str): The path to save experiment results and
                models.
            render (bool): Render the environment.
            steps_per_episode (Optional[int]): The number of steps in each
                episode.
            silent (bool): Will run without standard output from agents.
            action_mask (Optional[Tuple[bool, ...]]): The action mask to mask or
                unmask.
            masked (Optional[bool]): If an action mask is given, should be True
                if the returned agent actions are already masked.
            default_action (Optional[Tuple[float, ...]]): If an action mask is
                given and going from masked -> unmasked, this should be the
                default values for the actions.
            decay (float): The gamma decay for the target Q-values.
            n_steps (int): The number of decay steps.
            num_agents (int): The number of agents to run concurrently, 0 is
                single process.
            model_sync_interval (int): The number of training steps between
                agent model syncs, if 0, all processes will share the same
                model.
            num_prefetch_batches (int): The number of batches to prefetch to the
                learner in distributed learning.
            local_batch_size (int): The number of experiences the agent sends at
                once in distributed learning.
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
            RLAgent, algo=algo, render=args.render, silent=args.silent
        )
        agent_builder = compose(
            agent_builder,
            partial(OffPolicyAgent, decay=args.discount)
        )

        steps_per_episode = (
            args.steps_per_episode if "steps_per_episode" in args else None
        )

        agent_builder = compose(
            agent_builder,
            partial(TimeLimitAgent, max_steps=steps_per_episode)
        )

        if not args.play:
            # Experience replay
            # Won't increment in multiple processes to keep it consistent
            # across actors
            er_beta_increment = (
                args.er_beta_increment if args.num_agents == 0 else 0
            )

            if args.recurrent:
                experience_replay_func = partial(
                    TorchR2D2, alpha=args.er_alpha, beta=args.er_beta,
                    beta_increment=er_beta_increment, epsilon=args.er_epsilon,
                    max_factor=args.max_factor
                )
            else:
                experience_replay_func = partial(
                    TorchPER, alpha=args.er_alpha, beta=args.er_beta,
                    beta_increment=er_beta_increment, epsilon=args.er_epsilon
                )

            if args.num_agents > 0:
                recv_pipes = []
                send_pipes = []

                prestart_func = None

                if args.model_sync_interval == 0:
                    self._start_training(algo, args)
                    algo.share_memory()

                    recv_pipes = [None] * args.num_agents
                else:
                    prestart_func = partial(
                        self._start_training, algo=algo, args=args
                    )

                    # Force CPU for now to avoid re-instantiating cuda in
                    # subprocesses
                    algo.device = torch.device("cpu")
                    algo = algo.to(algo.device)

                    for i in range(args.num_agents):
                        param_pipe = mp.Pipe(duplex=False)

                        recv_pipes.append(param_pipe[0])
                        send_pipes.append(param_pipe[1])

                # Just needed to get the error/priority calculations
                dummy_experience_replay = experience_replay_func(capacity=1)

                # Must come before the other wrapper since there are infinite
                # recursion errors
                # TODO come up with a better way to implement wrappers
                agent_builder = compose(
                    agent_builder,
                    partial_iterator(
                        QueueAgent,
                        agent_id=(iter(range(args.num_agents)), True),
                        experience_replay=(dummy_experience_replay, False),
                        param_pipe=(iter(recv_pipes), True)
                    )
                )

        agent_builder = compose(
            agent_builder,
            partial(TorchRLAgent, batch_state=not args.vectorized)
        )
        
        if "action_mask" in args and args.action_mask:
            # TODO: Will have to add an action mask wrapper later
            if args.masked:
                agent_builder = compose(
                    agent_builder,
                    partial(
                        UnmaskedActionAgent, action_mask=args.action_mask,
                        default_action=args.default_action
                    )
                )

        agent_builder = compose(agent_builder, TorchOffPolicyAgent)

        if args.recurrent:
            agent_builder = compose(
                agent_builder, SequenceInputAgent, TorchRecurrentAgent
            )

        if args.play:
            algo = algo.to(args.device)
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

            algo.train()

            if args.recurrent:
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

            experience_replay = experience_replay_func(
                capacity=args.er_capacity
            )

            base_agent_logs_path = None
            if logs_path is not None:
                base_agent_logs_path = logs_path + "/train-agent"

            # Single process
            if args.num_agents == 0:
                self._start_training(algo, args)

                agent_logger = None
                if base_agent_logs_path is not None:
                    agent_logger = TensorboardLogger(base_agent_logs_path)

                agent = agent_builder(env=env_builder(), logger=agent_logger)

                end_steps = algo.training_steps + args.training_steps

                agent.train(
                    1,
                    args.n_steps + 1,
                    1,
                    experience_replay,
                    args.batch_size,
                    args.start_size,
                    save_path,
                    args.save_interval,
                    exit_condition=lambda: algo.training_steps >= end_steps
                )

            # Multiple processes
            else:
                done_event = mp.Event()

                # Number of agents + worker + learner
                queue_barrier = mp.Barrier(args.num_agents + 2)

                agent_queue = mp.Queue(
                    maxsize=args.num_prefetch_batches * args.num_agents * 4
                )
                sample_queue = mp.Queue(maxsize=args.num_prefetch_batches)
                priority_queue = mp.Queue(maxsize=args.num_prefetch_batches)

                learner_args = (dummy_experience_replay,)
                learner_train_args = (
                    algo, done_event, queue_barrier, args.training_steps,
                    sample_queue, priority_queue, send_pipes,
                    args.model_sync_interval, save_path, args.save_interval
                )

                worker = TorchApexWorker()
                worker_args = (
                    experience_replay, done_event, queue_barrier, agent_queue,
                    sample_queue, priority_queue, args.batch_size,
                    args.start_size
                )

                agent_builders = []
                agent_train_args = []
                agent_train_kwargs = []

                for i in range(args.num_agents):
                    agent_logger = None
                    if base_agent_logs_path is not None:
                        agent_logs_path = (
                            base_agent_logs_path + "-" + str(i + 1)
                        )
                        agent_logger = TensorboardLogger(agent_logs_path)

                    agent_builders.append(
                        partial(agent_builder, logger=agent_logger)
                    )

                    agent_train_args.append((
                        1, args.n_steps + 1, args.local_batch_size,
                        agent_queue
                    ))
                    agent_train_kwargs.append({
                        "exit_condition": done_event.is_set,
                        "queue_barrier": queue_barrier
                    })

                runner = ApexRunner(done_event)
                runner.start(
                    learner_args, learner_train_args, worker, worker_args,
                    env_builder, agent_builders, agent_train_args,
                    agent_train_kwargs, prestart_func
                )

    def _start_training(self, algo: TorchRLAlgo, args: Any) -> None:
        """
        A function to call at the start of training.

        Args:
            algo: The algorithm to start training for.
            args: Arguments for the program.
        """
        algo.device = args.device
        algo.to(args.device)
        algo.create_optimizers()
