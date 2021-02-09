from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch.multiprocessing as mp

from hlrl.core.envs import Env
from hlrl.core.distributed.apex.learner import ApexLearner
from hlrl.core.distributed.apex.worker import ApexWorker
from hlrl.core.agents import RLAgent

class ApexRunner():
    """
    A handler for the Ape-X framework.

    Based on Ape-X:
    https://arxiv.org/pdf/1803.00933.pdf
    """
    def __init__(self, done_event: mp.Event):
        """
        Creates the handler, with an event to allow stopping.

        Args:
            done_event: The event to be set that alerts the subprocesses to
                stop.
        """
        self.done_event = done_event

    def stop(self) -> None:
        """
        Sets the done event to allow subprocesses to stop.
        """
        self.done_event.set()

    def start(
            self,
            learner_args: Tuple[Any, ...],
            learner_train_args: Tuple[Any, ...],
            worker: ApexWorker,
            worker_args: Tuple[Any, ...],
            env_builder: Callable[[], Env],
            agent_builders: Tuple[Callable[[Env], RLAgent]],
            agent_train_args: Tuple[Tuple[Any], ...],
            agent_train_kwargs: Tuple[Dict[str, Any], ...],
            pretrain_func: Optional[Callable]
        ) -> None:
        """
        Starts the runner, creating and starting the learner, worker and agent
        procceses.

        Args:
            learner_args: Arguments for the Ape-X learner.
            learner_train_args: Arguments to start the train method of the
                learner.
            worker: The worker for Ape-X.
            worker_args: Arguments for the Ape-X worker.
            env_builder: A builder function to create an environment for an
                agent.
            agent_builders: A tuple of functions that create an individual agent
                each, when given the environment.
            agent_train_args: Arguments for the agent training processes.
            agent_train_kwargs: Keyword arguments for the agent training
                processes.
            pretrain_func: A function to call before training starts.
        """
        assert (
            len(agent_builders) == len(agent_train_args)
            == len(agent_train_kwargs)
        )

        all_procs = []

        # Create the worker for the model
        worker_proc = mp.Process(target=worker.train, args=worker_args)
        all_procs.append(worker_proc)

        # Create agent processes
        agent_procs = tuple(
            mp.Process(
                target=RLAgent.train_from_builder,
                args=(agent_builders[i], env_builder) + agent_train_args[i],
                kwargs=agent_train_kwargs[i]
            )
            for i in range(len(agent_builders))
        )

        for proc in agent_procs:
            all_procs.append(proc)

        # Start all processes
        for proc in all_procs:
            proc.start()

        # Create the learner
        if pretrain_func is not None:
            pretrain_func()

        learner = ApexLearner(*learner_args)
        learner.train(*learner_train_args)

        for proc in all_procs:
            proc.join()
