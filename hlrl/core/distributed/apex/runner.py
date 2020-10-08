import multiprocessing as mp

from typing import Tuple, Any, Callable

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
            done_event (multiprocessing.Event): The event to be set that alerts
                the subprocesses to stop.
        """
        self.done_event = done_event

    def stop(self):
        """
        Sets the done event to allow subprocesses to stop.
        """
        self.done_event.set()

    def start(self, learner_args: Tuple[Any], worker_args: Tuple[Any],
        agent_builder: Callable[[], RLAgent],
        agent_train_args: Tuple[Tuple[Any]]):
        """
        Starts the runner, creating and starting the learner, worker and agent
        procceses.

        Args:
            learner_args (Tuple[Any]): Arguments for the Ape-X learner.
            worker_args (Tuple[Any]): Arguments for the Ape-X worker.
            agent_builder (Callable[[], RLAgent]): A nullary function that
                constructs an agent.
            agent_train_args (Tuple[Tuple[Any]]): Arguments for the agent
                training processes.
        """
        # Create the learner
        learner = ApexLearner()
        learner_proc = mp.Process(target=learner.train, args=learner_args)

        # Create the worker for the model
        worker = ApexWorker()
        worker_proc = mp.Process(target=worker.train, args=worker_args)

        # Create the agents
        agents = []
        agent_procs = []

        for i in range(len(agent_train_args)):
            agent = agent_builder()

            agents.append(agent)
            agent_procs.append(
                mp.Process(target=agent.train_process, args=agent_train_args[i])
            )

        # Start all processes
        learner_proc.start()
        worker_proc.start()

        for agent_proc in agent_procs:
            agent_proc.start()

        # Wait for processes to end
        # TODO MAKE SURE TO SET DONE_EVENT IN ONE OF THE PROCESS PROPERLY
        learner_proc.join()
        worker_proc.join()

        for agent_proc in agent_procs:
            agent_proc.join()