import queue
from multiprocessing import Barrier, Pipe, Queue
from typing import Any, Dict, Iterable, List, Optional

from hlrl.core.agents.wrappers import OffPolicyAgent
from hlrl.core.experience_replay import PER
from hlrl.core.common.wrappers import MethodWrapper

class QueueAgent(MethodWrapper):
    """
    An agent that sends its experiences into a queue 1 at a time rather than
    training directly.
    """
    def __init__(
            self,
            agent: OffPolicyAgent,
            agent_id: int,
            experience_replay: PER,
            param_pipe: Optional[Pipe] = None
        ):
        """
        Creates the queue agent, that passes experiences to a queue to be
        inserting into replay buffer.

        Args:
            agent: The off policy agent to wrap.
            agent_id: The id of the agent.
            experience_replay: The PER object responsible for computing the
                errors and priorites of experiences.
            param_pipe: The pipe to receive models parameters from.
        """
        super().__init__(agent)

        self.id = agent_id
        self.experience_replay = experience_replay
        self.param_pipe = param_pipe

    def receive_parameters(self) -> None:
        """
        Checks to see if a new set of model parameters are available, and
        updates the model if there is.
        """
        if self.param_pipe is not None and self.param_pipe.poll():
            # Make sure to save env episodes and env steps since those would not
            # be correct coming from another process
            env_episodes = self.algo.env_episodes
            env_steps = self.algo.env_steps

            self.algo.load(load_dict=self.param_pipe.recv())

            self.algo.env_episodes = env_episodes
            self.algo.env_steps = env_steps

    def train_step(
            self,
            ready_experiences: Dict[str, Iterable[Any]],
            batch_size: int,
            experience_queue: Queue
        ) -> Dict[str, List[Any]]:
        """
        Trains on the ready experiences.

        Args:
            ready_experiences: The buffer of experiences that can be trained on.
            batch_size: The batch size for training.
            experience_queue: The queue to send experiences to.

        Returns:
            The ready experiences that were not added to the queue.
        """
        # Check to see if a new model was sent
        self.receive_parameters()

        # Get length of a random key
        keys = list(ready_experiences)
        if len(keys) > 0:
            key = keys[0]
            
            # Going to use >= here because in the case the queue is full, the
            # number of ready experiences may be greater than the batch size
            if len(ready_experiences[key]) >= batch_size:
                batch, q_vals, target_q_vals = self.create_batch(
                    ready_experiences
                )

                errors = self.experience_replay.get_error(q_vals, target_q_vals)
                priorities = self.experience_replay.get_priority(errors)

                for i in range(len(batch)):
                    batch[i]["id"] = (self.id, self.algo.env_steps, i)

                try:
                    experience_queue.put_nowait((batch, priorities))
                    ready_experiences = {}
                except queue.Full:
                    pass

        return ready_experiences

    def train(
            self,
            *args: Any,
            queue_barrier: Optional[Barrier] = None,
            **kwargs: Any
        ) -> None:
        """
        Trains the algorithm for the number of episodes specified on the
        environment.

        Args:
            args: The positional arguments for the underlying agent.
            kwargs: The keyword arguments for the underlying agent.
            queue_barrier: The barrier to wait on before exiting.
        """
        self.om.train(*args, **kwargs)
        
        # Wait for all processes to finish using queues
        if queue_barrier:
            queue_barrier.wait()
