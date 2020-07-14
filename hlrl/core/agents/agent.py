import torch.multiprocessing as mp
import os
from abc import abstractmethod

class RLAgent():
    """
    An agent that collects (state, action, reward, next state) tuple
    observations
    """
    def __init__(self, env, algo, render=False, logger=None):
        """
        Creates an agent that interacts with the given environment using the
        algorithm given.

        Args:
            env (Env): The environment the agent will explore in.

            algo (RLAlgo): The algorithm the agent will use the explore the
                           environment.
            render (bool): If the environment is to be rendered (if applicable)
            logger (Logger, optional) : The logger to log results while
                                        interacting with the environment.
        """
        self.env = env
        self.algo = algo
        self.render = render
        self.logger = logger

    def transform_state(self, state):
        """
        Creates the tuple of algorithm inputs from the env state
        """
        return (state,)

    def transform_algo_step(self, algo_step):
        """
        Transforms the algorithm step on the observation to
        (action, ...).
        """
        return (algo_step,)

    def transform_reward(self, reward):
        """
        Transforms the reward of an environment step.
        """
        return reward

    def transform_terminal(self, terminal):
        """
        Transforms the terminal of an environment step.
        """
        return terminal

    def transform_action(self, action):
        """
        Transforms the action of the algorithm output to be usable with the
        environment.
        """
        return action

    def reset(self):
        """
        Resets the agent.
        """
        # The simplest agent doesn't require anything to reset
        pass

    def step(self):
        """
        Takes 1 step in the agent's environment. Returns the
        (state, action, reward, next state, terminal, *additional algo returns,
        env info) tuple. Resets the environment if the current state is a
        terminal.
        """
        if(self.env.terminal):
            self.env.reset()

        if(self.render):
            self.env.render()

        state = self.env.state

        state_transed = self.transform_state(state)

        state, inp_extras = state_transed[0], state_transed[1:]

        algo_step = self.algo.step(state, *inp_extras)
        algo_step = self.transform_algo_step(algo_step)
        action, algo_extras = algo_step[0], algo_step[1:]

        env_action = self.transform_action(action)

        next_state, reward, terminal, info = self.env.step(env_action)

        next_state = self.transform_state(next_state)
        next_state, next_inp_extras = next_state[0], next_state[1:]

        reward = self.transform_reward(reward)
        terminal = self.transform_terminal(terminal)
        print("Pre return on step")
        return (state, action, reward, next_state, terminal, info, inp_extras,
                algo_extras, next_inp_extras)

    def play(self, num_episodes):
        """
        Resets and plays the environment with the algorithm. Returns the average
        reward per episode.

        Args:
            num_episodes (int): The numbers of episodes to play for.
        """
        avg_reward = 0

        for episode in range(1, num_episodes + 1):
            self.reset()
            self.env.reset()
            ep_reward = 0
            
            while(not self.env.terminal):
                ep_reward += self.step()[2]

            if(self.logger is not None):
                self.logger["Play/Episode Reward"] = ep_reward, episode

            print("Episode", str(episode) + ":", ep_reward)
            avg_reward += ep_reward / num_episodes

        if(self.logger is not None):
            self.logger["Play/Average Reward"] = avg_reward

        print("-------------------")
        print("Average Reward:", avg_reward)

        return avg_reward

    @abstractmethod
    def train(self, num_episodes):
        """
        Trains the algorithm for the number of episodes specified on the
        environment.

        Args:
            num_episodes (int): The number of episodes to train for.
        """
        pass

class AgentPool():
    """
    An asynchronous pool of agents.
    """
    def __init__(self, agents):
        """
        Properties:
            agents ([RLAgent]): A list of agents in the pool.
        """
        self.agents = agents

    def train(self, *agent_train_args):
        """
        Trains each agent using the arguments given.

        Args:
            agent_train_args ([object]): The list of arguments for each agents.
        """
        procs = [mp.Process(target = agent.train, args=agent_train_args)
                 for agent in self.agents]

        for proc in procs:
            proc.start()

        return procs