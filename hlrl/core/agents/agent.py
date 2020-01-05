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

    def _make_input_from_state(self, state):
        """
        Creates the algorithm input from the env state (does nothing by default)
        """
        return state

    def step(self):
        """
        Takes 1 step in the agent's environment. Returns the
        (state, action, reward, next state, terminal, additional algo returns,
        env info) tuple. Resets the environment if the current state is a
        terminal.
        """
        if(self.env.terminal):
            self.env.reset()

        if(self.render):
            self.env.render()

        state = self.env.state
        algo_inp = self._make_input_from_state(state)

        algo_step = self.algo.step(algo_inp)

        action = algo_step[0]
        env_act = action[0].cpu().numpy()

        # Get additional information if it is there
        if(len(algo_step) > 1):
            add_algo_ret = algo_step[1:]
        else:
            add_algo_ret = []

        next_state, reward, terminal, info = self.env.step(env_act)
        next_state = self._make_input_from_state(next_state)

        return (algo_inp, action, reward, next_state, terminal, info,
                add_algo_ret)

    def play(self, num_episodes):
        """
        Resets and plays the environment with the algorithm. Returns the average
        reward per episode.

        Args:
            num_episodes (int): The numbers of episodes to play for.
        """
        avg_reward = 0

        for episode in range(1, num_episodes + 1):
            self.env.reset()
            ep_reward = 0
            
            while(not self.env.terminal):
                ep_reward += self.step()[2]

            if(self.logger is not None):
                self.logger["Play"] = ep_reward, episode

            avg_reward += ep_reward / num_episodes

        if(self.logger is not None):
            self.logger["Play/Average"] = avg_reward

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