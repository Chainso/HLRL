from hlrl.core.algos import RLAlgoWrapper

class RND(RLAlgoWrapper):
    """
    The Random Network Distillation Algorithm
    https://arxiv.org/abs/1810.12894
    """
    def __init__(algo, rnd, rnd_optim):
        """
        Creates the wrapper to use RND exploration with the algorithm.

        Args:
            algo (TorchRLAlgo): The algorithm to run.
            rnd (torch.nn.Module): The RND network.
            rnd_optim (callable): The function to create the optimizer for RND.
        """
        super().__init__(algo)

        self.rnd = rnd

        def init_weights(m):
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight.data)

        self.rnd_target = deepcopy(rnd).apply(init_weights)
        self.rnd_optim = rnd_optim(self.rnd.parameters())

    def _get_loss(states):
        """
        Returns the loss of the RND network on the given states.
        """
        rnd_loss_func = nn.MSELoss()

        rnd_pred = self.rnd(states)
        rnd_target = self.rnd_target(states)

        rnd_loss = rnd_loss_func(rnd_pred, rnd_target)

        return rnd_loss

    def train_batch(states, actions, rewards, next_states, *training_args):
        """
        Trains the RND network before training the batch on the algorithm.
        """
        rnd_loss = self._get_loss(next_states)

        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()

        if self.logger is not None:
            self.logger["Train/RND Loss"] = (rnd_loss, self.training_steps) 

        return self.algo.train_batch(states, actions, rewards, next_states,
                                     *training_args)

    def intrinsic_reward(self, states):
        """
        Computes the intrinsic reward of the states.
        """
        return self._get_loss(states).cpu().numpy()