import numpy as np

def random_action(env, obs):
    function_id = np.random.choice(obs.observation["available_actions"])
    args = [[np.random.randint(0, size) for size in arg.sizes]
            for arg in env.action_spec[0].functions[function_id].args]
    return [function_id] + args

if __name__ == "__main__":
    import sys
    from absl import flags

    import torch
    import torch.nn as nn
    import torch.multiprocessing as mp
    import gym
    import lolgym.envs
    import yaml

    from argparse import ArgumentParser, Namespace
    from functools import partial
    from pathlib import Path

    from hlrl.core.logger import TensorboardLogger
    from hlrl.core.common.functional import compose
    from hlrl.torch.trainers import OffPolicyTrainer
    from hlrl.core.envs.gym import GymEnv
    from hlrl.torch.algos import (
        RainbowIQN, RainbowIQNRecurrent, RND, TorchRecurrentAlgo
    )
    from hlrl.torch.policies import LinearPolicy, LSTMPolicy

    mp.set_start_method("spawn")

    flags.FLAGS(sys.argv)

    env = gym.make("LoLGame-v0")
    env.settings["config_path"] = "./config.txt"
    env.settings["map_name"] = "Howling Abyss" # Set the map
    env.settings["human_observer"] = True # Set to true to run league client
    env.settings["host"] = "127.0.1.1" # Set this to a local ip
    env.settings["players"] = "Nidalee.BLUE,Lucian.PURPLE"

    print(env.observation_space)
    print(env.action_space)

    steps = 500

    obs_n = env.reset()
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0

    for step in range(1, steps + 1): # Use number of steps instead of deaths to end episode
        actions = [random_action(env, timestep) for timestep in obs_n]
        obs_n, reward_n, done_n, _ = env.step(actions)
        ep_reward += sum(reward_n)

        if any(done_n):
            break

    env.close()
    print("done")