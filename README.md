# HLRL

HLRL is High Level Reinforcement Learning, a library that implements many state of the art algorithms, and makes implementing your own a breeze. There is support for any generic backend library.

<br />
<br />

## Contents
- [Installation](#installation)
- [Code Structure](#code-structure)
    - [Agents](#agents)
    - [Algorithms](#algorithms)
    - [Experience Replay](#experience-replay)
    - [Environments](#environments)
    - [Policies](#policies)
- [Concepts](#concepts)
    - [Wrappers](#wrappers)
    - [Experiences](#experiences)
- [Examples](#examples)
    - [Implemented Examples](#implemented-examples)

<br />

## Installation

Installation is done by cloning the git repository and installing using [`setup.py`](https://github.com/Chainso/HLRL/tree/master/setup.py).

```
git clone https://github.com/Chainso/HLRL
cd HLRL
python setup.py install
```

<br />

## Code Structure

[`hlrl.core`](https://github.com/Chainso/HLRL/tree/master/hlrl/core) contains common modules that are agnostic to any particular framework. [`hlrl.torch`](https://github.com/Chainso/HLRL/tree/master/hlrl/torch) is for modules that are implemented using the [PyTorch](https://pytorch.org/) backend.

### Agents

[`hlrl.core.agents`](https://github.com/Chainso/HLRL/tree/master/hlrl/core/agents) packages contain agents that interact with the environment and train models.

### Algorithms

[`hlrl.core.algos`](https://github.com/Chainso/HLRL/tree/master/hlrl/core/algos) contain the logic for the inference and training of reinforcement learning algorithms.

### Experience Replay

[`hlrl.core.experience_replay`](https://github.com/Chainso/HLRL/tree/master/hlrl/core/experience_replay) are the storage components for off-policy algorithms.

### Environments

[`hlrl.core.envs`](https://github.com/Chainso/HLRL/tree/master/hlrl/core/envs) contains the base environment and wrappers for common environment types.

### Policies

The [`hlrl.torch.policies`](https://github.com/Chainso/HLRL/tree/master/hlrl/torch/policies) package contains multi-layer generalizations of single layers and common networks such as Gaussians. This is used to quickly spin up a model without needing to subclass `nn.Module` yourself.

<br />

## Concepts

### Wrappers

The base wrapper is implemented in [`hlrl.core.common.wrappers`](https://github.com/Chainso/HLRL/tree/master/hlrl/core/common/wrappers). Wrappers are used to add additional functionality to existing classes, or to change existing functionality. Functionally, wrapping a class creates prototypal inheritance, allowing for wrappers to work on any class. This creates a very flexible container that allows you to swap out and modifiy algorithms and agents by simply wrapping it with your desired class.

### Experiences

Experiences are passed between modules as a dictionaries. This allows you to add to additional values to experiences without affecting old functionality. Combined with [wrappers](#wrappers), you can create more functionality on top of base algorithms.

<br />

## Examples

Examples are in the [`examples`](https://github.com/Chainso/HLRL/tree/master/examples) directory. They take command line arguments to configure the algorithm and will log results using TensorBoard.


### Implemented Examples

Flexible algorithms can be used with any base algorithm that supports it. Wrappers can be used with any algorithm and in combination with any number of wrappers.


| Algorithm | Flexible | Wrapper | Recurrent | Description |
|:-|:-:|:-:|:-:|:-|
| [SAC](https://arxiv.org/abs/1801.01290) | ❌ | N/A | ✅ | SAC auto temperature tuning and optional twin Q-networks, recurrent with R2D2 |
| [DQN](https://arxiv.org/abs/1312.5602) | ❌ | N/A | ✅ | DQN with Rainbow features excluding noisy networks, dueling architecture and C51, recurrent with R2D2 |
| [IQN](https://arxiv.org/abs/1806.06923) | ❌ | N/A | ✅ | IQN with Rainbow features excluding noisy networks, recurrent R2D2 |
| [RND](https://arxiv.org/abs/1810.12894) | ✅ | ✅ | N/A | RND excluding state normalization |
| [MunchausenRL](https://arxiv.org/abs/2007.14430) | ✅ | ✅ | N/A | MunchausenRL as seen in the literature |
| [Ape-X](https://arxiv.org/abs/1803.00933) | ✅ | ❌ | N/A | Ape-X for multi-core machines with a single model shared across agents |
| [R2D2](https://openreview.net/forum?id=r1lyTjAqYX) | ✅ | ❌ | N/A | R2D2 with hidden state storing and burning in |
