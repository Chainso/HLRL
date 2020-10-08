import types

from multiprocessing.managers import Server, SharedMemoryServer, NamespaceProxy

from hlrl.core.common.wrappers import MethodWrapper

from hlrl.torch.experience_replay import TorchPER
def create(self, c, typeid, /, *args, **kwargs):
    if hasattr(self.registry[typeid][-1], "_shared_memory_proxy"):
        kwargs['shared_memory_context'] = self.shared_memory_context
    return Server.create(self, c, typeid, *args, **kwargs)

    SharedMemoryServer.create = create

def get_experience(replay):
    while len(replay) == 0:
        pass

    sample = replay.sample(2)
    print(sample)

def Proxy(target):
    dic = {'types': types}
    exec('''def __getattr__(self, key):
        result = self._callmethod('__getattribute__', (key,))
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                self._callmethod(key, args)
            return wrapper
        return result''', dic)
    proxyName = target.__name__ + "Proxy"
    ProxyType = type(proxyName, (NamespaceProxy,), dic)
    ProxyType._exposed_ = tuple(dir(target))
    return ProxyType

if __name__ == "__main__":
    import multiprocessing as mp
    import torch

    
    from multiprocessing.managers import SharedMemoryManager

    # Monkeypatching shared memory server bug since the fix is in python 3.9
    SharedMemoryServer.create = create
    exposed = tuple(dir(TorchPER))

    TorchPERProxy = Proxy(TorchPER)

    SharedMemoryManager.register("TorchPER", TorchPER, TorchPERProxy)

    with SharedMemoryManager() as smm:
        replay = smm.TorchPER(10000, 0.6, 0.4, 0.001, 1e-4)
        print(dir(replay))
        replay.add
        #recv_proc = mp.Process(target=get_experience, args=(replay,))
        #recv_proc.start()

        for i in range(1, 5):
            all_states = torch.ones((2, 2), device="cuda") * i
            states = all_states[:-1]
            next_states = all_states[1:]
            
            actions = torch.ones((1, 4), device="cuda") * 0.5 * i
            rewards = torch.ones((1, 1), device="cuda") * 0.1 * i

            q_vals = torch.ones((1, 1), device="cuda") * 0.3 * i
            target_q_vals = rewards + torch.ones((1, 1), device="cuda") * 0.8 * i
            experience = {
                "state": states,
                "action": actions,
                "reward": rewards,
                "next_state": next_states,
                "q_val": q_vals,
                "target_q_val": target_q_vals
            }

            replay.add(experience)

        recv_proc.join()

