import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

class MetaLoader(object):
    def __init__(self, loaders, accum_steps=1):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        for n, l in loaders.items():
            if isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n]*r)

        self.accum_steps = accum_steps
        self.step = 0

    def __iter__(self):
        task = self.sampling_pools[0]
        while True:
            if self.step % self.accum_steps == 0:
                task = random.choice(self.sampling_pools)
            self.step += 1
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_

            yield task, batch
