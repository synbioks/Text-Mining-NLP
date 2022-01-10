# variable learning rate
class NoamOptim:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, warmup, factor):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * min(step ** (-0.5), step * self.warmup ** (-1.5))