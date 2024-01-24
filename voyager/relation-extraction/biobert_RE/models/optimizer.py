# variable learning rate
class VarLROptim:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, warmup, factor, init_step):
        self.optimizer = optimizer
        self.step_count = init_step
        self.warmup = warmup
        self.factor = factor
        
    def step(self):
        "Update parameters and rate"
        self.step_count += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def rate(self):
        # min(decay, linear warmup)
        return self.factor * min(self.step_count ** (-0.5), self.step_count * self.warmup ** (-1.5))