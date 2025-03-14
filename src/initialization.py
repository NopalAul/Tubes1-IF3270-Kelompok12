import numpy as np

class Initialization:
    def __call__(self, shape):
        pass

class ZeroInitialization(Initialization):
    def __call__(self, shape):
        return np.zeros(shape)
    
class UniformInitialization(Initialization):
    # parameter lower dan upper bound, serta seed
    def __init__(self, low=0.0, high=1.0, seed=None):
        self.low = low
        self.high = high
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, shape):
        return self.rng.uniform(self.low, self.high, shape)
    
class NormalInitialization(Initialization):
    # parameter mean dan variance, serta seed
    def __init__(self, mean=0.0, variance=1.0, seed=None):
        self.mean = mean
        self.variance = variance
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, shape):
        return self.rng.normal(self.mean, np.sqrt(self.variance), shape)