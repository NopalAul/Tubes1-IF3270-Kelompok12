import numpy as np

class RMSNorm:
    def __init__(self, dim, eps=1e-8):
        self.eps = eps
        self.gamma = np.ones((1, dim))  
        self.grad_gamma = np.zeros_like(self.gamma)
        self.input = None
        self.norm = None

    def forward(self, x):
        self.input = x
        self.norm = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x / self.norm)

    def backward(self, grad_output):
        x = self.input
        N = x.shape[-1]
        norm = self.norm

        dx_norm = grad_output * self.gamma
        dnorm = -np.sum(dx_norm * x / (norm ** 2), axis=-1, keepdims=True)
        dx = dx_norm / norm + dnorm * x / (N * norm)

        self.grad_gamma = np.sum(grad_output * (x / norm), axis=0, keepdims=True)
        return dx

    def update_weights(self, lr):
        self.gamma -= lr * self.grad_gamma
