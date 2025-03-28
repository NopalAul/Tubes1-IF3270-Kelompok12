import numpy as np

class Activation:
    def __call__(self, x): # callable object, jadi object dipanggil seperti fungsi
        pass

    def derivative(self, x):
        pass

class Linear(Activation):
    # Linear: f(x) = x
    def __call__(self, x):
        return x
    
    def derivative(self, x):
        return np.ones_like(x)
    
class ReLU(Activation):
    # ReLU: f(x) = max(0, x)
    def __call__(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x <= 0, 0, 1)

class Sigmoid(Activation):
    # Sigmoid: f(x) = 1 / (1 + e^(-x))
    def __call__(self, x):
        return 1 / (1 + np.exp(-np.clip(x, 
        -500, 500))) # np.clip biar ga overflow
    
    def derivative(self, x):
        s = self.__call__(x)
        return s * (1 - s)
    
class Tanh(Activation):
    # Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x)) atau tanh(x) (numpy)
    def __call__(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.power(np.tanh(x), 2)
    
class Softmax(Activation):
    # Softmax: f(x)_i = e^(x_i) / sum(e^(x_j)) 
    def __call__(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True) # x dishift biar ga overflow
    
    def derivative(self, x):
        # masih bingung
        return np.ones_like(x)

# Implementasi Bonus: Tambahan dua kelas aktivasi
class LeakyReLU(Activation):
    # f(x) = x jika x > 0, alpha * x jika x <= 0
    def __call__(self, x):
        alpha = 0.01
        x_clipped = np.clip(x, -50, 50)  # Biar ga overflow
        return np.maximum(alpha * x_clipped, x_clipped)

    def derivative(self, x):
        return np.where(x > 0, 1, 0.01)


class ELU(Activation):
    # f(x) = x jika x > 0, alpha * (e^x - 1) jika x <= 0
    def __call__(self, x):
        alpha = 1
        x_clipped = np.clip(x, -50, 50)  # Biar ngga overflow
        return np.where(x > 0, x, alpha * (np.exp(x_clipped) - 1))

    def derivative(self, x):
        return np.where(x > 0, 1, self.__call__(x) + 1)
