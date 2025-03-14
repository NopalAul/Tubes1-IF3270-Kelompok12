import numpy as np

class Loss:
    def __call__ (self, y_true, y_pred):
        pass

    def derivative(self, y_true, y_pred):
        pass

class MeanSquaredError(Loss):
    # MSE: 1/n * sum((y_true - y_pred)^2)
    def __call__(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    
    def derivative(self, y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size
    
class BinaryCrossEntropy(Loss):
    # BCE: -1/n * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    def __call__(self, y_true, y_pred):
        eps = 1e-15 # buat menghindari log(0) dan overflow
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def derivative(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.size
    
class CategoricalCrossEntropy(Loss):
    # CCE: -1/n * sum(y_true * log(y_pred))
    def __call__(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred)) # nanti periksa lagi
    
    def derivative(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -y_true / y_pred / y_true.size
    