import numpy as np

from activation import Softmax

class Layer:
    """Neural network layer class"""
    def __init__(self, input_size, output_size, activation, initializer):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        self.weights = initializer((input_size, output_size))
        self.biases = initializer((1, output_size))
        
        self.weights_grad = None
        self.biases_grad = None
        
        self.input = None
        self.output_before_activation = None

    def forward(self, x):
        """Forward pass through the layer"""
        # Menyimpan input untuk digunakan dalam backpropagation
        self.input = x
        # Menghitung output = input * weights + biases
        self.output_before_activation = np.dot(x, self.weights) + self.biases
        # Menerapkan fungsi aktivasi pada output
        return self.activation(self.output_before_activation)

    def backward(self, grad_output):
        """Backward pass through the layer"""
        
        # Penanganan khusus untuk Softmax karena derivatifnya sudah dihitung dalam CrossEntropy
        if isinstance(self.activation, Softmax):
            grad_input = grad_output
        else:
            # Menghitung gradien dengan aturan rantai: grad_output * derivative_activation
            grad_input = grad_output * self.activation.derivative(self.output_before_activation)
        
        # Menghitung gradien untuk weights: input.T * grad_input
        self.weights_grad = np.dot(self.input.T, grad_input)
        # Menghitung gradien untuk biases: sum(grad_input)
        self.biases_grad = np.sum(grad_input, axis=0, keepdims=True)
        
        # Menghitung gradien untuk layer sebelumnya: grad_input * weights.T
        grad_prev = np.dot(grad_input, self.weights.T)
        
        return grad_prev

    def update_weights(self, learning_rate):
        """Update weights using gradient descent"""
        # Memperbarui bobot dan bias menggunakan metode gradient descent
        # weights_baru = weights_lama - learning_rate * gradien
        self.weights -= learning_rate * self.weights_grad
        self.biases -= learning_rate * self.biases_grad
