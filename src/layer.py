import numpy as np

from activation import Softmax
from initialization import HeInitialization
from rmsnorm import RMSNorm  # Import modul baru

class Layer:
    """Neural network layer class"""
    def __init__(self, input_size, output_size, activation, initializer, normalization=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.normalization = normalization(output_size) if normalization else None

        # default
        self.weight_bias_initializer = initializer or HeInitialization()
        
        self.weights = self.weight_bias_initializer((input_size, output_size))
        self.biases = self.weight_bias_initializer((1, output_size))
        
        self.weights_grad = np.zeros_like(self.weights)
        self.biases_grad = np.zeros_like(self.biases)
        
        self.input = None
        self.output_before_activation = None
        self.output = None

    def forward(self, x):
        """Forward pass through the layer"""
        # Simpan input untuk digunakan dalam backpropagation
        self.input = x
        # Ngitung output = input * weights + biases
        self.output_before_activation = np.dot(x, self.weights) + self.biases
        if self.normalization:
            self.output_before_activation = self.normalization.forward(self.output_before_activation)
        return self.activation(self.output_before_activation)


    def backward(self, grad_output):
        """Backward pass through the layer"""
        
        # Softmax ditangani khusus sebab derivatifnya udah dihitung di CrossEntropy
        if isinstance(self.activation, Softmax):
            grad_input = grad_output
        else:
            # Hitung gradien dengan aturan rantai
            grad_input = grad_output * self.activation.derivative(self.output_before_activation)
        
        if self.normalization:
            grad_input = self.normalization.backward(grad_input)

        # Hitung gradien untuk weights: input.T * grad_input
        self.weights_grad = np.dot(self.input.T, grad_input)
        # Hitung gradien untuk bias: sum(grad_input)
        self.biases_grad = np.sum(grad_input, axis=0, keepdims=True)
        
        # Hitung gradien untuk layer sebelumnya: grad_input * weights.T
        grad_prev = np.dot(grad_input, self.weights.T)
        
        return grad_prev

    def update_weights(self, learning_rate, l1_lambda=0.0, l2_lambda=0.0):
        # Regularisasi L1: Tambahin nilai sign ke gradien 
        if l1_lambda > 0:
            # Rumus Regularisasi L1
            self.weights_grad += l1_lambda * np.sign(self.weights)

        # Regularisasi L2: Tambahin faktor bobot ke gradien   
        if l2_lambda > 0:
            # Rumus Regularisasi L2
            self.weights_grad += l2_lambda * 2 * self.weights
        
        # Update bobot dan bias dengan gradient descent: weights_baru = weights_lama - learning_rate * gradien
        self.weights -= learning_rate * self.weights_grad
        self.biases -= learning_rate * self.biases_grad

        if self.normalization:
            self.normalization.update_weights(learning_rate)

