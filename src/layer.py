class Layer:
    def __init__(self, input_size, output_size, initialization, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        # nanti lanjut

    def forward(self, x):
        # forward pass di layer
        pass

    def backward(self, gradient_output):
        # backward pass di layer
        pass

    def update_weights(self, learning_rate):
        # update bobot
        pass