# main model FFNN
import pickle

class FFNN:
    def __init__(self, layer_sizes, initialization, activations, loss):
        self.layer_sizes = layer_sizes
        self.loss = loss

        # nanti bikin list layernya..

    def forward(self, x):
        # forward pass di seluruh layer
        pass

    def backward(self, x, y):
        # backward pass di seluruh layer
        pass

    def update_weights(self, learning_rate):
        # update bobot di seluruh layer
        pass

    def train(self, x_train, y_train, x_y_val=None, batch_size=32, learning_rate=0.01, epochs=10, verbose=1):
        # Batch size, Learning rate, Jumlah epoch, Verbose
        # x_y_val buat validation loss
        pass

    def predict(self, x):
        # predict buat data input
        pass
    
    # Instance model memiliki method untuk save dan load
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    # Instance model memiliki method untuk menampilkan model berupa struktur jaringan beserta bobot dan gradien bobot tiap neuron dalam bentuk graf. (Format graf  dibebaskan)
    # Instance model memiliki method untuk menampilkan distribusi bobot dari tiap layer.
    # Menerima masukan berupa list of integer (bisa disesuaikan ke struktur data lain sesuai kebutuhan) yang menyatakan layer mana saja yang distribusinya akan di-plot
    # Instance model memiliki method untuk menampilkan distribusi gradien bobot dari tiap layer.
    # Menerima masukan berupa list of integer (bisa disesuaikan ke struktur data lain sesuai kebutuhan) yang menyatakan layer mana saja yang distribusinya akan di-plot



