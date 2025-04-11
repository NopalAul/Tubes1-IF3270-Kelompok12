# main model FFNN
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm

from layer import Layer
from activation import Softmax
from loss import CategoricalCrossEntropy
from initialization import HeInitialization

class FFNN:
    def __init__(self, layer_sizes, initializations, activations, loss, normalization=[None, None, None]):
        self.layer_sizes = layer_sizes
        self.loss = loss

        # nanti bikin list layernya..
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            weight_bias_initialization = initializations[i] if initializations[i] else HeInitialization() # default

            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i], weight_bias_initialization, normalization[i]))

        # histori proses pelatihan
        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    def forward(self, x):
        # forward pass di seluruh layer
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, x, y):
        # backward pass di seluruh layer
        y_pred = self.forward(x)
        
        # gradien dr loss function 
        grad = self.loss.derivative(y, y_pred)
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self, learning_rate, l1_lambda=0.0, l2_lambda=0.0):
        # update bobot di seluruh layer
        for layer in self.layers:
            layer.update_weights(learning_rate, l1_lambda, l2_lambda)


    def train(self, x_train, y_train, x_y_val=None, batch_size=32, learning_rate=0.01, epochs=10, verbose=1, l1_lambda=0.0, l2_lambda=0.0):
        # Batch size, Learning rate, Jumlah epoch, Verbose
        # x_y_val buat validation loss

        n_samples = x_train.shape[0]
        
        for epoch in range(epochs):
            if verbose == 1:
                print(f"Epoch {epoch+1}/{epochs}")
                pbar = tqdm(total=n_samples)

            # shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            # mini batch (efisiensi?)
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred = self.forward(x_batch)
                batch_loss = self.loss(y_batch, y_pred)
                epoch_loss += batch_loss * len(x_batch)

                self.backward(x_batch, y_batch)
                self.update_weights(learning_rate, l1_lambda, l2_lambda)

                if verbose == 1:
                    pbar.update(len(x_batch))
                
            epoch_loss /= n_samples # rata rata training loss
            self.history['train_loss'].append(epoch_loss)

            if x_y_val is not None:
                x_val, y_val = x_y_val
                y_val_pred = self.forward(x_val)
                val_loss = self.loss(y_val, y_val_pred)
                self.history['val_loss'].append(val_loss)

                if verbose == 1:
                    pbar.set_postfix({"train_loss": f"{epoch_loss:.4f}", "val_loss": f"{val_loss:.4f}"})
            elif verbose == 1:
                pbar.set_postfix({"train_loss": f"{epoch_loss:.4f}"})

            if verbose == 1:
                pbar.close()

        return self.history

    def predict(self, x):
        # predict buat data input
        return self.forward(x)
    
    # Instance model memiliki method untuk save dan load
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def plot_model(self):
        G = nx.DiGraph()
        
        layer_nodes = []
        
        # untuk input layer
        input_nodes = []
        for i in range(self.layer_sizes[0]):
            node_id = f"Input_{i}"
            G.add_node(node_id, pos=(0, -i), layer=0, neuron=i)
            input_nodes.append(node_id)
        layer_nodes.append(input_nodes)
        
        # untuk hidden dan output layer
        for l, layer_size in enumerate(self.layer_sizes[1:], 1):
            layer_nodes_list = []
            for i in range(layer_size):
                node_id = f"Layer_{l}_{i}"
                G.add_node(node_id, pos=(l, -i), layer=l, neuron=i)
                layer_nodes_list.append(node_id)
            layer_nodes.append(layer_nodes_list)
        
        # edges (label bobot dan gradien)
        weights = []
        gradients = []
        for l in range(len(self.layers)):
            layer = self.layers[l]
            for i, prev_node in enumerate(layer_nodes[l]):
                for j, next_node in enumerate(layer_nodes[l+1]):
                    weight = layer.weights[i, j]
                    grad = layer.weights_grad[i, j]
                    G.add_edge(prev_node, next_node, weight=weight, gradient=grad)
                    weights.append(abs(weight))
                    gradients.append(abs(grad))
        
        weights_norm = (np.array(weights) - np.min(weights)) / (np.max(weights) - np.min(weights))
        gradients_norm = (np.array(gradients) - np.min(gradients)) / (np.max(gradients) - np.min(gradients))
        
        fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
        plt.subplots_adjust(bottom=0.1)
        
        pos = nx.get_node_attributes(G, 'pos')
        
        edge_colors = cm.coolwarm(weights_norm)
        
        layer_colors = plt.cm.Set3(np.linspace(0, 1, len(self.layer_sizes)))
        node_colors = [layer_colors[nx.get_node_attributes(G, 'layer')[node]] for node in G.nodes()]
        
        nodes = nx.draw_networkx_nodes(G, pos, 
                                        ax=ax,
                                        node_color=node_colors, 
                                        node_size=300, 
                                        alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        edges = nx.draw_networkx_edges(G, pos, 
                                        ax=ax,
                                        edge_color=edge_colors, 
                                        width=1.5, 
                                        arrows=True, 
                                        arrowsize=10)
        
        edge_labels = {(u, v): f"W:{G[u][v]['weight']:.2f}\nG:{G[u][v]['gradient']:.2f}" 
                for u, v in G.edges()}

        nx.draw_networkx_edge_labels(G, pos, 
                                    ax=ax,
                                    edge_labels=edge_labels, 
                                    font_size=6)
        
        ax.set_title("FFNN Model Structure")
        ax.axis('off')
        
        # zoom
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            
            xdata = event.xdata
            ydata = event.ydata
            
            if event.button == 'up':
                scale_factor = 0.5
            elif event.button == 'down':
                scale_factor = 2
            else:
                scale_factor = 1
            
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
            
            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            plt.draw()
        
        fig.canvas.mpl_connect('scroll_event', zoom)
        
        plt.tight_layout()
        plt.show()
    
    def plot_weight_distribution(self, layers=None):
        if layers is None:
            layers = list(range(len(self.layers)))
        
        num_layers = len(layers)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3 * num_layers))
        
        if num_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                print(f"Warning: Layer index {layer_idx} is out of range. Skipping.")
                continue
                
            layer = self.layers[layer_idx]
            weights = layer.weights.flatten()
            
            axes[i].hist(weights, bins=30)
            axes[i].set_title(f"Layer {layer_idx+1} Weight Distribution")
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_gradient_distribution(self, layers=None):
        if layers is None:
            layers = list(range(len(self.layers)))
        
        num_layers = len(layers)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3 * num_layers))
        
        if num_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                print(f"Warning: Layer index {layer_idx} is out of range. Skipping.")
                continue
                
            layer = self.layers[layer_idx]
            gradients = layer.weights_grad.flatten()
            
            axes[i].hist(gradients, bins=30)
            axes[i].set_title(f"Layer {layer_idx+1} Gradient Distribution")
            axes[i].set_xlabel("Gradient Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()



