import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# =============================================================================
# SHARED MODEL DEFINITIONS FOR PICKLE COMPATIBILITY
# =============================================================================

# =============================================================================
# APPROACH 1: OBJECT-ORIENTED ARCHITECTURE
# =============================================================================
class ActivationFunction:
    def forward(self, x):
        raise NotImplementedError
    def backward(self, x):
        raise NotImplementedError

class ReLU(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)
    def backward(self, x):
        return (x > 0).astype(float)

class MeanSquaredError:
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.last_input = None
        self.last_z = None
        self.last_a = None
    
    def forward(self, x):
        self.last_input = x
        self.last_z = np.dot(x, self.weights) + self.biases
        self.last_a = self.activation.forward(self.last_z)
        return self.last_a
    
    def backward(self, grad_output):
        grad_z = grad_output * self.activation.backward(self.last_z)
        self.grad_weights = np.dot(self.last_input.T, grad_z)
        self.grad_biases = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input
    
    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

class NeuralNetwork:
    def __init__(self, layers, loss_function, learning_rate=0.001):
        self.layers = layers
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.loss_history = []
    
    def forward(self, x):
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self, y_true, y_pred):
        grad = self.loss_function.backward(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update_weights(self):
        for layer in self.layers:
            layer.update_weights(self.learning_rate)
    
    def train(self, X, y, epochs=1000, print_every=100):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = self.loss_function.forward(y, predictions)
            self.loss_history.append(loss)
            self.backward(y, predictions)
            self.update_weights()
            if epoch % print_every == 0:
                print(f"OOP Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return self.forward(X)

# =============================================================================
# APPROACH 3: VECTORIZED IMPLEMENTATION
# =============================================================================
class VectorizedNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1
        
        self.weights = {}
        self.biases = {}
        
        for i in range(self.num_layers):
            self.weights[i] = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            self.biases[i] = np.zeros((1, layer_sizes[i+1]))
        
        self.cache = {}
    
    def relu_vectorized(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative_vectorized(self, Z):
        return (Z > 0).astype(np.float32)
    
    def forward_vectorized(self, X):
        A = X.astype(np.float32)
        self.cache['A0'] = A
        
        for i in range(self.num_layers):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.cache[f'Z{i}'] = Z
            
            if i < self.num_layers - 1:
                A = self.relu_vectorized(Z)
            else:
                A = Z
            
            self.cache[f'A{i+1}'] = A
        
        return A
    
    def backward_vectorized(self, y_true, y_pred):
        m = y_true.shape[0]
        gradients = {}
        
        dZ = (y_pred - y_true) / m
        
        for i in reversed(range(self.num_layers)):
            gradients[f'dW{i}'] = np.dot(self.cache[f'A{i}'].T, dZ)
            gradients[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True)
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.relu_derivative_vectorized(self.cache[f'Z{i-1}'])
        
        return gradients
    
    def update_weights_vectorized(self, gradients):
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * gradients[f'dW{i}']
            self.biases[i] -= self.learning_rate * gradients[f'db{i}']
    
    def train_batch_vectorized(self, X, y, batch_size=32, epochs=1000):
        loss_history = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                y_pred = self.forward_vectorized(X_batch)
                batch_loss = np.mean((y_batch - y_pred) ** 2)
                epoch_loss += batch_loss
                
                gradients = self.backward_vectorized(y_batch, y_pred)
                self.update_weights_vectorized(gradients)
            
            avg_loss = epoch_loss / (n_samples // batch_size)
            loss_history.append(avg_loss)
            
            if epoch % 200 == 0:
                print(f"Vectorized Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return loss_history
    
    def predict_vectorized(self, X):
        return self.forward_vectorized(X)

# =============================================================================
# APPROACH 4: SEQUENTIAL
# =============================================================================
class SequentialLayer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_type = activation
        
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        
        self.z = None
        self.a = None
        self.input_data = None
    
    def forward(self, x):
        self.input_data = x
        self.z = np.dot(x, self.W) + self.b
        
        if self.activation_type == 'relu':
            self.a = np.maximum(0, self.z)
        elif self.activation_type == 'linear':
            self.a = self.z
        
        return self.a
    
    def backward(self, grad_output):
        if self.activation_type == 'relu':
            grad_z = grad_output * (self.z > 0).astype(float)
        elif self.activation_type == 'linear':
            grad_z = grad_output
        
        self.dW = np.dot(self.input_data.T, grad_z)
        self.db = np.sum(grad_z, axis=0, keepdims=True)
        
        grad_input = np.dot(grad_z, self.W.T)
        return grad_input
    
    def update_parameters(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

class SequentialNetwork:
    def __init__(self):
        self.layers = []
        self.loss_history = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        current_output = x
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output
    
    def backward(self, loss_gradient):
        current_gradient = loss_gradient
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)
    
    def update_all_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)
    
    def train_sequential(self, X, y, epochs=1000, learning_rate=0.001):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = np.mean((y - predictions) ** 2)
            self.loss_history.append(loss)
            
            loss_grad = 2 * (predictions - y) / y.shape[0]
            self.backward(loss_grad)
            self.update_all_parameters(learning_rate)
            
            if epoch % 200 == 0:
                print(f"Sequential Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict_sequential(self, X):
        return self.forward(X)

# =============================================================================
# APPROACH 5: GRADIENT-FIRST
# =============================================================================
class GradientFirstNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.weights = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        }
        self.biases = {
            'b1': np.zeros((1, hidden_size)),
            'b2': np.zeros((1, output_size))
        }
        self.learning_rate = learning_rate
    
    def predict_gradient_first(self, X):
        # Simple forward pass for prediction
        A1 = X
        Z2 = np.dot(A1, self.weights['W1']) + self.biases['b1']
        A2 = np.maximum(0, Z2)
        Z3 = np.dot(A2, self.weights['W2']) + self.biases['b2']
        A3 = Z3
        return A3

# =============================================================================
# APPROACH 6: MODULAR SYSTEM
# =============================================================================
class ModularNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.learning_rate = learning_rate
        self.parameters = self.initialize_parameters()
        self.loss_history = []
    
    def initialize_parameters(self):
        parameters = {}
        for i in range(len(self.layer_sizes) - 1):
            parameters[f'W{i}'] = np.random.randn(
                self.layer_sizes[i], self.layer_sizes[i+1]
            ) * np.sqrt(2.0 / self.layer_sizes[i])
            parameters[f'b{i}'] = np.zeros((1, self.layer_sizes[i+1]))
        return parameters
    
    def predict(self, X):
        A = X
        num_layers = len(self.layer_sizes) - 1
        
        for i in range(num_layers):
            Z = np.dot(A, self.parameters[f'W{i}']) + self.parameters[f'b{i}']
            if i < num_layers - 1:
                A = np.maximum(0, Z)
            else:
                A = Z
        return A

# =============================================================================
# APPROACH 7: INCREMENTAL
# =============================================================================
class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        self.loss_history = []
        
        self.z1 = None
        self.a1 = None
        self.z2 = None
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
