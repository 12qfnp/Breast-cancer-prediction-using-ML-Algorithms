import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :3]  # Take first 3 features
y = (iris.target != 0).astype(int)  # Convert to binary classification (Setosa vs Non-Setosa)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network Architecture
n_input = 3
n_hidden = 4
n_output = 1

# Activation functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    return A * (1 - A)

def tanh(Z):
    return np.tanh(Z)

def tanh_derivative(A):
    return 1 - A ** 2

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(A):
    return (A > 0).astype(float)

def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)

def leaky_relu_derivative(A, alpha=0.01):
    return np.where(A > 0, 1, alpha)

# Initialize weights
np.random.seed(42)
def initialize_weights(init_type='random', scale=0.01):
    if init_type == 'zero':
        W1 = np.zeros((n_hidden, n_input))
        W2 = np.zeros((n_output, n_hidden))
    elif init_type == 'large':
        W1 = np.random.rand(n_hidden, n_input) * 100
        W2 = np.random.rand(n_output, n_hidden) * 100
    else:  # Small random values
        W1 = np.random.randn(n_hidden, n_input) * scale
        W2 = np.random.randn(n_output, n_hidden) * scale
    
    b1 = np.zeros((n_hidden, 1))
    b2 = np.zeros((n_output, 1))
    return W1, b1, W2, b2

# Forward and backward propagation
def train_network(activation_hidden, activation_output, init_type='random', scale=0.01, epochs=100, lr=0.01):
    W1, b1, W2, b2 = initialize_weights(init_type, scale)
    loss_history = []
    accuracy_history = []
    
    activation_funcs = {'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu, 'leaky_relu': leaky_relu}
    activation_derivatives = {'sigmoid': sigmoid_derivative, 'tanh': tanh_derivative, 'relu': relu_derivative, 'leaky_relu': leaky_relu_derivative}
    
    act_hidden = activation_funcs[activation_hidden]
    d_act_hidden = activation_derivatives[activation_hidden]
    act_output = activation_funcs[activation_output]
    d_act_output = activation_derivatives[activation_output]
    
    for epoch in range(epochs):
        # Forward Propagation
        Z1 = np.dot(W1, X_train.T) + b1
        A1 = act_hidden(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = act_output(Z2)
        
        # Compute loss
        loss = np.mean(-(y_train * np.log(A2 + 1e-8) + (1 - y_train) * np.log(1 - A2 + 1e-8)))
        loss_history.append(loss)
        
        # Compute accuracy
        preds = (A2 > 0.5).astype(int)
        acc = np.mean(preds == y_train)
        accuracy_history.append(acc)
        
        # Backpropagation
        dZ2 = A2 - y_train.reshape(1, -1)
        dW2 = np.dot(dZ2, A1.T) / len(y_train)
        db2 = np.sum(dZ2, axis=1, keepdims=True) / len(y_train)
        
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * d_act_hidden(A1)
        dW1 = np.dot(dZ1, X_train) / len(y_train)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / len(y_train)
        
        # Gradient Descent
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        
        # Print every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()
    
    print(f"Final Accuracy: {acc:.4f}")
    return acc

# Running different variations
variations = [
    ('sigmoid', 'sigmoid', 'random', 0.01),
    ('relu', 'sigmoid', 'random', 0.01),
    ('tanh', 'sigmoid', 'random', 0.01),
    ('leaky_relu', 'sigmoid', 'random', 0.01),
    ('relu', 'sigmoid', 'zero', 0.01),
    ('relu', 'sigmoid', 'large', 100)
]

for hidden_act, output_act, init_type, scale in variations:
    print(f"\nTesting {hidden_act} hidden layer, {output_act} output layer, {init_type} weights\n")
    train_network(hidden_act, output_act, init_type, scale)