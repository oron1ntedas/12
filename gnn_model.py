import numpy as np
from typing import Dict, List, Tuple
import json

class GNN:
    """Graph Neural Network for processing preprocessed 3D mesh data."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        """Initialize GNN with specified dimensions and number of layers."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        for i in range(num_layers):
            if i == 0:
                w = np.random.randn(input_dim, hidden_dim) * 0.01
            else:
                w = np.random.randn(hidden_dim, hidden_dim) * 0.01
            b = np.zeros((1, hidden_dim))
            self.weights.append(w)
            self.biases.append(b)
        
        # Output layer
        self.weights.append(np.random.randn(hidden_dim, output_dim) * 0.01)
        self.biases.append(np.zeros((1, output_dim)))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def message_passing(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Perform message passing using adjacency matrix."""
        # Aggregate messages from neighbors (sum of neighbor features)
        messages = np.dot(adjacency_matrix, node_features)
        return messages
    
    def forward(self, graph_data: Dict) -> np.ndarray:
        """Forward pass through the GNN."""
        node_features = graph_data["node_features"].copy()
        adjacency_matrix = graph_data["adjacency_matrix"]
        
        # Process through each layer
        for i in range(self.num_layers):
            # Message passing
            messages = self.message_passing(node_features, adjacency_matrix)
            # Update node features: combine messages and current features
            node_features = self.relu(np.dot(messages, self.weights[i]) + self.biases[i])
        
        # Final output layer
        node_features = np.dot(node_features, self.weights[-1]) + self.biases[-1]
        return node_features
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean squared error loss."""
        return np.mean((predictions - targets) ** 2)
    
    def compute_gradients(self, predictions: np.ndarray, targets: np.ndarray, 
                         node_features: np.ndarray, adjacency_matrix: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute gradients for weights and biases using numerical differentiation."""
        grad_weights = []
        grad_biases = []
        epsilon = 1e-6
        
        # Compute gradients for each layer
        for layer in range(self.num_layers + 1):
            grad_w = np.zeros_like(self.weights[layer])
            grad_b = np.zeros_like(self.biases[layer])
            
            # Numerical gradient for weights
            for i in range(self.weights[layer].shape[0]):
                for j in range(self.weights[layer].shape[1]):
                    original = self.weights[layer][i, j]
                    self.weights[layer][i, j] += epsilon
                    pred_plus = self.forward({
                        "node_features": node_features,
                        "adjacency_matrix": adjacency_matrix
                    })
                    loss_plus = self.compute_loss(pred_plus, targets)
                    
                    self.weights[layer][i, j] = original - epsilon
                    pred_minus = self.forward({
                        "node_features": node_features,
                        "adjacency_matrix": adjacency_matrix
                    })
                    loss_minus = self.compute_loss(pred_minus, targets)
                    
                    grad_w[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                    self.weights[layer][i, j] = original
            
            # Numerical gradient for biases
            for i in range(self.biases[layer].shape[1]):
                original = self.biases[layer][0, i]
                self.biases[layer][0, i] += epsilon
                pred_plus = self.forward({
                    "node_features": node_features,
                    "adjacency_matrix": adjacency_matrix
                })
                loss_plus = self.compute_loss(pred_plus, targets)
                
                self.biases[layer][0, i] = original - epsilon
                pred_minus = self.forward({
                    "node_features": node_features,
                    "adjacency_matrix": adjacency_matrix
                })
                loss_minus = self.compute_loss(pred_minus, targets)
                
                grad_b[0, i] = (loss_plus - loss_minus) / (2 * epsilon)
                self.biases[layer][0, i] = original
            
            grad_weights.append(grad_w)
            grad_biases.append(grad_b)
        
        return grad_weights, grad_biases
    
    def train(self, graph_data: Dict, targets: np.ndarray, epochs: int, learning_rate: float, progress_callback=None) -> List[float]:
        """Train the GNN using gradient descent."""
        losses = []
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(graph_data)
            loss = self.compute_loss(predictions, targets)
            losses.append(loss)
            
            # Compute gradients
            grad_weights, grad_biases = self.compute_gradients(
                predictions, targets, graph_data["node_features"], graph_data["adjacency_matrix"]
            )
            
            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * grad_weights[i]
                self.biases[i] -= learning_rate * grad_biases[i]
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
            
            if progress_callback:
                progress = (epoch + 1) / epochs * 100
                progress_callback(progress, f"GNN Training: Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

        return losses

    def save_weights(self, file_path):
        """Сохраняет веса и смещения модели в JSON файл."""
        # Собираем все параметры в один словарь.
        # .tolist() конвертирует numpy массивы в обычные списки для сохранения в JSON.
        model_state = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(file_path, 'w') as f:
            json.dump(model_state, f, indent=4)
        print(f"GNN weights saved to {file_path}")

    def load_weights(self, file_path):
        """Загружает веса и смещения модели из JSON файла."""
        with open(file_path, 'r') as f:
            model_state = json.load(f)
        
        # Загружаем параметры, преобразуя их обратно в numpy массивы.
        self.weights = [np.array(w) for w in model_state['weights']]
        self.biases = [np.array(b) for b in model_state['biases']]
        print(f"GNN weights loaded from {file_path}")

def example_usage():
    """Example usage of GNN with data from DataPreprocessor."""
    # Load preprocessed data (assumes files exist from DataPreprocessor)
    with open("preprocessed_data_defective.json", "r") as f:
        defective_data = json.load(f)
    
    with open("preprocessed_data_reference.json", "r") as f:
        reference_data = json.load(f)
    
    # Convert data to numpy arrays
    defective_data = {
        "node_features": np.array(defective_data["node_features"]),
        "adjacency_matrix": np.array(defective_data["adjacency_matrix"])
    }
    targets = np.array(reference_data["node_features"])  # Use reference node features as targets
    
    # Initialize and train GNN
    gnn = GNN(input_dim=defective_data["node_features"].shape[1], 
              hidden_dim=16, 
              output_dim=targets.shape[1], 
              num_layers=2)
    
    losses = gnn.train(defective_data, targets, epochs=100, learning_rate=0.01)
    
    # Save final predictions for debugging
    predictions = gnn.forward(defective_data)
    np.save("gnn_predictions.npy", predictions)
    
    print(f"Final loss: {losses[-1]:.6f}")

if __name__ == "__main__":
    example_usage()

