import numpy as np
from typing import Dict, List, Tuple
import json
from gnn_model import GNN

class Generator:
    """Generator for correcting vertex coordinates using GNN features and noise."""
    
    def __init__(self, input_dim: int, noise_dim: int, hidden_dim: int, output_dim: int):
        """Initialize Generator with input dimensions (GNN features + noise), hidden, and output (x, y, z)."""
        self.input_dim = input_dim + noise_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for three-layer MLP (increased depth)
        self.weights.append(np.random.randn(self.input_dim, hidden_dim) * 0.01)
        self.biases.append(np.zeros((1, hidden_dim)))
        self.weights.append(np.random.randn(hidden_dim, hidden_dim) * 0.01)  # Additional hidden layer
        self.biases.append(np.zeros((1, hidden_dim)))
        self.weights.append(np.random.randn(hidden_dim, output_dim) * 0.01)
        self.biases.append(np.zeros((1, output_dim)))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def batch_norm(self, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """Batch normalization."""
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + epsilon)
    
    def forward(self, gnn_features: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Forward pass with three layers and batch normalization."""
        x = np.concatenate([gnn_features, noise], axis=1)
        x = self.batch_norm(self.relu(np.dot(x, self.weights[0]) + self.biases[0]))
        x = self.batch_norm(self.relu(np.dot(x, self.weights[1]) + self.biases[1]))
        x = np.dot(x, self.weights[2]) + self.biases[2]
        return x

class Discriminator:
    def __init__(self, input_dim: int, hidden_dim: int, gnn_feature_dim: int = 0):
        self.input_dim = input_dim + gnn_feature_dim
        self.hidden_dim = hidden_dim
        self.weights = []
        self.biases = []
        self.weights.append(np.random.randn(self.input_dim, hidden_dim) * 0.01)
        self.biases.append(np.zeros((1, hidden_dim)))
        self.weights.append(np.random.randn(hidden_dim, hidden_dim) * 0.01)
        self.biases.append(np.zeros((1, hidden_dim)))
        self.weights.append(np.random.randn(hidden_dim, 1) * 0.01)
        self.biases.append(np.zeros((1, 1)))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def batch_norm(self, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """Batch normalization."""
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + epsilon)
    
    def forward(self, vertices: np.ndarray, gnn_features: np.ndarray = None) -> np.ndarray:
        if gnn_features is not None:
            x = np.concatenate([vertices, gnn_features], axis=1)
        else:
            x = vertices
        x = self.batch_norm(self.relu(np.dot(x, self.weights[0]) + self.biases[0]))
        x = self.batch_norm(self.relu(np.dot(x, self.weights[1]) + self.biases[1]))
        x = self.sigmoid(np.dot(x, self.weights[2]) + self.biases[2])
        return x

class GAN:
    """GAN combining Generator and Discriminator for vertex coordinate correction."""
    
    def __init__(self, gnn_input_dim: int, noise_dim: int, hidden_dim: int, coord_dim: int):
        """Initialize GAN with Generator and Discriminator."""
        self.generator = Generator(gnn_input_dim, noise_dim, hidden_dim, coord_dim)
        self.discriminator = Discriminator(coord_dim, hidden_dim, gnn_input_dim)  # Include GNN features
        self.noise_dim = noise_dim
        self.fixed_noise = None  # For debugging
    
    def compute_bce_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute binary cross-entropy loss with label smoothing."""
        epsilon = 1e-10
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        targets = np.where(targets == 1, 0.9, 0.1)  # Label smoothing
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    def compute_gradients(self, model, inputs: np.ndarray, targets: np.ndarray, is_generator: bool = False,
                         gnn_features: np.ndarray = None, noise: np.ndarray = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute numerical gradients using one-sided differentiation."""
        grad_weights = []
        grad_biases = []
        epsilon = 1e-6
        
        for layer in range(len(model.weights)):
            grad_w = np.zeros_like(model.weights[layer])
            grad_b = np.zeros_like(model.biases[layer])
            
            # Numerical gradient for weights
            for i in range(model.weights[layer].shape[0]):
                for j in range(model.weights[layer].shape[1]):
                    original = model.weights[layer][i, j]
                    model.weights[layer][i, j] += epsilon
                    if is_generator:
                        pred = model.forward(gnn_features, noise)
                        if model is self.generator:
                            pred = self.discriminator.forward(pred, gnn_features)
                    else:
                        pred = model.forward(inputs, gnn_features)
                    loss_plus = self.compute_bce_loss(pred, targets)
                    model.weights[layer][i, j] = original
                    loss_original = self.compute_bce_loss(
                        self.discriminator.forward(model.forward(gnn_features, noise), gnn_features) if is_generator
                        else model.forward(inputs, gnn_features), targets)
                    grad_w[i, j] = (loss_plus - loss_original) / epsilon
                    model.weights[layer][i, j] = original
            
            # Numerical gradient for biases
            for i in range(model.biases[layer].shape[1]):
                original = model.biases[layer][0, i]
                model.biases[layer][0, i] += epsilon
                if is_generator:
                    pred = model.forward(gnn_features, noise)
                    if model is self.generator:
                        pred = self.discriminator.forward(pred, gnn_features)
                else:
                    pred = model.forward(inputs, gnn_features)
                loss_plus = self.compute_bce_loss(pred, targets)
                model.biases[layer][0, i] = original
                loss_original = self.compute_bce_loss(
                    self.discriminator.forward(model.forward(gnn_features, noise), gnn_features) if is_generator
                    else model.forward(inputs, gnn_features), targets)
                grad_b[0, i] = (loss_plus - loss_original) / epsilon
                model.biases[layer][0, i] = original
            
            grad_weights.append(grad_w)
            grad_biases.append(grad_b)
        
        return grad_weights, grad_biases
    
    def denormalize_vertices(self, normalized_vertices: np.ndarray, min_coords: np.ndarray, range_coords: np.ndarray) -> np.ndarray:
        """Denormalize vertices to original scale."""
        return normalized_vertices * range_coords + min_coords
    
    def train(self, gnn: GNN, graph_data: Dict, real_coords: np.ndarray, min_coords: np.ndarray, 
              range_coords: np.ndarray, epochs: int, learning_rate: float, update_discriminator_freq: int = 1, progress_callback=None) -> List[float]:
        """Train the GAN using GNN features and real coordinates."""
        losses_d = []
        losses_g = []
        
        # Initialize fixed noise for debugging
        self.fixed_noise = np.random.randn(graph_data["node_features"].shape[0], self.noise_dim)
        
        # Get GNN features
        gnn_features = gnn.forward(graph_data)
        
        for epoch in range(epochs):
            # Generate noise
            noise = np.random.randn(graph_data["node_features"].shape[0], self.noise_dim)
            
            # Generate fake coordinates
            fake_coords = self.generator.forward(gnn_features, noise)
            
            # Train Discriminator (less frequently to balance)
            if epoch % update_discriminator_freq == 0:
                real_labels = np.ones((real_coords.shape[0], 1))
                fake_labels = np.zeros((fake_coords.shape[0], 1))
                
                d_real_pred = self.discriminator.forward(real_coords, gnn_features)
                d_loss_real = self.compute_bce_loss(d_real_pred, real_labels)
                d_fake_pred = self.discriminator.forward(fake_coords, gnn_features)
                d_loss_fake = self.compute_bce_loss(d_fake_pred, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) / 2
                
                grad_w_d_real, grad_b_d_real = self.compute_gradients(self.discriminator, real_coords, real_labels, gnn_features=gnn_features)
                grad_w_d_fake, grad_b_d_fake = self.compute_gradients(self.discriminator, fake_coords, fake_labels, gnn_features=gnn_features)
                grad_w_d = [(w1 + w2) / 2 for w1, w2 in zip(grad_w_d_real, grad_w_d_fake)]
                grad_b_d = [(b1 + b2) / 2 for b1, b2 in zip(grad_b_d_real, grad_b_d_fake)]
                
                for i in range(len(self.discriminator.weights)):
                    self.discriminator.weights[i] -= learning_rate * grad_w_d[i]
                    self.discriminator.biases[i] -= learning_rate * grad_b_d[i]
            else:
                d_loss = losses_d[-1] if losses_d else 0.0
            
            # Train Generator
            noise = np.random.randn(graph_data["node_features"].shape[0], self.noise_dim)
            g_labels = np.ones((real_coords.shape[0], 1))
            grad_w_g, grad_b_g = self.compute_gradients(self.generator, None, g_labels, is_generator=True,
                                                       gnn_features=gnn_features, noise=noise)
            
            for i in range(len(self.generator.weights)):
                self.generator.weights[i] -= learning_rate * grad_w_g[i]
                self.generator.biases[i] -= learning_rate * grad_b_g[i]
            
            fake_coords = self.generator.forward(gnn_features, noise)
            g_pred = self.discriminator.forward(fake_coords, gnn_features)
            g_loss = self.compute_bce_loss(g_pred, g_labels)
            
            losses_d.append(d_loss)
            losses_g.append(g_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Discriminator Loss: {d_loss:.6f}, Generator Loss: {g_loss:.6f}")
                
                # Save intermediate results with fixed noise for debugging
                fixed_coords = self.generator.forward(gnn_features, self.fixed_noise)
                fixed_coords_denorm = self.denormalize_vertices(fixed_coords, min_coords, range_coords)
                np.save(f"gan_corrected_coords_epoch_{epoch}.npy", fixed_coords_denorm)
            
            if progress_callback:
                progress = (epoch + 1) / epochs * 100
                progress_callback(progress, f"GAN Training: Epoch {epoch+1}/{epochs}, D_Loss: {d_loss:.6f}, G_Loss: {g_loss:.6f}")
        
        return losses_d, losses_g
    # gan_model.py -> class GAN

    # ... (после метода train)

    def save_weights(self, g_path, d_path):
        """Сохраняет веса генератора и дискриминатора в отдельные файлы."""
        # Сохраняем генератор
        g_state = {
            'weights': [w.tolist() for w in self.generator.weights],
            'biases': [b.tolist() for b in self.generator.biases]
        }
        with open(g_path, 'w') as f:
            json.dump(g_state, f, indent=4)
        print(f"Generator weights saved to {g_path}")

        # Сохраняем дискриминатор
        d_state = {
            'weights': [w.tolist() for w in self.discriminator.weights],
            'biases': [b.tolist() for b in self.discriminator.biases]
        }
        with open(d_path, 'w') as f:
            json.dump(d_state, f, indent=4)
        print(f"Discriminator weights saved to {d_path}")

    def load_weights(self, g_path, d_path):
        """Загружает веса генератора и дискриминатора из файлов."""
        # Загружаем генератор
        with open(g_path, 'r') as f:
            g_state = json.load(f)
        self.generator.weights = [np.array(w) for w in g_state['weights']]
        self.generator.biases = [np.array(b) for b in g_state['biases']]
        print(f"Generator weights loaded from {g_path}")

        # Загружаем дискриминатор
        with open(d_path, 'r') as f:
            d_state = json.load(f)
        self.discriminator.weights = [np.array(w) for w in d_state['weights']]
        self.discriminator.biases = [np.array(b) for b in d_state['biases']]
        print(f"Discriminator weights loaded from {d_path}")


def example_usage():
    """Example usage of GAN with GNN features and reference data."""
    # Load preprocessed data
    with open("preprocessed_data_defective.json", "r") as f:
        defective_data = json.load(f)
    with open("preprocessed_data_reference.json", "r") as f:
        reference_data = json.load(f)
    
    # Convert data to numpy arrays
    defective_data = {
        "node_features": np.array(defective_data["node_features"]),
        "adjacency_matrix": np.array(defective_data["adjacency_matrix"])
    }
    real_coords = np.array(reference_data["normalized_vertices"])  # Use normalized vertices
    min_coords = np.array(reference_data.get("min_coords", np.zeros(3)))  # Assume saved in JSON
    range_coords = np.array(reference_data.get("range_coords", np.ones(3)))
    
    # Initialize GNN
    gnn = GNN(input_dim=defective_data["node_features"].shape[1], 
              hidden_dim=16, 
              output_dim=real_coords.shape[1], 
              num_layers=2)
    
    # Initialize and train GAN
    gan = GAN(gnn_input_dim=real_coords.shape[1], 
              noise_dim=10, 
              hidden_dim=32, 
              coord_dim=3)
    
    losses_d, losses_g = gan.train(gnn, defective_data, real_coords, min_coords, range_coords,
                                   epochs=3, learning_rate=0.01, update_discriminator_freq=2)
    
    # Save final denormalized coordinates
    noise = np.random.randn(defective_data["node_features"].shape[0], gan.noise_dim)
    gnn_features = gnn.forward(defective_data)
    corrected_coords = gan.generator.forward(gnn_features, noise)
    corrected_coords_denorm = gan.denormalize_vertices(corrected_coords, min_coords, range_coords)
    np.save("gan_corrected_coords.npy", corrected_coords_denorm)
    
    print(f"Final Discriminator Loss: {losses_d[-1]:.6f}, Final Generator Loss: {losses_g[-1]:.6f}")

if __name__ == "__main__":
    example_usage()