import numpy as np
import networkx as nx
from typing import Tuple, Dict
#from collections import defaultdict

class OBJProcessor:
    """Utility class for processing 3D models in OBJ format and converting to graph representations."""
    
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.graph = nx.Graph()
        
    def load_obj(self, file_path: str) -> None:
        """Load an OBJ file and extract vertices and faces."""
        self.vertices = []
        self.faces = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    # Parse vertex: v x y z
                    parts = line.split()
                    if len(parts) >= 4:
                        vertex = [float(parts[i]) for i in range(1, 4)]
                        self.vertices.append(vertex)
                elif line.startswith('f '):
                    # Parse face: f v1 v2 v3 ...
                    parts = line.split()
                    face = []
                    for part in parts[1:]:
                        # Handle vertex/texture/normal format (v/t/n) or just vertex (v)
                        v_idx = int(part.split('/')[0]) - 1  # OBJ indices are 1-based
                        face.append(v_idx)
                    self.faces.append(face)
    
    def build_graph(self) -> nx.Graph:
        """Convert OBJ mesh to a graph representation (vertices as nodes, edges from faces)."""
        self.graph.clear()
        
        # Add vertices as nodes with their coordinates as features
        for i, vertex in enumerate(self.vertices):
            self.graph.add_node(i, pos=np.array(vertex))
        
        # Add edges based on faces (connect vertices that share a face)
        for face in self.faces:
            # Create edges between consecutive vertices in the face
            for i in range(len(face)):
                v1 = face[i]
                v2 = face[(i + 1) % len(face)]  # Connect to next vertex, wrap around
                self.graph.add_edge(v1, v2)
        
        return self.graph
    
    def compute_adjacency_matrix(self) -> np.ndarray:
        """Compute the adjacency matrix of the graph."""
        return np.array(nx.adjacency_matrix(self.graph).todense())
    
    def compute_node_features(self) -> np.ndarray:
        """Extract node features (vertex coordinates and degree)."""
        features = []
        for node in self.graph.nodes:
            pos = self.graph.nodes[node]['pos']
            degree = self.graph.degree[node]
            # Combine position and degree as node features
            feature = np.append(pos, degree)
            features.append(feature)
        return np.array(features)
    
    def compute_edge_features(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Compute edge features (e.g., Euclidean distance between vertices)."""
        edge_features = {}
        for u, v in self.graph.edges:
            pos_u = self.graph.nodes[u]['pos']
            pos_v = self.graph.nodes[v]['pos']
            distance = np.linalg.norm(pos_u - pos_v)
            edge_features[(u, v)] = np.array([distance])
        return edge_features
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize node features to zero mean and unit variance."""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (features - mean) / std
    
    def analyze_mesh(self) -> Dict[str, float]:
        """Analyze mesh properties: surface area, volume, and bounding box."""
        analysis = {}
        
        # Surface area: sum of triangle areas (for triangular faces)
        surface_area = 0.0
        for face in self.faces:
            if len(face) == 3:  # Assume triangular faces
                v0 = np.array(self.vertices[face[0]])
                v1 = np.array(self.vertices[face[1]])
                v2 = np.array(self.vertices[face[2]])
                # Area of triangle using cross product
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                surface_area += area
        analysis['surface_area'] = surface_area
        
        # Bounding box
        vertices_array = np.array(self.vertices)
        min_coords = np.min(vertices_array, axis=0)
        max_coords = np.max(vertices_array, axis=0)
        analysis['bounding_box_volume'] = np.prod(max_coords - min_coords)
        
        # Approximate volume using signed tetrahedron method (simplified)
        volume = 0.0
        for face in self.faces:
            if len(face) == 3:
                v0 = np.array(self.vertices[face[0]])
                v1 = np.array(self.vertices[face[1]])
                v2 = np.array(self.vertices[face[2]])
                # Signed volume of tetrahedron formed with origin
                volume += np.abs(np.dot(v0, np.cross(v1, v2))) / 6.0
        analysis['volume'] = volume
        
        return analysis
    
    def get_graph_data(self) -> Dict:
        """Return graph data suitable for GNN processing."""
        return {
            'adjacency_matrix': self.compute_adjacency_matrix(),
            'node_features': self.normalize_features(self.compute_node_features()),
            'edge_features': self.compute_edge_features(),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges()
        }

"""
def example_usage():
    #Example usage of OBJProcessor.
    processor = OBJProcessor()
    # Replace 'model.obj' with your OBJ file path
    processor.load_obj('cube.obj')
    graph = processor.build_graph()
    graph_data = processor.get_graph_data()
    analysis = processor.analyze_mesh()
    
    print(f"Number of nodes: {graph_data['num_nodes']}")
    print(f"Number of edges: {graph_data['num_edges']}")
    print(f"Mesh analysis: {analysis}")
"""