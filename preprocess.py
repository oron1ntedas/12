import numpy as np
import networkx as nx
from typing import Dict
import json
from obj_process import OBJProcessor

class DataPreprocessor:
    """Class for preprocessing 3D model data for GNN and GAN training."""
    
    def __init__(self):
        self.processor = OBJProcessor()
        
    def load_and_preprocess_mesh(self, file_path: str) -> Dict:
        """Load OBJ file, normalize vertices, and create graph for GNN."""
        # Load OBJ file and build graph
        self.processor.load_obj(file_path)
        graph = self.processor.build_graph()
        
        # Normalize vertex coordinates to [0, 1]
        vertices = np.array(self.processor.vertices)
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1  # Avoid division by zero
        normalized_vertices = (vertices - min_coords) / range_coords
        
        # Update graph node features with normalized vertices
        for i, node in enumerate(graph.nodes):
            graph.nodes[node]['pos_normalized'] = normalized_vertices[i]
        
        # Use normalized positions for node features
        node_features = self._compute_normalized_node_features(graph)
        
        # Get graph data with normalized features
        graph_data = {
            'adjacency_matrix': self.processor.compute_adjacency_matrix(),
            'node_features': self.processor.normalize_features(node_features),
            'edge_features': self.processor.compute_edge_features(),
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'normalized_vertices': normalized_vertices
        }
        
        return graph_data
    
    def _compute_normalized_node_features(self, graph: nx.Graph) -> np.ndarray:
        """Compute node features using normalized positions and degree."""
        features = []
        for node in graph.nodes:
            pos = graph.nodes[node]['pos_normalized']  # Use normalized coordinates
            degree = graph.degree[node]
            feature = np.append(pos, degree)
            features.append(feature)
        return np.array(features)
    
    def save_graph_data(self, graph_data: Dict, output_path: str) -> None:
        """Save preprocessed graph data for debugging or training."""
        save_data = {
            'adjacency_matrix': graph_data['adjacency_matrix'].tolist(),
            'node_features': graph_data['node_features'].tolist(),
            'edge_features': {str(k): v.tolist() for k, v in graph_data['edge_features'].items()},
            'num_nodes': graph_data['num_nodes'],
            'num_edges': graph_data['num_edges'],
            'normalized_vertices': graph_data['normalized_vertices'].tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=4)
    
    def match_vertices(self, defective_data: Dict, reference_data: Dict) -> Dict:
        """
        Сопоставляет вершины и приводит дефектные данные к размеру эталонных.
        Корректно обрабатывает случай, когда в дефектной модели нет вершин.
        """
        def_vertices = defective_data['normalized_vertices']
        ref_vertices = reference_data['normalized_vertices']
        
        num_def_verts = def_vertices.shape[0]
        num_ref_verts = ref_vertices.shape[0]

        # Создаем новые массивы для дефектных данных, размером с эталонные
        aligned_def_features = np.zeros_like(reference_data['node_features'])
        aligned_def_adj = np.zeros_like(reference_data['adjacency_matrix'])

        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ---
        # Мы выполняем сопоставление, только если в дефектной модели ЕСТЬ вершины.
        if num_def_verts > 0:
            # Для каждой эталонной вершины ищем ближайшую дефектную
            for i in range(num_ref_verts):
                ref_vertex = ref_vertices[i]
                distances = np.linalg.norm(def_vertices - ref_vertex, axis=1)
                closest_def_idx = np.argmin(distances)
                aligned_def_features[i] = defective_data['node_features'][closest_def_idx]
            
            # Копируем матрицу смежности, насколько это возможно
            min_dim = min(num_def_verts, num_ref_verts)
            aligned_def_adj[:min_dim, :min_dim] = defective_data['adjacency_matrix'][:min_dim, :min_dim]
        
        # Если в дефектной модели не было вершин (num_def_verts == 0),
        # то aligned_def_features и aligned_def_adj останутся нулевыми,
        # но, что важно, они будут ПРАВИЛЬНОГО РАЗМЕРА (как у эталона).

        # Возвращаем новый, "выровненный" словарь данных
        return {
            'adjacency_matrix': aligned_def_adj,
            'node_features': aligned_def_features,
            'edge_features': defective_data.get('edge_features', {}), # .get для безопасности
            'num_nodes': num_ref_verts, # Количество узлов теперь всегда равно эталону
            'num_edges': defective_data.get('num_edges', 0),
            'normalized_vertices': defective_data.get('normalized_vertices', np.array([]))
        }

    # ЗАМЕНИТЕ И ЭТОТ МЕТОД
    def prepare_training_data(self, defective_path: str, reference_path: str, output_path: str) -> Dict:
        """
        Готовит парные данные, где дефектная модель ГАРАНТИРОВАННО приведена к размеру эталонной.
        """
        defective_data_original = self.load_and_preprocess_mesh(defective_path)
        reference_data = self.load_and_preprocess_mesh(reference_path)
        
        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ---
        # Создаем новую версию дефектных данных, подогнанную под размер эталона
        aligned_defective_data = self.match_vertices(defective_data_original, reference_data)

        # Сохраняем для отладки (по желанию)
        # self.save_graph_data(defective_data_original, output_path + '_defective_original.json')
        # self.save_graph_data(reference_data, output_path + '_reference.json')
        
        # ВОЗВРАЩАЕМ ВЫРОВНЕННЫЕ ДАННЫЕ
        return {
            'defective': aligned_defective_data,
            'reference': reference_data
        }
        
"""
def example_usage():
    #Example usage of DataPreprocessor.
    preprocessor = DataPreprocessor()
    
    # Replace with your OBJ file paths
    defective_path = 'cube1.obj'
    reference_path = 'cube.obj'
    output_path = 'pp_data'
    
    training_data = preprocessor.prepare_training_data(
        defective_path, reference_path, output_path
    )
    
    print(f"Defective model nodes: {training_data['defective']['num_nodes']}")
    print(f"Reference model nodes: {training_data['reference']['num_nodes']}")
    print(f"Number of matched vertices: {len(training_data['matched']['matches'])}")
"""