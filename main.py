# main.py

import os
import numpy as np
from collections import defaultdict
import networkx as nx
import shutil
from preprocess import DataPreprocessor
from gnn_model import GNN
from gan_model import GAN

def load_and_preprocess_data(defective_file_path, reference_file_path):
    """
    Загружает и предобрабатывает данные для обучения, делегируя всю работу DataPreprocessor.
    """
    preprocessor = DataPreprocessor()
    # output_path теперь не обязателен, но можно оставить для отладки
    output_path_prefix = os.path.join("./training_data/preprocessed", os.path.basename(defective_file_path))
    os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
    
    # Эта функция теперь возвращает словарь с уже выровненными данными
    training_data = preprocessor.prepare_training_data(
        defective_file_path, reference_file_path, output_path_prefix
    )
    return training_data

def train_models(gnn_model, gan_model, defective_data, reference_data, epochs=30, progress_callback=None):
    """Обучает модели GAN."""
    real_coords = np.array(reference_data['normalized_vertices'])
    min_coords = np.array(reference_data.get('min_coords', np.zeros(3)))
    range_coords = np.array(reference_data.get('range_coords', np.ones(3)))
    print("Starting GAN training...")
    losses_d, losses_g = gan_model.train(gnn_model, defective_data, real_coords, min_coords, range_coords, 
                                         epochs=epochs, learning_rate=0.0001, update_discriminator_freq=2,
                                         progress_callback=progress_callback)
    print("GAN training finished.")
    return gnn_model, gan_model

def correct_model(obj_file_path, gnn_model, gan_model, output_file_path):
    """Исправляет 3D модель с использованием обученных GNN и GAN (старый метод)."""
    preprocessor = DataPreprocessor()
    defective_data = preprocessor.load_and_preprocess_mesh(obj_file_path)
    gnn_features = gnn_model.forward(defective_data)
    noise = np.random.randn(defective_data['node_features'].shape[0], gan_model.noise_dim)
    corrected_coords_normalized = gan_model.generator.forward(gnn_features, noise)
    temp_preprocessor = DataPreprocessor()
    temp_preprocessor.load_and_preprocess_mesh(obj_file_path)
    original_vertices = np.array(temp_preprocessor.processor.vertices)
    min_coords = np.min(original_vertices, axis=0)
    max_coords = np.max(original_vertices, axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1
    corrected_coords_denorm = gan_model.denormalize_vertices(corrected_coords_normalized, min_coords, range_coords)
    with open(output_file_path, 'w') as f:
        for v in corrected_coords_denorm:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in temp_preprocessor.processor.faces:
            f.write("f " + " ".join([str(idx + 1) for idx in face]) + "\n")
    print(f"Corrected model saved to {output_file_path}")
    return output_file_path

def patch_model(obj_file_path, gnn_model, gan_model, output_file_path):
    """Исправляет модель, добавляя заплатки на отверстия, но НЕ изменяя исходную геометрию."""
    preprocessor = DataPreprocessor()
    preprocessor.processor.load_obj(obj_file_path)
    original_vertices = preprocessor.processor.vertices
    original_faces = preprocessor.processor.faces
    edge_count = defaultdict(int)
    for face in original_faces:
        for i in range(len(face)):
            e = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
            edge_count[e] += 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        print("Отверстий не найдено. Сохранение копии исходной модели.")
        shutil.copy(obj_file_path, output_file_path)
        return output_file_path
    boundary_graph = nx.Graph(boundary_edges)
    cycles = list(nx.cycle_basis(boundary_graph))
    if not cycles:
        print("Не найдено замкнутых контуров для исправления.")
        shutil.copy(obj_file_path, output_file_path)
        return output_file_path
    full_graph_data = preprocessor.load_and_preprocess_mesh(obj_file_path)
    gnn_features_full = gnn_model.forward(full_graph_data)
    new_gnn_features = []
    valid_cycles = []
    for cycle in cycles:
        if len(cycle) < 3: continue
        valid_cycles.append(cycle)
        cycle_features = np.mean([gnn_features_full[i] for i in cycle], axis=0)
        new_gnn_features.append(cycle_features)
    if not new_gnn_features:
        print("Не найдено подходящих контуров для исправления.")
        shutil.copy(obj_file_path, output_file_path)
        return output_file_path
    new_gnn_features = np.array(new_gnn_features)
    noise = np.random.randn(new_gnn_features.shape[0], gan_model.noise_dim)
    predicted_new_vertices_normalized = gan_model.generator.forward(new_gnn_features, noise)
    min_coords = np.array(full_graph_data.get('min_coords', np.min(original_vertices, axis=0)))
    range_coords = np.array(full_graph_data.get('range_coords', np.max(original_vertices, axis=0) - min_coords))
    range_coords[range_coords == 0] = 1
    predicted_new_vertices = gan_model.denormalize_vertices(predicted_new_vertices_normalized, min_coords, range_coords)
    final_vertices = original_vertices + predicted_new_vertices.tolist()
    final_faces = list(original_faces)
    vertex_offset = len(original_vertices)
    for i, cycle in enumerate(valid_cycles):
        new_vertex_index = vertex_offset + i
        for j in range(len(cycle)):
            v1 = cycle[j]
            v2 = cycle[(j + 1) % len(cycle)]
            final_faces.append([new_vertex_index, v2, v1])
    with open(output_file_path, 'w') as f:
        for v in final_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in final_faces:
            f.write("f " + " ".join([str(idx + 1) for idx in face]) + "\n")
    print(f"Patched model saved to {output_file_path}")
    return output_file_path

# Код для тестирования, который можно закомментировать
if __name__ == "__main__":
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    defective_obj = os.path.join(data_dir, "defective.obj")
    reference_obj = os.path.join(data_dir, "reference.obj")

    # Создаем фиктивные OBJ файлы для тестирования
    with open(defective_obj, 'w') as f:
        f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    with open(reference_obj, 'w') as f:
        f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    print(f"Loading and preprocessing data from {defective_obj} and {reference_obj}...")
    training_data = load_and_preprocess_data(defective_obj, reference_obj)
    defective_graph_data = training_data['defective']
    reference_graph_data = training_data['reference']

    # Инициализируем модели для теста
    gnn_model = GNN(input_dim=defective_graph_data['node_features'].shape[1], hidden_dim=16, output_dim=8, num_layers=2)
    gan_model = GAN(gnn_input_dim=8, noise_dim=16, hidden_dim=32, coord_dim=3)

    gnn_model, gan_model = train_models(gnn_model, gan_model, defective_graph_data, reference_graph_data, epochs=1)

    output_corrected_obj = os.path.join(data_dir, "corrected_model.obj")
    correct_model(defective_obj, gnn_model, gan_model, output_corrected_obj)
    print(f"Correction process completed. Corrected model saved to {output_corrected_obj}")
