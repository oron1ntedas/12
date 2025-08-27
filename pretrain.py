# pretrain.py

import os
import numpy as np
import glob
import random
import sys

# Импортируем наши классы и функции
from preprocess import DataPreprocessor
from gnn_model import GNN
from gan_model import GAN
from main import train_models
from obj_process import OBJProcessor # Нам нужен загрузчик

# --- Настройки ---
REFERENCE_MODELS_DIR = "./rereference_models"  # Папка с вашими эталонными OBJ-файлами
WEIGHTS_DIR = "./weights"                  # Папка для сохранения обученных весов
TEMP_TRAINING_DIR = "./temp_training"      # Временная папка для сгенерированных пар

# ... (вставьте сюда новую функцию create_defective_version, которую мы написали выше) ...
# pretrain.py

# ... (после всех import)

def create_defective_version(vertices, faces, noise_level=0.08, hole_chance=0.4, vertex_drop_chance=0.2):
    """
    Принимает вершины и грани и возвращает их измененные версии с дефектами.
    """
    new_vertices = list(vertices)
    new_faces = list(faces)

    # 1. Шум для вершин
    if noise_level > 0:
        for i in range(len(new_vertices)):
            noise = (np.random.rand(3) - 0.5) * 2 * noise_level
            new_vertices[i] = np.array(new_vertices[i]) + noise

    # 2. Удаление случайных граней (создание дыр)
    if np.random.rand() < hole_chance and len(new_faces) > 10:
        num_to_remove = np.random.randint(1, int(len(new_faces) * 0.1) + 2) # Удаляем до 10% граней
        for _ in range(num_to_remove):
            if new_faces:
                new_faces.pop(np.random.randint(len(new_faces)))

    # 3. Удаление случайных вершин (самый сложный дефект)
    if np.random.rand() < vertex_drop_chance and len(new_vertices) > 10:
        num_to_remove = np.random.randint(1, int(len(new_vertices) * 0.1) + 2) # Удаляем до 10% вершин
        
        verts_to_remove_indices = set(random.sample(range(len(new_vertices)), num_to_remove))
        
        # Обновляем вершины
        temp_verts = [v for i, v in enumerate(new_vertices) if i not in verts_to_remove_indices]
        
        # Обновляем индексы в гранях
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(i for i in range(len(new_vertices)) if i not in verts_to_remove_indices)}
        
        temp_faces = []
        for face in new_faces:
            new_face = [mapping.get(idx) for idx in face]
            # Если вершина из грани была удалена, грань становится невалидной (None)
            if None not in new_face:
                temp_faces.append(new_face)
        
        new_vertices = temp_verts
        new_faces = temp_faces

    return new_vertices, new_faces


# --- Основная функция обучения ---
def run_pretraining():
    # 1. Подготовка папок
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(TEMP_TRAINING_DIR, exist_ok=True)
    
    reference_files = glob.glob(os.path.join(REFERENCE_MODELS_DIR, "*.obj"))
    if len(reference_files) < 5: # Ставим минимальный порог для разнообразия
        print(f"ВНИМАНИЕ: Найдено всего {len(reference_files)} эталонных моделей.")
        print("Для качественного обучения рекомендуется иметь хотя бы 5-10 РАЗНЫХ моделей.")
        if not reference_files: return

    print(f"Найдено {len(reference_files)} эталонных моделей для генерации обучающих данных.")

    # 2. Инициализация моделей
    print("\n--- Шаг 1: Инициализация моделей ---")
    GNN_INPUT_DIM, GNN_OUTPUT_DIM, VERTEX_COORD_DIM, LATENT_DIM = 4, 8, 3, 16
    HIDDEN_DIM_GNN, HIDDEN_DIM_GAN = 16, 32
    
    gnn_model = GNN(input_dim=GNN_INPUT_DIM, hidden_dim=HIDDEN_DIM_GNN, output_dim=GNN_OUTPUT_DIM, num_layers=2)
    gan_model = GAN(gnn_input_dim=GNN_OUTPUT_DIM, noise_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM_GAN, coord_dim=VERTEX_COORD_DIM)

    # 3. Цикл обучения с генерацией "на лету"
    print("\n--- Шаг 2: Запуск обучения ---")
    total_epochs = 10  # Сколько раз мы пройдемся по всем эталонам
    iterations_per_epoch = 5 # Сколько сломанных версий мы сгенерируем за эпоху
    
    obj_loader = OBJProcessor()
    preprocessor = DataPreprocessor()

    for epoch_num in range(total_epochs):
        print(f"\n*** Эпоха {epoch_num + 1}/{total_epochs} ***")
        
        # Перемешиваем эталоны в начале каждой эпохи
        random.shuffle(reference_files)
        
        for i in range(iterations_per_epoch):
            # Выбираем случайный эталон
            ref_path = random.choice(reference_files)
            
            # 1. Загружаем эталон
            obj_loader.load_obj(ref_path)
            
            # 2. Создаем сломанную версию
            def_verts, def_faces = create_defective_version(obj_loader.vertices, obj_loader.faces)
            
            # 3. Сохраняем сломанную и эталонную модели во временные файлы
            temp_def_path = os.path.join(TEMP_TRAINING_DIR, "temp_defective.obj")
            temp_ref_path = os.path.join(TEMP_TRAINING_DIR, "temp_reference.obj")

            with open(temp_def_path, 'w') as f:
                for v in def_verts: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in def_faces: f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")
            
            # Копируем оригинальный эталон для пары
            from shutil import copyfile
            copyfile(ref_path, temp_ref_path)

            # 4. Загружаем пару и обучаемся на ней
            try:
                training_data = preprocessor.prepare_training_data(temp_def_path, temp_ref_path, "")
                gnn_model, gan_model = train_models(
                    gnn_model, gan_model, training_data['defective'], training_data['reference'], epochs=5 # Несколько итераций на одной паре
                )
            except Exception as e:
                print(f"  Пропущена итерация из-за ошибки при обработке '{os.path.basename(ref_path)}': {e}")
                continue

            sys.stdout.write(f"\r  Итерация {i+1}/{iterations_per_epoch}...")
            sys.stdout.flush()

    # 4. Сохранение весов
    print("\n\n--- Шаг 3: Сохранение финальных весов ---")
    gnn_model.save_weights(os.path.join(WEIGHTS_DIR, "gnn_weights.json"))
    gan_model.save_weights(
        os.path.join(WEIGHTS_DIR, "generator_weights.json"),
        os.path.join(WEIGHTS_DIR, "discriminator_weights.json")
    )
    
    print("\nПредобучение завершено!")

if __name__ == "__main__":
    run_pretraining()
