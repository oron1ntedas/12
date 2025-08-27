# ui.py

import pygame
import sys
import math
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import tkinter as tk
from tkinter import filedialog
import pygame.freetype
import os
from collections import defaultdict
import networkx as nx

# Импортируем необходимые компоненты из других модулей
from main import load_and_preprocess_data, train_models, correct_model, patch_model
from gnn_model import GNN
from gan_model import GAN

class Simple3DViewer:
    def __init__(self, obj_file_path, width=800, height=650, gnn_model=None, gan_model=None):
        self.width = width
        self.height = height-50
        self.menu_bar_height = 50
        self.obj_file = obj_file_path

        self.vertices = []
        self.faces = []
        self.edges = []
        self.boundary_edges = []
        self.predicted_new_vertices = []
        self.predicted_new_edges = []

        self.highlight_bad_geometry = False
        self.highlight_missing_faces = False
        self.highlight_patched_areas = False
        self.progress = 0.0
        self.progress_message = ""

        self.distance = 5.0
        self.azimuth = 0.0
        self.elevation = 0.0

        self.mouse_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        self.model_offset = [0.0, 0.0]
        self.right_dragging = False

        self.gnn_model = gnn_model
        self.gan_model = gan_model
        self.original_vertices = []
        self.original_faces = []
        self.original_edges = []

        self.init_pygame()
        self.init_opengl()
        self.load_obj(obj_file_path)
        self.boundary_edges = self.detect_boundary_edges()

    def init_pygame(self):
        pygame.init()
        pygame.freetype.init()
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        self.font = pygame.freetype.SysFont(None, 16)
        pygame.display.set_caption(f"3D Viewer - {os.path.basename(self.obj_file)}")

    def init_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glPointSize(6)
        glLineWidth(1.5)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT, GL_SHININESS, [10.0])
        glMaterialfv(GL_BACK, GL_AMBIENT, [0.3, 0.1, 0.1, 1.0])
        glMaterialfv(GL_BACK, GL_DIFFUSE, [0.5, 0.1, 0.1, 1.0])
        glMaterialfv(GL_BACK, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        glMaterialfv(GL_BACK, GL_SHININESS, [0.0])
        
        # Фон для всей области окна (будет виден под 3D-видом)
        glClearColor(0.92, 0.92, 0.92, 1.0)

        # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
        # Настраиваем матрицу проекции для 3D-вида
        glMatrixMode(GL_PROJECTION)
        # Рассчитываем соотношение сторон для области ПОД меню-баром
        view_height = self.height - self.menu_bar_height
        aspect_ratio = self.width / view_height if view_height > 0 else 1
        gluPerspective(45, aspect_ratio, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)

    def load_obj(self, file_path):
        self.obj_file = file_path
        self.vertices = []
        self.faces = []
        self.edges = set()
        try:
            with open(self.obj_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('v '):
                        parts = line.split()
                        vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                        self.vertices.append(vertex)
                    elif line.startswith('f '):
                        parts = line.split()
                        face_indices = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                        for i in range(len(face_indices)):
                            v1_idx, v2_idx = face_indices[i], face_indices[(i + 1) % len(face_indices)]
                            self.edges.add(tuple(sorted((v1_idx, v2_idx))))
                        if len(face_indices) >= 3:
                            for i in range(1, len(face_indices) - 1):
                                self.faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])
            self.edges = list(self.edges)
            if self.vertices:
                vertices_array = np.array(self.vertices)
                center = np.mean(vertices_array, axis=0)
                max_size = np.max(np.max(vertices_array, axis=0) - np.min(vertices_array, axis=0))
                if max_size > 0:
                    self.vertices = [((v - center) / max_size * 2).tolist() for v in vertices_array]
            self.original_vertices, self.original_faces, self.original_edges = list(self.vertices), list(self.faces), list(self.edges)
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")

    def detect_boundary_edges(self):
        edge_count = defaultdict(int)
        for face in self.faces:
            for i in range(len(face)):
                e = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                edge_count[e] += 1
        return [e for e, c in edge_count.items() if c == 1]

    def generate_patch_preview(self):
        boundary_graph = nx.Graph(self.boundary_edges)
        cycles = list(nx.cycle_basis(boundary_graph))
        self.predicted_new_vertices, self.predicted_new_edges = [], []
        v_idx = len(self.vertices)
        for cycle in cycles:
            if len(cycle) < 3: continue
            center = np.mean([np.array(self.vertices[i]) for i in cycle], axis=0)
            self.predicted_new_vertices.append(center.tolist())
            for v in cycle:
                self.predicted_new_edges.append((v_idx, v))
            v_idx += 1

    def update_camera(self):
        glLoadIdentity()
        x = self.distance * math.cos(math.radians(self.elevation)) * math.cos(math.radians(self.azimuth))
        y = self.distance * math.sin(math.radians(self.elevation))
        z = self.distance * math.cos(math.radians(self.elevation)) * math.sin(math.radians(self.azimuth))
        gluLookAt(x, y, z, 0, 0, 0, 0, 1, 0)

    def calculate_normal(self, v1, v2, v3):
        edge1, edge2 = np.array(v2) - np.array(v1), np.array(v3) - np.array(v1)
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 0 else normal

    # ui.py -> class Simple3DViewer

    # ЗАМЕНИТЕ СТАРЫЙ МЕТОД draw_text НА ЭТОТ:
    # ui.py -> class Simple3DViewer

    def draw_text(self, text, x, y):
        """
        Надежно отрисовывает текст, создавая из него текстуру OpenGL.
        """
        # 1. Рендерим текст на поверхность Pygame
        text_surface, rect = self.font.render(text, (0, 0, 0))
        
        # 2. Получаем данные о пикселях и размеры, ПЕРЕВОРАЧИВАЯ ИЗОБРАЖЕНИЕ ПО ВЕРТИКАЛИ
        text_data = pygame.image.tostring(pygame.transform.flip(text_surface, False, True), "RGBA", True)
        width, height = text_surface.get_width(), text_surface.get_height()

        # 3. Создаем и настраиваем текстуру OpenGL
        glEnable(GL_TEXTURE_2D)
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # 4. Рисуем прямоугольник (квад) с этой текстурой
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + width, y)
        glTexCoord2f(1, 1); glVertex2f(x + width, y + height)
        glTexCoord2f(0, 1); glVertex2f(x, y + height)
        glEnd()

        # 5. Отключаем текстурирование и смешивание
        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures(texid)

    # ui.py -> class Simple3DViewer

    # ЗАМЕНИТЕ СУЩЕСТВУЮЩИЙ МЕТОД render
    # ui.py -> class Simple3DViewer

    # ui.py -> class Simple3DViewer

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(self.model_offset[0], self.model_offset[1], 0.0)
        self.update_camera()

        glPushMatrix()

        # --- Отрисовка 3D-модели ---
        # Рисуем грани
        glEnable(GL_LIGHTING)
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            try:
                normal = self.calculate_normal(self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]])
                glNormal3fv(normal)
                for vertex_idx in face:
                    glVertex3fv(self.vertices[vertex_idx])
            except IndexError:
                # Пропускаем грань, если индекс вершины выходит за пределы
                continue
        glEnd()

        # Рисуем ребра
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_LINES)
        for edge in self.edges:
            try:
                for vertex_idx in edge:
                    glVertex3fv(self.vertices[vertex_idx])
            except IndexError:
                continue
        glEnd()

        # --- Отрисовка выделений и предпросмотра ---
        # Предпросмотр заплаток (красные линии и точки)
        if self.highlight_patched_areas:
            glDisable(GL_LIGHTING)
            glColor3f(1.0, 0.0, 0.0) # Красный цвет
            glLineWidth(1.5)
            glBegin(GL_LINES)
            for e in self.predicted_new_edges:
                try:
                    # Новые вершины еще не в основном списке, берем их из predicted_new_vertices
                    v1 = self.vertices[e[1]]
                    v2 = self.predicted_new_vertices[e[0] - len(self.vertices)]
                    glVertex3fv(v1)
                    glVertex3fv(v2)
                except IndexError:
                    continue
            glEnd()
            
            glPointSize(5.0)
            glBegin(GL_POINTS)
            for v in self.predicted_new_vertices:
                glVertex3fv(v)
            glEnd()

        glPopMatrix()

        # Вызываем отдельный метод для отрисовки всего 2D UI
        self.draw_ui()

        pygame.display.flip()



    # ЗАМЕНИТЕ СУЩЕСТВУЮЩИЙ МЕТОД draw_ui
    def draw_ui(self):
        # Переключаемся в 2D-режим для всего окна
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # --- Рисуем фон меню-бара ---
        # Цвет фона (например, светло-серый)
        glColor3f(0.8, 0.8, 0.8)
        # Рисуем прямоугольник вверху экрана
        glRectf(0, 0, self.width, self.menu_bar_height)

        # --- Рисуем кнопки внутри меню-бара ---
        # Рассчитываем Y-координату, чтобы текст был по центру бара
        button_y = (self.menu_bar_height - 16) / 2 # 16 - примерная высота шрифта
        self.draw_text("Preview Patches", 10, button_y)
        self.draw_text("Full Correction", 200, button_y)

        # --- Рисуем прогресс-бар (если нужно) ---
        # Он будет рисоваться поверх меню-бара
        if self.progress_message:
            # Смещаем его вниз, чтобы не перекрывать кнопки
            progress_y_offset = self.menu_bar_height + 10
            self.draw_text(self.progress_message, 10, progress_y_offset)
            
            bar_width, bar_height = 200, 20
            fill_width = int(bar_width * (self.progress / 100))
            
            bar_y = progress_y_offset + 20
            glColor3f(0.7, 0.7, 0.7)
            glRectf(10, bar_y, 10 + bar_width, bar_y + bar_height)
            glColor3f(0.0, 1.0, 0.0)
            glRectf(10, bar_y, 10 + fill_width, bar_y + bar_height)
            glColor3f(0.0, 0.0, 0.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(10, bar_y); glVertex2f(10 + bar_width, bar_y)
            glVertex2f(10 + bar_width, bar_y + bar_height); glVertex2f(10, bar_y + bar_height)
            glEnd()

        # Возвращаем стандартные настройки
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    # ЗАМЕНИТЕ СУЩЕСТВУЮЩИЙ МЕТОД handle_mouse_click
    def handle_mouse_click(self, pos):
        x, y = pos
        # Область кнопки "Preview Patches"
        if y <= self.menu_bar_height:
            if 10 <= x <= 190 :
                self.highlight_patched_areas = not self.highlight_patched_areas
                if self.highlight_patched_areas:
                    print("Предпросмотр заплаток активирован.")
                    self.generate_patch_preview()
                else:
                    print("Предпросмотр заплаток деактивирован.")
                    # Сбрасываем предпросмотр
                    self.predicted_new_vertices = []
                    self.predicted_new_edges = []

        # Область кнопки "Full Correction"
            elif 200 <= x <= 400:
                print("Запущено полное исправление модели...")
                # Сбрасываем предпросмотр перед исправлением
                self.highlight_patched_areas = False
                self.predicted_new_vertices = []
                self.predicted_new_edges = []
                self.correct_and_display_model()


    def handle_mouse_motion(self, x, y):
        dx, dy = x - self.last_mouse_x, y - self.last_mouse_y
        if self.mouse_dragging:
            self.azimuth += dx * 0.5
            self.elevation = max(-90, min(90, self.elevation + dy * 0.5))
        elif self.right_dragging:
            self.model_offset[0] += dx / 300.0
            self.model_offset[1] -= dy / 300.0
        self.last_mouse_x, self.last_mouse_y = x, y

    def progress_update(self, percent, message):
        self.progress, self.progress_message = percent, message
        self.render()

    def correct_and_display_model(self):
        if not self.gnn_model or not self.gan_model:
            self.progress_message = "Ошибка: Модели не загружены."
            return
        
        output_dir = "./data"
        output_patched_obj = os.path.join(output_dir, "patched_" + os.path.basename(self.obj_file))
        
        try:
            # Переобучение моделей остается полезным для адаптации к стилю модели
            self.progress_update(10, "Адаптация моделей...")
            defective_obj_for_training = os.path.join(output_dir, "defective_for_training.obj")
            reference_obj_for_training = os.path.join(output_dir, "reference_for_training.obj")
            training_data = load_and_preprocess_data(defective_obj_for_training, reference_obj_for_training)
            self.gnn_model, self.gan_model = train_models(
                self.gnn_model, self.gan_model, training_data['defective'], training_data['reference'],
                epochs=10, progress_callback=self.progress_update # Меньше эпох для скорости
            )

            self.progress_update(80, "Поиск и закрытие отверстий...")
            
            # ВЫЗЫВАЕМ НОВУЮ ФУНКЦИЮ
            patched_file_path = patch_model(self.obj_file, self.gnn_model, self.gan_model, output_patched_obj)
            
            self.progress_update(95, "Загрузка результата...")
            self.load_obj(patched_file_path)
            pygame.display.set_caption(f"3D Viewer - {os.path.basename(patched_file_path)} (Дополнено)")
            self.progress_update(100, "Достраивание завершено!")

        except Exception as e:
            self.progress_message = f"Ошибка: {e}"
            print(f"Ошибка в процессе достраивания: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1: self.mouse_dragging = True
                    elif event.button == 3: self.right_dragging = True
                    self.last_mouse_x, self.last_mouse_y = event.pos
                    self.handle_mouse_click(event.pos)
                elif event.type == MOUSEBUTTONUP:
                    if event.button == 1: self.mouse_dragging = False
                    elif event.button == 3: self.right_dragging = False
                elif event.type == MOUSEMOTION: self.handle_mouse_motion(*event.pos)
                elif event.type == MOUSEWHEEL: self.distance = max(1.0, self.distance - event.y * 0.5)
            self.render()
            clock.tick(60)
        pygame.quit()

CONFIG_FILE = "config.txt"

def save_last_directory(path):
    """Сохраняет путь к директории в файл конфигурации."""
    try:
        # os.path.dirname() получает путь к папке из полного пути к файлу
        directory = os.path.dirname(path)
        with open(CONFIG_FILE, "w") as f:
            f.write(directory)
    except Exception as e:
        print(f"Не удалось сохранить последнюю директорию: {e}")

def load_last_directory():
    """Загружает путь к последней директории из файла конфигурации."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                directory = f.read().strip()
                # Проверяем, существует ли еще эта директория
                if os.path.isdir(directory):
                    return directory
    except Exception as e:
        print(f"Не удалось загрузить последнюю директорию: {e}")
    # Возвращаем None, если файл не найден или директория не существует
    return None

# ui.py

def main():
    print("Инициализация и загрузка предобученной модели...")
    
    # --- Настройки и пути к файлам весов ---
    WEIGHTS_DIR = "./weights"
    GNN_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "gnn_weights.json")
    GENERATOR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "generator_weights.json")
    DISCRIMINATOR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "discriminator_weights.json")

    # 1. Проверяем, существуют ли все необходимые файлы весов
    required_weights = [GNN_WEIGHTS_PATH, GENERATOR_WEIGHTS_PATH, DISCRIMINATOR_WEIGHTS_PATH]
    if not all(os.path.exists(p) for p in required_weights):
        print("="*60)
        print("ОШИБКА: Файлы весов предобученной модели не найдены!")
        print(f"Ожидались файлы в папке: {os.path.abspath(WEIGHTS_DIR)}")
        print("Пожалуйста, запустите скрипт 'python pretrain.py' для создания весов.")
        print("="*60)
        # Можно показать окно с ошибкой, но пока просто выходим
        # (Для Tkinter окна можно использовать: tk.messagebox.showerror(...))
        return

    # 2. Инициализация пустых моделей с правильными параметрами
    # !!! ВАЖНО: Эти параметры должны быть ТЕ ЖЕ САМЫЕ, что и в pretrain.py !!!
    try:
        # Определяем параметры моделей
        # GNN_INPUT_DIM нужно взять из сохраненных весов или задать константой.
        # Проще всего задать его константой, так как он зависит от признаков в OBJProcessor.
        # В нашем случае это: x, y, z, degree -> 4 признака.
        GNN_INPUT_DIM = 4 
        GNN_OUTPUT_DIM = 8
        VERTEX_COORD_DIM = 3
        LATENT_DIM = 16
        HIDDEN_DIM_GNN = 16
        HIDDEN_DIM_GAN = 32
        
        gnn_model = GNN(input_dim=GNN_INPUT_DIM, hidden_dim=HIDDEN_DIM_GNN, output_dim=GNN_OUTPUT_DIM, num_layers=2)
        gan_model = GAN(gnn_input_dim=GNN_OUTPUT_DIM, noise_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM_GAN, coord_dim=VERTEX_COORD_DIM)

        # 3. Загрузка весов в созданные модели
        gnn_model.load_weights(GNN_WEIGHTS_PATH)
        gan_model.load_weights(GENERATOR_WEIGHTS_PATH, DISCRIMINATOR_WEIGHTS_PATH)
        
        print("Предобученная модель успешно загружена.")
    
    except Exception as e:
        print(f"Критическая ошибка при инициализации или загрузке моделей: {e}")
        # Выводим traceback для детальной отладки
        import traceback
        traceback.print_exc()
        return

    # 4. Диалог выбора файла для исправления (без изменений)
    last_dir = load_last_directory()
    root = tk.Tk()
    root.withdraw()
    obj_file = filedialog.askopenfilename(
        title="Выбор 3D модели для исправления",
        filetypes=[("OBJ files", "*.obj")],
        initialdir=last_dir
    )

    # 5. Запуск просмотрщика с уже готовыми моделями
    if obj_file:
        save_last_directory(obj_file)
        # Передаем уже полностью готовые, обученные модели в просмотрщик
        viewer = Simple3DViewer(obj_file, gnn_model=gnn_model, gan_model=gan_model)
        viewer.run()
    else:
        print("Файл не выбран. Завершение.")

if __name__ == "__main__":
    main()
