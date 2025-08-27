# 3D Model Repair and Visualization Tool

This project is a Python 3.12 application designed to repair and visualize 3D models using Graph Neural Networks (GNNs) and Generative Adversarial Networks (GANs). It provides a graphical user interface (UI) to load, correct, and display 3D models in OBJ format, leveraging pre-trained models for automated defect correction.

## Features
- Load and visualize 3D models in OBJ format.
- Automatically detect and repair defects (holes or deformations) using pre-trained GNN and GAN models.
- Option to patch holes without altering the original geometry.
- Interactive 3D viewer with rotation, zoom, and highlighting of repaired areas.
- Pre-training pipeline to generate and train models on augmented 3D data.

## Requirements
- Python 3.12
- Required Python packages (install via `pip`):
  - `numpy`
  - `networkx`
  - `trimesh`
  - `pygame`
  - `OpenGL`
  - `PyOpenGL`
  - `tkinter` (usually included with Python)
  - `asyncio` (included with Python 3.12)

## Installation
1. Ensure Python 3.12 is installed on your system.
2. Clone the repository or download the source code.
3. Navigate to the project directory and install dependencies:
   ```bash
   pip install numpy networkx trimesh pygame PyOpenGL
   ```

## Usage
1. **Pre-training Models**:
   - Run `python pretrain.py` to generate augmented training data and train the GNN and GAN models.
   - This will create weight files in the `./weights` directory or use weight files from repository.
2. **Launching the UI**:
   - Run `python ui.py` to start the graphical interface.
   - Use the file dialog to select a 3D model (e.g., `cone.obj`) for repair and visualization.
3. **Interacting with the UI**:
   - Rotate the model with left-click and drag.
   - Zoom with the mouse wheel.
   - Use the "Correct Model" or "Patch Model" options to apply repairs.
   - View the corrected model in the 3D viewer or export it.

## Project Structure
- `gnn_model.py`: Implements the Graph Neural Network for feature extraction.
- `gan_model.py`: Implements the Generative Adversarial Network for model correction.
- `obj_process.py`: Processes OBJ files into graph representations.
- `preprocess.py`: Preprocesses 3D data for training.
- `main.py`: Contains core functions for data loading, training, and correction.
- `pretrain.py`: Orchestrates the pre-training process.
- `ui.py`: Main UI script with 3D viewer functionality.

## Configuration
- Ensure the `./data`, `./weights`, and `./rereference_models` directories are writable.

## Contributing
Feel free to submit issues or pull requests on the repository. Contributions to improve model accuracy or UI features are welcome.

## Acknowledgments
- Built using Python 3.12 and various open-source libraries.
- Inspired by 3D model repair techniques using neural networks.
