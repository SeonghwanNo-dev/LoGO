# LoGO: LoRA on the Go
### Instance-level Dynamic LoRA Selection and Merging

This repository is an implementation of the research paper **"LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"** ([arXiv:2511.07129](https://arxiv.org/abs/2511.07129)). This project was initiated under the recommendation of my advising professor to explore practical strategies for **"Multi-LoRA Serving Optimization on NVIDIA Jetson"**, which is my current area of active research.

## 🌟 Key Research Focus
*   **Train-free Adaptation**: Focuses on a train-free approach for dynamic adapter selection, making it highly efficient for resource-constrained edge devices.
*   **Activation-based Selection**: Leverages intermediate activations to determine the most relevant LoRA adapters for each specific instance.
*   **Technical Depth**: Utilizes **PyTorch's hook mechanism** to intercept and manipulate model activations.

---

## ⚙️ Project Structure

The project is organized into three primary directories, each handling a distinct stage of the machine learning pipeline:

### 1. 📂 [Data Directory](./Data)
*   **Role**: Responsible for data downloading, preprocessing, and allocation.
*   **Key Components**:
    *   `data_config.py`: Centralized configuration managing directories, dataset sources, and paths.
    *   `download_from_huggingface.py`: Utility script to download FLAN-v2 datasets directly from Hugging Face.
    *   `data_allocation.py`: A script that partitions the dataset for parallel training across multiple GPUs.

### 2. 📂 [Inference Directory](./Inference)
*   **Role**: Handles model inference and performance benchmarking.
*   **Key Components**:
    *   `inference_config.py`: Centralized `LogoConfig` for managing model IDs, target layers, and sampling parameters.
    *   `logo.py`: Main inference engine refactored for modular execution.
    *   `run_lm_eval.py`: Benchmarking tool for evaluating model accuracy across various tasks (e.g., `arc_easy`).

### 3. 📂 [Train Directory](./Train)
*   **Role**: Manages parallel LoRA fine-tuning and automated resource optimization.
*   **Key Components**:
    *   `train_config.py`: Integrated configuration for training hyperparameters and disk management.
    *   `adapter_training_1.py` & `2.py`: Scripts for simultaneous dual-GPU training.
    *   `observer_1.py` & `2.py`: Disk monitoring systems that trigger automated cloud uploads to prevent storage overflow.
    *   `google_upload.py`: Automated Google Drive integration for seamless checkpoint backup.

---

## 🚀 Getting Started

### Environment Setup

You can set up the environment using one of the following two options:

#### Option 1: Docker (Recommended for GPU Isolation)

Build the Docker image and run the container with GPU support:

```bash
# Build the Docker image
gcsudo docker build --network host -t my-experiment .

# Run the container (with GPU access, host networking, and workspace volume mount)
gcsudo docker run --gpus all -it --network host -v $(pwd):/workspace my-experiment /bin/bash
```

#### Option 2: Local Virtual Environment (venv)

Alternatively, initialize a local Python virtual environment and install all dependencies:

```bash
bash setup.sh
source venv/bin/activate
```

### Basic Workflow

The core workflow follows a **Data ➔ Train ➔ Inference** sequence.

*  **Configure First**: You must always configure the respective config file before running any processes.
*  **Execute Pipeline**: Proceed with the workflow in the order of Data, Train, and then Inference.
*  **Detailed Usage**: Please refer to the `README.md` inside each directory for specific usage and commands.

---

## 🛠 Tech Stack
*   **Core**: Python, PyTorch
*   **PEFT**: LoRA (Low-Rank Adaptation), Multi-LoRA Serving
*   **DevOps**: tmux, Google Drive API
