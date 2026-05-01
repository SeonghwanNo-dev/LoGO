# LoGO: LoRA on the Go
### Instance-level Dynamic LoRA Selection and Merging

This repository is an implementation of the research paper **"LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"** ([arXiv:2511.07129](https://arxiv.org/abs/2511.07129)). This project was initiated under the recommendation of my advising professor to explore practical strategies for **"Multi-LoRA Serving Optimization on NVIDIA Jetson"**, which is my current area of active research.

## 🌟 Key Research Focus
*   **Train-free Adaptation**: Focuses on a train-free approach for dynamic adapter selection, making it highly efficient for resource-constrained edge devices.
*   **Activation-based Selection**: Leverages intermediate activations to determine the most relevant LoRA adapters for each specific instance.
*   **Technical Depth**: Extensively utilizes **PyTorch's hook mechanism** to intercept and manipulate model activations.

---

## ⚙️ Project Structure

The project is organized into three primary modules, each handling a distinct stage of the machine learning pipeline:

### 1. 📂 Data Module(./Data)
*   **Role**: Responsible for data loading, preprocessing, and management.

### 2. 📂 [Inference Module](./Inference)
*   **Role**: Handles model inference and performance benchmarking.
*   **Key Components**:
    *   `logo.py`: Main inference engine refactored for modular execution.
    *   `inference_config.py`: Centralized `LogoConfig` for managing model IDs, target layers, and sampling parameters.
    *   `run_lm_eval.py`: Benchmarking tool for evaluating model accuracy across various tasks (e.g., `arc_easy`).

### 3. 📂 [Train Module](./Train)
*   **Role**: Manages parallel LoRA fine-tuning and automated resource optimization.
*   **Key Components**:
    *   `adapter_training_1.py` & `2.py`: Scripts for simultaneous dual-GPU training.
    *   `train_config.py`: Integrated configuration for training hyperparameters and disk management.
    *   `observer_1.py` & `2.py`: Disk monitoring systems that trigger automated cloud uploads to prevent storage overflow.
    *   `google_upload.py`: Automated Google Drive integration for seamless checkpoint backup.

---

## 🚀 Getting Started

### Prerequisites
*   **tmux**: Required for persistent background execution of training and monitoring sessions.
*   **Python 3.8+** with necessary PEFT and Transformers libraries.

### Basic Workflow
1.  **Configure**: Set your model paths and hyperparameters in `data_config.py`, `train_config.py`, `inference_config.py`.
2.  **Train**: Launch parallel training sessions using tmux to maximize GPU utilization.
3.  **Monitor**: Ensure observers are running to manage local disk space via automated cloud uploads.
4.  **Inference**: Deploy fine-tuned adapters using the modular inference engine in the `Inference` directory.

---

## 🛠 Tech Stack
*   **Core**: Python, PyTorch
*   **PEFT**: LoRA (Low-Rank Adaptation)
*   **DevOps**: tmux, Google Drive API
