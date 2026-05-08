# Train Module Guide

This module is designed for efficient **LoRA adapter training** and automated resource management within an NVIDIA Jetson environment.

---

## 📂 File Structure
*   **`lora_adapters_fine_tuned/`**: Base directory where trained adapters are stored.
*   **`train_config.py`**: Central configuration file to manage all training parameters and upload settings.
*   **`adapter_training_1.py` / `2.py`**: Main training scripts designed for parallel GPU execution.
*   **`observer_1.py` / `2.py`**: Real-time monitoring scripts to manage local disk space.
*   **`google_upload.py`**: Module for Google Drive API integration and automated file uploading.

---

## ⚙️ Configuration
Before starting the training, you must manually update the parameters in `train_config.py`:

*   **`AdapterTrainerConfig_1 / 2`**: Dataset paths, output directories, and trainer hyperparameters.
*   **`ObserverConfig_1 / 2`**: Target monitoring folders, Google Drive folder IDs, and batch size triggers.
*   **`GoogleUploadConfig`**: Specific credentials and upload trigger conditions for Google Drive.

---

## 🚀 Parallel Training (Dual GPU)
The training process is bifurcated to maximize the utilization of two available GPUs, ensuring high performance through independent execution.

*   **Process Isolation via tmux**: Each training script (`adapter_training_1.py`, `2.py`) is executed within its own dedicated **tmux session**. This ensures that training continues persistently even if the SSH connection is interrupted.
*   **Simultaneous Training**: Each script references datasets in separate folders, allowing for concurrent training without resource contention.
*   **Efficiency**: This approach significantly reduces total training time and handles larger datasets effectively by distributing the workload.

---

## 💾 Storage Management & Cloud Upload
To prevent disk overflow caused by frequent checkpointing on the Jetson device, an automated management system is implemented.

1.  **Session Isolation**: Two dedicated **tmux** sessions run `observer_1.py` and `observer_2.py` in the background, providing uninterrupted monitoring.
2.  **Real-time Monitoring**: The observers monitor their respective storage folders within `lora_adapters_fine_tuned/` for new checkpoint files.
3.  **Auto Upload**: When the number of stored files (checkpoints) reaches a **batch size of 5**, `google_upload.py` is triggered immediately.
4.  **Cloud Sync & Cleanup**: Local files are securely transferred to Google Drive. Once the upload is verified, local copies are managed to free up disk space, ensuring a stable and continuous training environment.

---

## 🛠 Usage
1.  Update configurations in `train_config.py`.
2.  Start the training processes in separate tmux sessions:
    ```bash
    tmux new -s train1 'python3 ./Train/adapter_training_1.py'
    tmux new -s train2 'python3 ./Train/adapter_training_2.py'
    ```
3.  Start the monitoring processes in background tmux sessions:
    ```bash
    tmux new -s watch1 'python3 ./Train/lora_apaters_rine_tuned/observer_1.py'
    tmux new -s watch2 'python3 ./Train/lora_apaters_rine_tuned/observer_2.py'
    ```