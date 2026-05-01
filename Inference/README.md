# Inference Module Guide

This module manages the inference process for the **LoGO** model.

---

## 📂 File Structure
*   **`inference_config.py`**: Contains the `logoConfig` class, which centralizes all inference-related settings and hyperparameters.
*   **`logo.py`**: The primary inference script. It initializes the model and generates outputs based on the attributes defined in `logoConfig`.
*   **`run_lm_eval.py`**: Used for library testing and performance benchmarking.

---

## ⚙️ Configuration
Before running inference, you should manage your settings in `inference_config.py`. The `logoConfig` class allows you to easily modify:

**Configuration Details:**
*   **Model Setup**: Defines the base model (`model_id`) and the specific layer (`target_layer_idx`) for activation.
*   **Inference Strategy**: Controls the generation behavior using `top_k` adapter sampling.
*   **Adapter Management**: Maintains a list of paths to multiple fine-tuned LoRA adapters, allowing for modular weight application.
*   **Evaluation Focus**: Specifies the benchmark task (`task_name`) to be used during the `run_lm_eval.py` process.
---

## 🛠 Usage
1.  Open `inference_config.py` and set your desired parameters within the `logoConfig` class.
2.  Run the inference script:
    ```bash
    python3 ./Inference/logo.py
    ```
3.  (Optional) Run evaluation for testing:
    ```bash
    python3 ./Inference/run_lm_eval.py
    ```
