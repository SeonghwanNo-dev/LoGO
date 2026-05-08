# 📊 Data Module Guide

This module is designed for **efficient dataset management and distribution**, supporting parallel LoRA adapter training for the LoGO project.

---

## 📂 File Structure

* **`dataset_1/`**: (Outdated) Initial experimental datasets.
* **`dataset_2/`**: The primary directory containing the datasets used in the LoGO paper.
* **`dataset_2/flan_v2_huggingface.py`**: A utility script to download the FLAN-v2 dataset directly from Hugging Face and save it to the local disk
* **`data_allocation.py`**: A core script that divides the entire dataset into multiple folders to enable **Parallel Training** across multiple GPUs.
* **`data_config.py`**: A centralized configuration file for managing data module parameters, such as download paths, splitting logic, and target directories.

---

## ⚙️ Configuration

Before starting the training process, you must manually update the parameters in `data_config.py`. This file contains two primary configuration classes:

### 1. `Dataset_2_Config`

This class manages the initial acquisition and storage of the raw dataset.

* **`huggingface_dataset_ID`**: The source repository on Hugging Face (default: `"lorahub/flanv2"`).
* **`save_directory`**: The local path where the downloaded dataset will be stored (e.g., `./Data/dataset_2/local_flan_v2`).
* **`target_data_txt`**: Path to a metadata or temporary text file used during the download/filtering process.

### 2. `DataAllocationConfig`

This class defines how the raw dataset is partitioned for parallel training across multiple GPUs.

* **`base_path`**: The source directory containing the full dataset (should match `Dataset_2_Config.save_directory`).
* **`split_num`**: The total number of partitions to create (e.g., `2` for dual-GPU setups).
* **`target_X_path`**: The specific destination paths for each split (e.g., `target_1_path`, `target_2_path`).
> **Note:** Each training process will be assigned to one of these paths to ensure task isolation.

---

## 🛠 Usage

### 1. Update Configurations

Modify the parameters in `data_config.py` to match your current system environment

### 2. Download Dataset

Retrieve the raw data from Hugging Face and store it locally:

```bash
python3 ./Data/dataset_2/flan_v2_huggingface.py

```

### 3. Allocate Data for Parallel Training

Execute the allocation script to partition the dataset into separate task folders. This ensures that each training process can reference an independent data directory, preventing conflicts during parallel execution.

```bash
python3 ./Data/data_allocation.py

```
