import re
import os
from datasets import load_dataset
from huggingface_hub import list_repo_files

def load_target_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 1. Split words by comma (,) or newline (\n)
        # 2. Strip whitespace and exclude empty strings or comments starting with '#'
        raw_names = [n.strip() for n in re.split(r'[,\n]', content) if n.strip() and not n.startswith('#')]

    return list(raw_names)

def validate_targets(repo_id, target_names, log_file="result.txt"):
    """
    Validates targets and maps each to a LIST of matching remote files.
    """
    print(f"Validating targets against remote repository: {repo_id}...")
    file_counter = 0
    try:
        # Fetch all available files from the Hugging Face repo
        remote_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        valid_targets_1 = {}  # Format: { 'target_name': ['file1.json', 'file2.json'] }
        valid_targets_2 = {}
        invalid_targets = [] # List for names with ZERO matches
        
        for name in target_names:
            
            # Handle wildcard (*) logic
            if '*' in name:
                prefix = name.replace('_*', '')
            else:
                prefix = name
            
            # Find all remote files that contain the target name
            matches = []
            for f in remote_files:
                if f.startswith(prefix):
                    matches.append(f)
                    file_counter+=1
            
            if matches:
                if '*' in name:
                    # Store all matched files in a list as the dictionary value
                    valid_targets_1[name] = matches
                else:
                    valid_targets_2[name] = matches
            else:
                invalid_targets.append(name)
        
        # Write report to result.txt
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"--- Validation Report for {repo_id} ---\n")
            f.write(f"Total processed: {len(target_names)}\n")
            f.write(f"Valid (with matches): {len(valid_targets_1)+len(valid_targets_2)}\n")
            f.write(f"Invalid (zero matches): {len(invalid_targets)}\n")
            f.write(f"The number of total files: {file_counter}\n\n")
            
            f.write("[MATCHED TARGETS with *]\n")
            for name, files in valid_targets_1.items():
                # Write how many files were matched for each target
                f.write(f"{name} ({len(files)} files):\n")
                for file_path in files:
                    f.write(f"  - {file_path}\n")
            f.write(f"\n\n")
                    
            f.write("[MATCHED TARGETS without *]\n")
            for name, files in valid_targets_2.items():
                # Write how many files were matched for each target
                f.write(f"{name} ({len(files)} files):\n")
                for file_path in files:
                    f.write(f"  - {file_path}\n")
            f.write(f"\n\n")
            
            f.write("\n[INVALID TARGETS]\n")
            for it in invalid_targets:
                f.write(f"{it}\n")
            f.write(f"\n\n")
        
        print(f"Validation results saved to {log_file}")
        print(f"Verified: {len(valid_targets_1)+len(valid_targets_2)} targets found with matching files.")
        print(f"The number of total files: {file_counter}")
        return (valid_targets_1, valid_targets_2)

    except Exception as e:
        print(f"Error during validation: {e}")
        return {}

def download_datasets(repo_id, verified_names, base_save_path):
    """
    Downloads and saves the datasets from the verified list.
    """
    success_count = 0
    errored_files = []
    for name, files in verified_names.items():
        for file in files:
            try:
                print(f"Download: {file} from {name}...", flush=True)
                save_path = os.path.join(base_save_path, file)    
                if os.path.exists(save_path):
                    print(f"skipped: {save_path}\n")
                    continue
                else:
                    print(f"download: {save_path} ")
                    # Load & Save
                    dataset = load_dataset(repo_id, data_files=file, download_mode="force_redownload")
                    dataset.save_to_disk(save_path)
                
                print("▶ DONE")
                success_count += 1
            except Exception as e:
                errored_files.append(save_path)
                print(f"▶ {save_path}FAILED: {e}")
    return errored_files, success_count

if __name__ == "__main__":
    # Settings
    LIST_FILE = './dataset_list.txt'
    HF_DATASET_ID = "lorahub/flanv2"
    SAVE_DIR = "./local_flan_v2"

    # 1. Load your local list
    raw_targets = load_target_names(LIST_FILE)

    # 2. Separate function to validate targets before any download
    verified_targets_1, verified_targets_2 = validate_targets(repo_id = HF_DATASET_ID, target_names = raw_targets, log_file="result.txt")

    # 3. Download only the verified ones
    print(f"\nStarting download for {len(verified_targets_1)+len(verified_targets_2)} verified tasks...")
    errored_files_1, count_1 = download_datasets(HF_DATASET_ID, verified_targets_1, SAVE_DIR)
    print(f"Failed in partition 1: {errored_files_1}\n\n")
    errored_files_2, count_2 = download_datasets(HF_DATASET_ID, verified_targets_2, SAVE_DIR)
    print(f"Failed at partition 2: {errored_files_2}\n")
    print(f"\nFinal Summary: {count_1}+{count_2} datasets successfully saved to {SAVE_DIR}.")
    
    
    # # 4. Load from the local directory
    # dataset = load_from_disk(SAVE_DIR)
    
    
'''
[root/dataset]
- I attempted to cross-reference the names by directly calling them using seqIO and flan, but the comparison failed
- I contacted the lead author of the paper via email and identified the dataset source
- The source: https://huggingface.co/datasets/lorahub/flanv2
[root/dataset_2]
- Using the huggingface's datasets
- I organized the datasets into two partitions: Partition 1 (datasets containing *) and Partition 2 (datasets without *)
- Status: 241 datasets are downloaded
(
    Total: 260
    Not found (step 2): 2
    (hotpotqa, glue_rte)
    Failed (step 3): 15
    ('./local_flan_v2/coqa:1.0.0.json', './local_flan_v2/drop:2.0.0.json', './local_flan_v2/openbookqa:0.1.0.json', './local_flan_v2/quac:1.0.0.json', './local_flan_v2/snli:1.1.0.json', './local_flan_v2/sentiment140:1.0.0.json', './local_flan_v2/trec:1.0.0.json', './local_flan_v2/hellaswag:1.1.0.json', './local_flan_v2/lambada:1.0.0.json', './local_flan_v2/piqa:1.0.0.json', './local_flan_v2/winogrande:1.1.0.json', './local_flan_v2/aeslc:1.0.0.json', './local_flan_v2/samsum:1.0.0.json', './local_flan_v2/gigaword:1.2.0.json', './local_flan_v2/huggingface:xsum.json')
)
'''