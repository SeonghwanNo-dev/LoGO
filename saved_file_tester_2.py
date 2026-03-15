import re
import os

# Configuration: Paths and Files
base_paths = [
    # "./dataset_2/local_flan_v2_1",
    # "./dataset_2/local_flan_v2_2",
    "./dataset_2/local_flan_v2_3",
    "./dataset_2/local_flan_v2_4"
]
weights_paths = [
    "./lora_adapters_fine_tuned/weights/chunk_1",
    "./lora_adapters_fine_tuned/weights/chunk_2"
]
output_file = "./lora_adapters_fine_tuned/saved_file_test_results_2.txt"
done_list_file = "./lora_adapters_fine_tuned/done_tasks.txt"

def cross_check_log_and_folders():
    # 1. Retrieve the list of actual dataset folders (format: ~~.json)
    actual_folders = set()
    for path in base_paths:
        if os.path.exists(path):
            # Only include directories within the base paths
            folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            actual_folders.update(folders)
        else:
            print(f"⚠️ Warning: Path not found: {path}. Skipping...")

    # 2. Extract filenames from weight chunk folders (filtering for ~~.json)
    logged_names = set()
    for w_path in weights_paths:
        if os.path.exists(w_path):
            files = os.listdir(w_path)
            for filename in files:
                # Extracts the '~~.json' pattern from .safetensors or other extensions
                # Example: "wiki_qa.json_adapter_model.safetensors" -> "wiki_qa.json"
                match = re.search(r"^(.*?\.json)", filename)
                if match:
                    logged_names.add(match.group(1))
        else:
            print(f"⚠️ Warning: Folder not found: {w_path}. Skipping...")

    # 3. Compare and generate results
    results = []
    done_tasks = []  # List for successful items (exists in both weights and folders)
    
    results.append("=== Cross-Check: Weight Files vs Original Folders ===\n")

    # Get a sorted list of all unique names from both sets
    all_names = sorted(list(logged_names | actual_folders))

    found_count = 0
    missing_folder = 0
    missing_weight = 0

    for name in all_names:
        in_log = name in logged_names
        in_folder = name in actual_folders

        if in_log and in_folder:
            status = "[MATCH] Weight-Folder synchronized"
            found_count += 1
            done_tasks.append(name)
        elif in_log and not in_folder:
            status = "[⚠️ FOLDER MISSING] Weight exists, but original folder is missing"
            missing_folder += 1
        elif not in_log and in_folder:
            status = "[❓ WEIGHT MISSING] Folder exists, but weight file is missing"
            missing_weight += 1
        
        results.append(f"{status} : {name}")

    # 4. Append Summary Information
    results.append(f"\n--- Summary ---")
    results.append(f"Total Unique Items: {len(all_names)}")
    results.append(f"Matched (Complete): {found_count}")
    results.append(f"Missing Folders: {missing_folder}")
    results.append(f"Missing Weights: {missing_weight}")

    # 5. Save results to files
    try:
        # Save detailed report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(results))

        # Save completed task list (for future reference/resumption)
        with open(done_list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(done_tasks)))
        
        print(f"✅ Comparison Complete!")
        print(f"   - Detailed Report: {output_file}")
        print(f"   - Completion List: {done_list_file} ({len(done_tasks)} items recorded)")
    except Exception as e:
        print(f"❌ Error occurred during file saving: {e}")

if __name__ == "__main__":
    cross_check_log_and_folders()