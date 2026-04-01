import os
import re
import json
import csv

import tensorflow as tf
import seqio
import flan.v2.tasks
# import flan.v2.constants_niv2
# import flan.v2.constants_t0
# import flan.v2.constants
# import flan.v2.few_shot
# import flan.v2.flan_templates_branched
# import flan.v2.mixtures
# import flan.v2.mixtures_utils
# import flan.v2.postprocessors
# import flan.v2.preprocessors
# import flan.v2.run_example
# import flan.v2.task_configs_v1
# import flan.v2.task_configs
# import flan.v2.templates
# import flan.v2.utils_test
# import flan.v2.utils


def load_dataset_names(file_path):
    """
    Loads dataset names from a text file.
    Splits by commas or newlines and filters out comments/empty lines.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 1. Split words by comma (,) or newline (\n)
        # 2. Strip whitespace and exclude empty strings or comments starting with '#'
        raw_names = [n.strip() for n in re.split(r'[,\n]', content) if n.strip() and not n.startswith('#')]

    return list(raw_names)


if __name__ == "__main__":
    os.makedirs('./data', exist_ok=True)
    
    # 1. Manually specify a base task that is inconsistent with the paper for testing
    # base_names = load_dataset_names('./dataset_list.txt')
    # print(f"불러온 베이스 이름 개수: {len(base_names)}개")
    base_names = ["wiki_lingua_english_en"]

    # 2. Retrieve all registered tasks from SeqIO
    all_names = list(seqio.TaskRegistry.names())
    print(all_names[:10])
    print(len(all_names))

    # 3. Dictionary to store matching results (key: base_name, value: list of matched full_names)
    matching_log = {}
    total_matched_count = 0

    for base in base_names:
        base = base.replace('-', '_')
        base = base.lower()
        matching_log[base] = []
        
        for full_name in all_names:
            full_name = full_name.replace('-', '_')
            full_name = full_name.lower()

            # Handle wildcard (*) logic
            if '*' in base:
                prefix = base.replace('_*', '')
                # Perform a loose match to ensure at least one match per dataset mentioned in the paper
                # Checks if the first 5 characters of the full name are present in the base name
                if full_name[:5] in prefix:
                    matching_log[base].append(full_name)
            else:
                if full_name[:5] in base:
                    matching_log[base].append(full_name)
            
    # --- Record results to 'matching_results_3.txt' ---
    unmatched_count = 0
    matched_count = 0
    
    with open('matching_results_3.txt', 'w', encoding='utf-8') as f:
        f.write("=== Dataset Matching Results Report ===\n\n")
        
        for base, matches in matching_log.items():
            f.write(f"Matching Rule: [{base}]\n")
            if not matches:
                f.write("  -> No matching tasks found\n")
                unmatched_count += 1
            else:
                for m in matches:
                    f.write(f"  - {m}\n")
                    total_matched_count += 1
                matched_count += 1
            f.write("-" * 50 + "\n")
        f.write(f"Total number of successfully matched tasks: {total_matched_count}\n")
        f.write(f"Matching Success Rate: {matched_count} / {matched_count + unmatched_count}")
        
    print(f"Matching results have been saved to 'matching_results_3.txt'.")
    print(f"Total number of successfully matched tasks: {total_matched_count}")
    print(f"Matching Success Rate: {matched_count} / {matched_count + unmatched_count}")

