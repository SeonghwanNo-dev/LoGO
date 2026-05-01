import os
import re
import json
import csv

import seqio
import flan.v2.tasks


def load_dataset_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 1. Split words by comma (,) or newline (\n)
        # 2. Strip whitespace and exclude empty strings or comments starting with '#'
        raw_names = [n.strip() for n in re.split(r'[,\n]', content) if n.strip() and not n.startswith('#')]

    return list(raw_names)


if __name__ == "__main__":
    os.makedirs('./data', exist_ok=True)
    
    # 1. Load datasets addressed in LoGO paper
    base_names = load_dataset_names('./dataset_list.txt')
    print(f"Number of base names loaded: {len(base_names)}")

    # 2. Retrieve all registered tasks from 'flan/v2/flan_collection_info.csv'
    source_names = set()
    with open('flan/v2/flan_collection_info.csv', mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Skip the header row (e.g., Category, Dataset, etc.)
        
        for row in reader:
            if len(row) >= 2:
                source_names.add(row[1]) # Add the Dataset Source Name from the 2nd column (index 1)

    # Convert to a sorted list
    all_names = sorted(list(source_names))
    print(f"Number of extracted Source Datasets: {len(all_names)}")
    print("Top 10 samples:", all_names[:10])


    # 3. Dictionary to store matching results (key: base_name, value: list of matched full_names)
    matching_log = {}
    total_matched_count = 0

    for base in base_names:
        base = base.replace('-', '_')
        base = base.lower()
        matching_log[base] = []
        
        for full_name in all_names:
            if '-' in base:
                print(base)
            # Handle wildcard (*) logic
            if '*' in base:
                prefix = base.replace('_*', '')
                # if full_name.lower().startswith(prefix):
                if full_name.lower()[:5] in prefix:
                    matching_log[base].append(full_name)
            
            # Perform a loose match to ensure at least one match per dataset mentioned in the paper
            # Checks if the first 5 characters of the full name are present in the base name
            else:
                # if full_name.lower().startswith(base):
                if full_name.lower()[:5] in base:
                    matching_log[base].append(full_name)


    # --- Record results to 'matching_results_2.txt' ---
    unmatched_count = 0
    matched_count = 0
    
    with open('matching_results_2.txt', 'w', encoding='utf-8') as f:
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

    print(f"Matching results have been saved to 'matching_results_2.txt'.")
    print(f"Total number of successfully matched tasks: {total_matched_count}")
    print(f"Matching Success Rate: {matched_count} / {matched_count + unmatched_count}")


