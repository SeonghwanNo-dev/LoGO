import os
import shutil
from data_config import DataAllocationConfig

def split_tasks(config):
    if config.split_num != 2: 
        print("Not Implemented yet..")

    # 1. Create target directories (exist_ok=True prevents error if they already exist)
    os.makedirs(config.target_1_path, exist_ok=True)
    os.makedirs(config.target_2_path, exist_ok=True)

    # 2. Get the list of subdirectories and sort them to ensure consistent order
    # os.listdir order is arbitrary, so sorted() is essential
    task_folders = sorted([f for f in os.listdir(config.base_path) if os.path.isdir(os.path.join(config.base_path, f))])

    # 3. Calculate the midpoint to split the list in half
    mid_point = len(task_folders) // 2

    # 4. Split the list using slicing
    first_half = task_folders[:mid_point]
    second_half = task_folders[mid_point:]

    return first_half, second_half

def move_tasks(base_path, tasks, destination):
    for task in tasks:
        source_dir = os.path.join(config.base_path, task)
        dest_dir = os.path.join(destination, task)
        
        # Move the entire directory to the new location
        shutil.move(source_dir, dest_dir)
        print(f"Successfully moved: {task} -> {destination}")

if __name__ == "__main__":
    # setting
    config = DataAllocationConfig()

    first_half, second_half = split_tasks(config)
    move_tasks(config.base_path, first_half, config.target_1_path)
    move_tasks(config.base_path, second_half, config.target_2_path)

    print(f"\nTask completed! (Part 1: {len(first_half)}, Part 2: {len(second_half)})")
    # Task completed! (Part 1: 120, Part 2: 121)