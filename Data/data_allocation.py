import os
import shutil

# Define the source and target paths
base_path = "./local_flan_v2"
target_1 = "./local_flan_v2_1"
target_2 = "./local_flan_v2_2"

# 1. Create target directories (exist_ok=True prevents error if they already exist)
os.makedirs(target_1, exist_ok=True)
os.makedirs(target_2, exist_ok=True)

# 2. Get the list of subdirectories and sort them to ensure consistent order
# os.listdir order is arbitrary, so sorted() is essential
task_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

# 3. Calculate the midpoint to split the list in half
mid_point = len(task_folders) // 2

# 4. Split the list using slicing
first_half = task_folders[:mid_point]
second_half = task_folders[mid_point:]

# 5. Define a helper function to move directories
def move_tasks(tasks, destination):
    for task in tasks:
        source_dir = os.path.join(base_path, task)
        dest_dir = os.path.join(destination, task)
        
        # Move the entire directory to the new location
        shutil.move(source_dir, dest_dir)
        print(f"Successfully moved: {task} -> {destination}")

# 6. Execute the move operations
move_tasks(first_half, target_1)
move_tasks(second_half, target_2)

print(f"\nTask completed! (Part 1: {len(first_half)}, Part 2: {len(second_half)})")

# Task completed! (Part 1: 120, Part 2: 121)