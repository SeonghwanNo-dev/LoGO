import os

def count_folders_recursive(path):
    folder_count = 0
    for _, dirs, _ in os.walk(path):
        folder_count += len(dirs)
    return folder_count

def count_immediate_folders(path):
    # 경로 내 아이템 중 디렉토리인 것만 리스트로 만들어 길이를 잽니다.
    return len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])

# 사용 예시
target_path = './lora_adapters_fine_tuned/6_1_copy' # 여기에 경로 입력

# total_folders = count_folders_recursive(target_path)
# print(f"전체 하위 폴더 개수: {total_folders}")

immediate_folders = count_immediate_folders(target_path)
print(f"직계 하위 폴더 개수: {immediate_folders}")