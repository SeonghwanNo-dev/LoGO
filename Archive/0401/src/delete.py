import shutil
import os
from pathlib import Path

# 1. 경로 설정
dataset_base_paths = [
    "./dataset_2/local_flan_v2_1",
    "./dataset_2/local_flan_v2_2"
]

weights_chunk_paths = [
    "./lora_adapters_fine_tuned/weights/chunk_1",
    "./lora_adapters_fine_tuned/weights/chunk_2"
]

def cleanup_duplicate_weights():
    # 2. 삭제 대상 이름(데이터셋 폴더명) 수집
    target_names = set()
    for base in dataset_base_paths:
        path = Path(base)
        if path.exists():
            folders = [d.name for d in path.iterdir() if d.is_dir()]
            target_names.update(folders)
            print(f"✅ Collected {len(folders)} names from {base}")
        else:
            print(f"⚠️ Warning: Dataset path not found: {base}")

    if not target_names:
        print("❌ No dataset folders found. Aborting cleanup.")
        return

    print(f"🔍 Total unique targets to delete: {len(target_names)}")
    print("-" * 50)

    # 3. Weights 폴더 전수조사 및 삭제
    deleted_count = 0
    for chunk in weights_chunk_paths:
        chunk_path = Path(chunk)
        if not chunk_path.exists():
            continue

        # 해당 청크 안의 모든 폴더 확인
        for weight_dir in [d for d in chunk_path.iterdir() if d.is_dir()]:
            # 폴더 이름이 데이터셋 이름 목록에 있는지 확인
            # (만약 '~~.json_2' 처럼 접미사가 붙은 것도 지우고 싶다면 추가 로직이 필요함)
            if weight_dir.name in target_names:
                try:
                    print(f"🔥 Deleting: {weight_dir.absolute()}")
                    shutil.rmtree(weight_dir)
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Failed to delete {weight_dir.name}: {e}")

    print("-" * 50)
    print(f"✨ Cleanup Complete! Total deleted: {deleted_count} folders.")

if __name__ == "__main__":
    # 실행 전 정말 삭제할 것인지 확인하는 메시지
    confirm = input("⚠️ This will permanently delete folders in weights/chunk_x that match dataset names. Proceed? (y/n): ")
    if confirm.lower() == 'y':
        cleanup_duplicate_weights()
    else:
        print("Cleanup cancelled.")