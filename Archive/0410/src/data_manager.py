import os
import shutil

# 경로 설정
TARGET_DIR = "./data/local_flan_v2_selected_100"  # 파일을 삭제할 대상 폴더
DONE_TASKS_PATH = "./lora_adapters_fine_tuned/6_1"               # 완료된 항목들이 들어있는 폴더

def remove_duplicate_tasks(target_dir, done_dir):
    # 1. 폴더 존재 여부 확인
    if not os.path.exists(target_dir):
        print(f"오류: 대상 폴더 '{target_dir}'가 존재하지 않습니다.")
        return
    if not os.path.exists(done_dir):
        print(f"오류: 완료 폴더 '{done_dir}'가 존재하지 않습니다.")
        return

    # 2. 완료된 폴더의 항목 리스트업 (숨김 파일 제외)
    # set 자료형을 사용하여 검색 속도를 최적화합니다.
    done_items = {f for f in os.listdir(done_dir) if not f.startswith('.')}
    print(f"--- 기준 폴더에서 {len(done_items)}개의 완료 항목을 로드했습니다. ---")

    # 3. 대상 폴더를 순회하며 중복된 항목 삭제
    deleted_count = 0
    all_target_items = os.listdir(target_dir)
    
    for item in all_target_items:
        # 숨김 파일 무시
        if item.startswith('.'):
            continue
            
        # 대상 폴더의 항목이 완료 폴더에도 존재한다면 삭제 실행
        if item in done_items:
            item_path = os.path.join(target_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  # 파일 또는 심볼릭 링크 삭제
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # 폴더 삭제
                deleted_count += 1
                print(f"삭제됨: {item}")
            except Exception as e:
                print(f"삭제 실패: {item} - {e}")

    print("\n" + "="*40)
    print(f"결과: 총 {deleted_count}개의 중복 항목이 제거되었습니다.")
    print(f"현재 '{target_dir}'에 남은 항목: {len(os.listdir(target_dir))}개")
    print("="*40)

if __name__ == "__main__":
    remove_duplicate_tasks(TARGET_DIR, DONE_TASKS_PATH)