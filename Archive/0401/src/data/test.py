import os
import re
import json
import csv

import tensorflow as tf
import seqio
import flan.v2.tasks

def load_dataset_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 1. 쉼표(,)나 줄바꿈으로 단어들을 분리
        # 2. 양쪽 공백 제거 및 빈 문자열/주석 제외
        raw_names = [n.strip() for n in re.split(r'[,\n]', content) if n.strip() and not n.startswith('#')]

    return list(raw_names)


def export_task(full_task_name):
    """실제 seqio 태스크를 JSONL로 덤프"""
    # 파일명에서 palmflan_ 및 _zs_noopt 제거하여 깔끔하게 저장
    simple_name = full_task_name.replace("palmflan_", "").replace("_zs_noopt", "")
    output_path = f'./data/{simple_name}.jsonl'
    
    if os.path.exists(output_path):
        return f"SKIP: {simple_name}"

    try:
        ds = seqio.get_mixture_or_task(full_task_name).get_dataset(
            split="train",
            num_epochs=1,
            sequence_length={'inputs': 4096, 'targets': 4096}
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in ds.as_numpy_iterator():
                f.write(json.dumps({
                    "inputs": vocab.decode(ex["inputs"]),
                    "targets": vocab.decode(ex["targets"]),
                    "task": simple_name
                }, ensure_ascii=False) + "\n")
        return f"SUCCESS: {simple_name}"
    except Exception as e:
        return f"FAIL: {simple_name} | {str(e)}"

if __name__ == "__main__":
    os.makedirs('./data', exist_ok=True)
    
    # 1. 파일에서 패턴 불러오기
    base_names = load_dataset_names('./dataset_list.txt')
    print(f"불러온 베이스 이름 개수: {len(base_names)}개")

    # 2. 전체 태스크에서 매칭되는 것만 골라내기
    source_names = set() # 중복 제거를 위해 set 사용
    with open('flan/v2/flan_collection_info.csv', mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 만약 첫 줄이 헤더(Category, Dataset...)라면 next(reader)로 한 줄 건너뜁니다.
        next(reader) 
        
        for row in reader:
            if len(row) >= 2:
                source_names.add(row[1]) # 2열(index 1)인 Dataset Source Name 추가

    # 리스트로 변환 및 정렬
    all_names = sorted(list(source_names))
    print(f"추출된 원재료(Source) 개수: {len(all_names)}")
    print("상위 10개:", all_names[:10])   

    # all_names = list(seqio.TaskRegistry.names())
    # print(all_names[:10])
    # print(len(all_names))

    # 매칭 결과를 담을 딕셔너리 (key: base_name, value: 매칭된 full_names 리스트)
    matching_log = {}
    total_matched_count = 0  # 전체 태스크 개수를 셀 변수 추가

    for base in base_names:
        base = base.replace('-', '_')
        base = base.lower()
        matching_log[base] = []  # 각 base_name별로 리스트 초기화
        
        for full_name in all_names:
            if '-' in base:
                print(base)
            # 1. 와일드카드(*)가 있는 경우: 시작 단어만 맞으면 오케이
            if '*' in base:
                prefix = base.replace('_*', '')
                # if full_name.lower().startswith(prefix):
                if full_name.lower()[:5] in prefix:
                    matching_log[base].append(full_name)
            
            # 2. 와일드카드(*)가 없는 경우: 시작 단어 맞고 + _zero_shot 포함
            else:
                # if f"{base}_zero_shot" == full_name:
                # if full_name.startswith(base) and "_zero_shot" in full_name:
                # if full_name.lower().startswith(base):
                if full_name.lower()[:5] in base:
                    matching_log[base].append(full_name)

    # --- 다른 파일(matching_results.txt)에 기록하기 ---
    with open('matching_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== 데이터셋 매칭 결과 보고서 ===\n\n")
        
        for base, matches in matching_log.items():
            f.write(f"규칙: [{base}]\n")
            if not matches:
                f.write("  -> 매칭된 태스크 없음\n")
            else:
                for m in matches:
                    f.write(f"  - {m}\n")
                    total_matched_count += 1
            f.write("-" * 50 + "\n")

    print("매칭 결과가 'matching_results.txt'에 저장되었습니다.")
    print(f"최종 매칭된 전체 태스크 개수: {total_matched_count}개")



    # # 2. 멀티프로세싱 추출 (Pool 20으로 가속)
    # with Pool(20) as p:
    #     results = p.map(export_task, target_tasks)

    # # 결과 요약
    # success_count = len([r for r in results if "SUCCESS" in r])
    # print(f"추출 성공: {success_count} / 전체: {len(target_tasks)}")


    '''
    Trouble Shooting

    import seqio
    import flan.v2.tasks
    all_names = list(seqio.TaskRegistry.names())

    - 이 방법으로 하면, 모든 가공된 버전만 나옴
    - 그러나 가공된 버전에다 논문에 제시된 데이터셋을 검색해보면 매칭되는 데이터셋이 없음 (matching_results_1.txt)
    - "-" -> "_", 소문자 바꾼 뒤 검색 등을 시도해 봤음에도 안됨..

    - 따라서 원본 데이터셋을 검색해보고, 그 variation을 데이터셋으로 선택하려고 함. (matching_results_2.txt)
    - 원본 데이터셋은 "transformer" 제외하곤, 전부 1개 이상의 데이터셋과 매칭됨
    - 완전히 같은 데이터셋을 선택할 수 없음. 존재하지 않으므로.

    - 시도
    그 전에! seqio와 flan 사용법 공부하기ㄴ
    1. info.csv에서 원재료 이름 확보: 논문의 키워드와 유사한 Source Name을 찾습니다.
    2. tasks.py에서 실제 이름 매칭: 해당 Source Name을 사용하는 실제 Task Name들을 all_names에서 검색합니다.
    3. 부분 일치 검색: if "wiki_hop" in name 같은 식으로 이름의 일부분만 걸려도 일단 다 리스트업한 뒤 논문의 설명과 대조합니다.



    '''