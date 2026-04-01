import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict

def get_score_from_block(block, task_name):
    """
    블록 내에서 가장 적절한 대표 점수를 추출합니다.
    순위: acc_norm -> acc -> Final Score (Acc)
    """
    lines = block.split('\n')
    table_rows = []
    
    # 1. 테이블 형태의 데이터 행 추출 (| 로 구분된 행)
    for line in lines:
        if line.count('|') >= 5:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 8: # 유효한 데이터 행으로 간주
                table_rows.append(parts)
    
    # 2. leaderboard_bbh의 경우 'leaderboard_bbh' 이름이 적힌 행의 acc_norm을 우선함
    if "leaderboard_bbh" in task_name:
        for row in table_rows:
            # 첫 번째 컬럼이 정확히 task_name이거나 Metric 컬럼(index 5)이 acc_norm인 경우
            if row[1] == "leaderboard_bbh" and "acc_norm" in row[5]:
                try: return float(row[7])
                except: continue

    # 3. 일반적인 태스크에서 acc_norm 찾기 (보통 index 5가 Metric, index 7이 Value)
    for row in table_rows:
        if "acc_norm" in row[5]:
            try: return float(row[7])
            except: continue
            
    # 4. acc_norm이 없으면 acc 찾기
    for row in table_rows:
        if "acc" in row[5]:
            try: return float(row[7])
            except: continue

    # 5. 테이블이 없거나 못 찾은 경우 Final Score (Acc) 라인에서 추출
    fs_match = re.search(r"Final Score \(Acc\):\s*([\d.]+)", block)
    if fs_match:
        return float(fs_match.group(1))
        
    return None

def parse_and_plot(file_path):
    results = defaultdict(lambda: {"top_k": [], "score": []})
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 블록 단위로 나누기
    blocks = content.split("============================================================")
    
    for block in blocks:
        if not block.strip(): continue
        
        # Task와 Top_K 정보 추출
        task_info = re.search(r"Task:\s*(\S+).*?Top_?K:\s*(\w+)", block, re.IGNORECASE)
        if task_info:
            task_name = task_info.group(1)
            top_k_value = task_info.group(2)
            
        try:
            top_k = int(top_k_value)
        except ValueError:
            top_k = top_k_value
            
            score = get_score_from_block(block, task_name)
            
            if score is not None:
                results[task_name]["top_k"].append(top_k)
                results[task_name]["score"].append(score)

    # 시각화 설정
    sns.set_theme(style="whitegrid")
    
    for task_name, data in results.items():
        if not data["top_k"]: continue
        
        df = pd.DataFrame(data).sort_values("top_k")
        df['top_k_str'] = df['top_k'].astype(str) # 범주형 변환
        
        plt.figure(figsize=(9, 6))
        
        # 바 그래프
        sns.barplot(data=df, x='top_k_str', y='score', alpha=0.3, color='#3498db')
        
        # 선 그래프 (중앙 정렬을 위해 range 사용)
        plt.plot(range(len(df)), df['score'], marker='o', markersize=10, 
                 linewidth=2.5, color='#e74c3c')
        
        # 수치 표시 (중앙 정렬)
        for i, score in enumerate(df['score']):
            plt.text(i, score, f"{score:.4f}", 
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='#c0392b')

        plt.title(f"Benchmark Result: {task_name.upper()}", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Top_K", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        
        # Y축 자동 조절 (점수들이 잘 보이게)
        y_min, y_max = df['score'].min(), df['score'].max()
        plt.ylim(y_min * 0.98, min(1.0, y_max * 1.02)) 

        file_name = f"log_result_{task_name}.png"
        plt.tight_layout()
        plt.savefig(file_name, dpi=300)
        plt.close()
        print(f"📊 {task_name} 그래프 생성 완료 (값 확인: {df['score'].tolist()})")


if __name__ == "__main__":
    # 파일이 실제로 존재하는지 확인 후 실행하세요.
    file_path = "./logo_result.txt"
    try:
        parse_and_plot(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} 파일을 찾을 수 없습니다.")