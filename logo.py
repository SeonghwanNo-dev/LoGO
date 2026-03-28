import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd

import lm_eval
import gc
from lm_eval.utils import make_table

# 1. 모델 및 토크나이저 로드
model_id = "meta-llama/Llama-3.1-8B-Instruct"
target_layer_idx = 31
num_adapters = 10
q_results = []
v_results = []
loaded_adapters = []
task_name = "arc_easy"
current_input_id = None
dirty = 0
handle_q = None
handle_v = None

adapter_paths = ["lora_adapters_fine_tuned/6_1/amazon_polarity_Is_this_product_review_positive.json/adapter_model.safetensors",
                 "lora_adapters_fine_tuned/6_1/anli_r3:0.1.0.json/adapter_model.safetensors",
                 "lora_adapters_fine_tuned/6_1/cos_e_v1.11_aligned_with_common_sense.json/adapter_model.safetensors",
                 "lora_adapters_fine_tuned/6_1/glue_mnli:2.0.0.json/adapter_model.safetensors",
                 "lora_adapters_fine_tuned/6_1/sciq_Multiple_Choice.json/adapter_model.safetensors",
                 "lora_adapters_fine_tuned/6_1/social_i_qa_Generate_answer.json/adapter_model.safetensors",
                 "lora_adapters_fine_tuned/6_1/super_glue_cb:1.0.2.json/adapter_model.safetensors",
                 "lora_adapters_fine_tuned/6_1/trivia_qa_rc:1.1.0.json/adapter_model.safetensors",
                 "lora_adapters_fine_tuned/6_1/wiqa_effect_with_string_answer.json/adapter_model.safetensors",
                 "lora_adapters_fine_tuned/6_1/yelp_polarity_reviews:0.2.0.json/adapter_model.safetensors",
                ]


# 2. 훅 함수 정의
def calculate_weight_hook(module, input, output, adapters, result_list, mode="q"):
    x = input[0][:, -1:, :]
    temp = []
    alpha = 64
    r = 32
    scaling = alpha / r
    
    with torch.no_grad():
        for i in range(len(adapters)):
            # 어댑터에서 q 또는 v에 맞는 가중치 선택
            a_key, b_key = (f'{mode}a', f'{mode}b')
            a_w = adapters[i][a_key]
            b_w = adapters[i][b_key]
            
            # LoRA 연산: (x @ A.T) @ B.T
            delta = scaling * ((x @ a_w.T) @ b_w.T)
            norm = torch.norm(delta)
            temp.append(norm)
        norms_tensor = torch.stack(temp)
        weights = torch.softmax(norms_tensor, dim=0)
        result_list.clear()
        result_list.append(weights.cpu().numpy())
    return output

def apply_hook(module, input, output, weight, adapters, mode="q"):
    temp = []
    alpha = 64
    r = 32
    scaling = alpha / r
    x = input[0]
    
    delta = 0
    with torch.no_grad():
        for i in range(len(adapters)):
            a_key, b_key = (f'{mode}a', f'{mode}b')
            a_w = adapters[i][a_key]
            b_w = adapters[i][b_key]

            # LoRA 연산: (x @ A.T) @ B.T
            delta += scaling * weight[i] * ((x @ a_w.T) @ b_w.T)

    return output + delta
s
def controller(moduel, input, output):
    global current_input_id, dirty, handle_q, handle_v, q_results, v_results
    current_len = input[0].shape[1]
    
    # 1. 리셋 로직 (새로운 샘플 시작 시)
    if dirty == 0 or not torch.equal(current_input_id, input[0][0]):
        if handle_q: handle_q.remove()
        if handle_v: handle_v.remove()
        current_input_id = input[0][0].clone()
        
        # 리스트 초기화
        q_results.clear()
        v_results.clear()

        q_module = base_model.model.layers[target_layer_idx].self_attn.q_proj
        v_module = base_model.model.layers[target_layer_idx].self_attn.v_proj

        handle_q = q_module.register_forward_hook(
            partial(calculate_weight_hook, adapters=loaded_adapters, result_list=q_results, mode="q")
        )
        handle_v = v_module.register_forward_hook(
            partial(calculate_weight_hook, adapters=loaded_adapters, result_list=v_results, mode="v")
        )
        dirty = 1
        
    # 2. 적용 로직 (분석 데이터가 존재할 때)
    elif dirty == 1 and len(q_results) > 0:
        handle_q.remove()
        handle_v.remove()

        # ✅ 중요: 장치(Device) 및 타입(Dtype) 일치
        qw = torch.tensor(q_results[0]).to(base_model.device, dtype=base_model.dtype)
        vw = torch.tensor(v_results[0]).to(base_model.device, dtype=base_model.dtype)
        
        # L2 Norm 기반 결합
        combined_weight = torch.sqrt(qw**2 + vw**2)
        combined_weight = combined_weight / combined_weight.sum()

        q_module = base_model.model.layers[target_layer_idx].self_attn.q_proj
        v_module = base_model.model.layers[target_layer_idx].self_attn.v_proj

        handle_q = q_module.register_forward_hook(
            partial(apply_hook, adapters=loaded_adapters, weight=combined_weight, mode="q")
        )
        handle_v = v_module.register_forward_hook(
            partial(apply_hook, adapters=loaded_adapters, weight=combined_weight, mode="v")
        )
        print(f"\n✅ Hooks swapped! Weights: {combined_weight.cpu().numpy().round(3)}")
        dirty = 2

    

# 3. 가중치 로드 및 어댑터 리스트 생성
q_lora_a_key = f"base_model.model.model.layers.{target_layer_idx}.self_attn.q_proj.lora_A.weight"
q_lora_b_key = f"base_model.model.model.layers.{target_layer_idx}.self_attn.q_proj.lora_B.weight"
v_lora_a_key = f"base_model.model.model.layers.{target_layer_idx}.self_attn.v_proj.lora_A.weight"
v_lora_b_key = f"base_model.model.model.layers.{target_layer_idx}.self_attn.v_proj.lora_B.weight"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

for path in adapter_paths:
    w = load_file(path)
    # 필요한 것만 추려서 GPU로 이동
    loaded_adapters.append({
        'qa': w[q_lora_a_key].to(device=base_model.device, dtype=base_model.dtype),
        'qb': w[q_lora_b_key].to(device=base_model.device, dtype=base_model.dtype),
        'va': w[v_lora_a_key].to(device=base_model.device, dtype=base_model.dtype),
        'vb': w[v_lora_b_key].to(device=base_model.device, dtype=base_model.dtype)
    })


# 4. 훅 등록
q_module = base_model.model.layers[target_layer_idx].self_attn.q_proj
handle_controller = q_module.register_forward_hook(
    partial(controller)
)

from lm_eval.models.huggingface import HFLM
lm_obj = HFLM(pretrained=base_model, tokenizer=tokenizer) # 이미 훅이 걸린 모델을 래핑

results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks= task_name,           # Evaluation tasks (list format)
    num_fewshot=5,                # Number of few-shot examples (0-5 is standard)
    batch_size=1,                 # Adjust based on your GPU VRAM
    # limit=10                    # Uncomment this line for a quick "sanity check" (only 10 samples)
)

# 2. Output Results
# Print the results in a formatted table as seen in the CLI version
print(make_table(results))

# 3. Extract Specific Metrics
# Example: Extracting the accuracy score for a specific task
if task_name in results["results"]:
    # Accessing the 'acc,none' metric (standard accuracy)
    score = results["results"][task_name].get("acc,none") or results["results"][task_name].get("acc")
    print(f"✨ {task_name.upper()} Final Score: {score:.4f}")

# Optional: Clean up GPU memory after evaluation
gc.collect()
torch.cuda.empty_cache()

# ## 시각화

# # q_results 데이터를 numpy array로 변환 (Time, Adapter_Index)
# data_q = np.array(q_results) 
# data_v = np.array(v_results)

# plt.figure(figsize=(15, 6))
# for i in range(num_adapters):
#     plt.plot(data_q[:, i], label=f"q_Adapter {i}")
#     plt.plot(data_v[:, i], label=f"v_Adapter {i}")

# plt.title("Adapter Contribution over Time (Tokens)")
# plt.xlabel("Token Position")
# plt.ylabel("Softmax Weight")
# plt.legend(loc='upper right')



# # plt.show() 대신 아래 코드를 넣으세요
# plt.savefig('adapter_analysis.png', dpi=300) 
# print("그래프가 'adapter_analysis.png'로 저장되었습니다.")

# # 데이터 준비 (예시: result_list에서 가져온 numpy array라고 가정)
# # q_res, v_res 크기: [num_tokens, 10]

# # 1. 데이터 준비
# q_res = np.array(q_results) # (48, 10)
# v_res = np.array(v_results) # (48, 10)

# # 2. 10x10 상관계수 행렬 계산 (48개 토큰의 흐름을 샘플로 사용)
# # Q_i 어댑터의 48개 값과 V_j 어댑터의 48개 값 사이의 상관관계
# corr_matrix = np.zeros((10, 10))
# for i in range(10):
#     for j in range(10):
#         # 전체 시점(:, i) 데이터를 넣어야 상관계수가 계산됩니다.
#         c = np.corrcoef(q_res[:, i], v_res[:, j])[0, 1]
#         corr_matrix[i, j] = c

# # 전체 데이터에 대한 통합 상관계수 (ax2용)
# corr_total = np.corrcoef(q_res.flatten(), v_res.flatten())[0, 1]

# # 3. 시각화
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# # 왼쪽: 10x10 히트맵
# # ax=ax1을 명시해줘야 왼쪽 칸에 그려집니다.
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0,
#             xticklabels=[f'V_{i}' for i in range(10)],
#             yticklabels=[f'Q_{i}' for i in range(10)],
#             ax=ax1)
# ax1.set_title("10x10 Adapter Correlation (Q vs V)")
# ax1.set_xlabel("Value Adapters")
# ax1.set_ylabel("Query Adapters")

# # 각 어댑터(0~9)별로 48개 시점 데이터의 Q-V 상관계수 계산
# adapter_corrs = []
# for i in range(10):
#     # i번째 어댑터의 Q 변화(48개)와 V 변화(48개)의 상관관계
#     r = np.corrcoef(q_res[:, i], v_res[:, i])[0, 1]
#     adapter_corrs.append(r)

# # 오른쪽: 어댑터별 상관계수 막대 그래프
# colors = plt.cm.viridis(np.linspace(0, 1, 10))
# ax2.bar(range(10), adapter_corrs, color=colors)
# ax2.set_xticks(range(10))
# ax2.set_xticklabels([f'Adp {i}' for i in range(10)])
# ax2.set_title("Q-V Correlation per Adapter (Over all Tokens)")
# ax2.set_ylabel("Correlation (r)")
# ax2.set_ylim(-1, 1) # 상관계수 범위
# ax2.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.savefig('correlation_analysis.png', dpi=300)
# print("그래프가 'correlation_analysis.png'로 저장되었습니다.")