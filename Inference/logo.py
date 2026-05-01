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
loaded_adapters = []
task_name = "arc_easy"
current_input_id = None
handle_q = None
handle_v = None
top_k = 2

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
    
    print(f"✅ Hooks swapped! Weights: {combined_weight.cpu().numpy().round(3)}")

    return output + delta

# 2. 통합 컨트롤러 및 적용 훅
class LoGO_controller:
    def __init__(self, adapters_list, top_k, scaling, device, dtype):
        self.scaling = scaling
        self.current_weights = torch.full((len(adapters_list),), 1.0 / len(adapters_list), device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.adapters = {
            'qa': torch.stack([a['qa'] for a in adapters_list]).to(self.device, self.dtype),
            'qb': torch.stack([a['qb'] for a in adapters_list]).to(self.device, self.dtype),
            'va': torch.stack([a['va'] for a in adapters_list]).to(self.device, self.dtype),
            'vb': torch.stack([a['vb'] for a in adapters_list]).to(self.device, self.dtype)
        }
        self.top_k = top_k


    def controller_pre_hook(self, module, args, kwargs):
        x = kwargs.get('hidden_states')
        
        seq_len = x.size(1)
        if seq_len > 1:
            # print("새로운 input 탐지, weight 재조정")
            # Attention 블록 시작 시 1회 실행
            x_last = x[:, -1:, :] # [batch, 1, d] - 마지막 토큰 기준
            
            with torch.no_grad():
                # 벡터화된 LoRA 연산 (루프 없음)
                # (x @ A.T) @ B.T 연산을 배치로 처리
                # x: [1, 1, d], A: [10, r, d], B: [10, d, r]
                
                # Q 기여도 계산
                q_delta = (x_last @ self.adapters['qa'].transpose(1, 2)) @ self.adapters['qb'].transpose(1, 2)
                q_norms = torch.norm(q_delta * self.scaling, dim=-1).squeeze() # [10]
                qw = torch.softmax(q_norms, dim=0)
                
                # V 기여도 계산
                v_delta = (x_last @ self.adapters['va'].transpose(1, 2)) @ self.adapters['vb'].transpose(1, 2)
                v_norms = torch.norm(v_delta * self.scaling, dim=-1).squeeze() # [10]
                vw = torch.softmax(v_norms, dim=0)
                
                # L2 결합 및 정규화
                combined = torch.sqrt(qw**2 + vw**2)
                top_values, top_indices = torch.topk(combined, k=self.top_k, dim=0)
                k_masked_weights = torch.zeros_like(combined)
                k_masked_weights.scatter_(0, top_indices, top_values)
                self.current_weights = k_masked_weights / (k_masked_weights.sum() + 1e-6)

    # def controller_pre_hook(self, module, args, kwargs):
    #     print("\n" + "="*50)
    #     print("Hook Triggered!")
    #     print(f"1. Args Type: {type(args)}")
    #     print(f"2. Args Length: {len(args)}")
        
    #     for i, val in enumerate(args):
    #         if torch.is_tensor(val):
    #             print(f"   - args[{i}] is Tensor: {val.shape}")
    #         else:
    #             print(f"   - args[{i}] is {type(val)}")
    #             if isinstance(val, (list, tuple)) and len(val) > 0:
    #                 print(f"     -> First element of args[{i}] is: {type(val[0])}")

    #     print(f"3. Kwargs Keys: {list(kwargs.keys())}")
    #     for k, v in kwargs.items():
    #         if torch.is_tensor(v):
    #             print(f"   - kwargs['{k}'] is Tensor: {v.shape}")
    #     print("="*50 + "\n")
        
    #     # 에러 방지를 위해 일단 여기서 리턴 (임시)
    #     return

    def apply_adapter_hook(self, module, input, output, mode='q'):
        # Q 또는 V 프로젝션 후 실행
        x = input[0] # [batch, seq, d]
        w = self.current_weights.view(-1, 1, 1) # [10, 1, 1]
        
        with torch.no_grad():
            # 모든 어댑터의 델타를 한 번에 계산 후 가중합
            a_w = self.adapters[f'{mode}a']
            b_w = self.adapters[f'{mode}b']
            
            # 배치 행렬 곱을 이용해 10개 어댑터 결과 동시 계산
            # (batch, seq, d) @ (10, d, r) -> (10, batch, seq, r)
            deltas = (x @ a_w.transpose(1, 2)) @ b_w.transpose(1, 2)
            final_delta = (deltas * w * self.scaling).sum(dim=0)
            
        return output + final_delta





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
controller = LoGO_controller(adapters_list=loaded_adapters, top_k = top_k,scaling=64/32, device = base_model.device, dtype=base_model.dtype)

attn_module = base_model.model.layers[target_layer_idx].self_attn
q_proj = attn_module.q_proj
v_proj = attn_module.v_proj

attn_module.register_forward_pre_hook(controller.controller_pre_hook, with_kwargs=True)
q_proj.register_forward_hook(partial(controller.apply_adapter_hook, mode='q'))
v_proj.register_forward_hook(partial(controller.apply_adapter_hook, mode='v'))



from lm_eval.models.huggingface import HFLM
lm_obj = HFLM(pretrained=base_model, tokenizer=tokenizer) # 이미 훅이 걸린 모델을 래핑

results = lm_eval.simple_evaluate(
    model=lm_obj,
    # model="hf",
    # model_args={
    #     "pretrained": "meta-llama/Llama-3.1-8B-Instruct",
    #     "peft": "./lora_adapters_fine_tuned/6_1/cos_e_v1.11_aligned_with_common_sense.json", # Path to your LoRA adapter
    #     "device": "cuda:0",
    # },
    tasks= task_name,           # Evaluation tasks (list format)
    num_fewshot=0,                # Number of few-shot examples (0-5 is standard)
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
    # task_results = results["results"][task_name]
    # score = task_results.get("exact_match,none") or \
    #         task_results.get("exact_match") or \
    #         task_results.get("mean_3class_f1,none")
    # if score is not None:
    #     print(f"✨ {task_name.upper()} Final Score: {score:.4f}")
    # else:
    #     print(f"✨ {task_name.upper()} Results: {task_results}")

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