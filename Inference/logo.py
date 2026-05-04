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
from inference_config import LogoConfig

current_input_id = None
handle_q = None
handle_v = None


# 2. 통합 컨트롤러 및 적용 훅
class LoGO_controller:
    def __init__(self, adapters_list, top_k, scaling, device, dtype):
        self.scaling = scaling
        self.current_weights = torch.full((len(adapters_list),), 1.0 / len(adapters_list), device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.adapters = {
            # torch.stack은 새로운 0번 차원을 만들어 쌓는다.
            'qa': torch.stack([a['qa'] for a in adapters_list]).to(self.device, self.dtype),   
            'qb': torch.stack([a['qb'] for a in adapters_list]).to(self.device, self.dtype),
            'va': torch.stack([a['va'] for a in adapters_list]).to(self.device, self.dtype),
            'vb': torch.stack([a['vb'] for a in adapters_list]).to(self.device, self.dtype)
        }
        self.top_k = top_k


    def controller_pre_hook(self, module, args, kwargs):
        x = kwargs.get('hidden_states')
        
        batch_size = x.size(0)  # B
        seq_len = x.size(1)     # S
        if seq_len > 1:
            # print("새로운 input 탐지, weight 재조정")
            # Attention 블록 시작 시 1회 실행
            x_last = x[:, -1:, :] # [B, 1, D] - 마지막 토큰 기준
            print(f"pre_hook, x_last.shape={x_last.shape}")
            
            with torch.no_grad():
                # 벡터화된 LoRA 연산 (루프 없음)
                # (x @ A.T) @ B.T 연산을 배치로 처리
                # x: [B, 1, D], A: [N, R, D], B: [N, D, R]

                # Q 기여도 계산
                # (1, B, 1, D) @ (N, 1, D, R) @ (N, 1, R, D) -> (N, B, 1, R) @ (N, 1, R, D) -> (N, B, 1, D)
                q_delta = (x_last.unsqueeze(0) @ self.adapters['qa'].transpose(1, 2)) @ self.adapters['qb'].transpose(1, 2)
                q_norms = torch.norm(q_delta * self.scaling, dim=-1)    # [N, B, 1]
                qw = torch.softmax(q_norms.view(-1, batch_size), dim=0) # [N, B]
        
                
                # V 기여도 계산
                v_delta = (x_last.unsqueeze(0) @ self.adapters['va'].transpose(1, 2)) @ self.adapters['vb'].transpose(1, 2)
                v_norms = torch.norm(v_delta * self.scaling, dim=-1)
                vw = torch.softmax(v_norms.view(-1, batch_size), dim=0) # [N, B]
                
                # L2 결합 및 정규화
                combined = torch.sqrt(qw**2 + vw**2)    # [N, B]
                top_values, top_indices = torch.topk(combined, k=self.top_k, dim=0) # [K, B]
                k_masked_weights = torch.zeros_like(combined)   # [N, B]
                k_masked_weights.scatter_(0, top_indices, top_values)
                self.current_weights = (k_masked_weights / (k_masked_weights.sum() + 1e-6)).view(-1, batch_size, 1, 1) # [N, B, 1, 1]

    def apply_adapter_hook(self, module, input, output, mode='q'):
        # Q 또는 V 프로젝션 후 실행
        x = input[0] # [B, S, D]
        
        with torch.no_grad():
            # 모든 어댑터의 델타를 한 번에 계산 후 가중합
            a_w = self.adapters[f'{mode}a'].transpose(1, 2).unsqueeze(1) # [N, 1, D, R]
            b_w = self.adapters[f'{mode}b'].transpose(1, 2).unsqueeze(1) # [N, 1, R, D]
            x = x.unsqueeze(0)
            
            # 뒤의 두 차원{(S, D), (D, R)}만 계산하고 앞의 두 차원(N, B)은 배치로 본다.
            # (1, B, S, D) @ (N, 1, D, R) -> (N, B, S, R)
            # (N, B, S, R) @ (N, 1, R, D) -> (N, B, S, D)
            deltas = (x @ a_w) @ b_w
            print(f"DEBUG: mode={mode}, x.shape={x.shape}, deltas.shape={deltas.shape}")
            final_delta = (deltas * self.current_weights * self.scaling).sum(dim=0) # [B, S, D]
            print(f"DEBUG: final_delta.shape={final_delta.shape}")
            
            
        return output + final_delta





# 3. 가중치 로드 및 어댑터 리스트 생성
config = LogoConfig()

q_lora_a_key = f"base_model.model.model.layers.{config.target_layer_idx}.self_attn.q_proj.lora_A.weight"
q_lora_b_key = f"base_model.model.model.layers.{config.target_layer_idx}.self_attn.q_proj.lora_B.weight"
v_lora_a_key = f"base_model.model.model.layers.{config.target_layer_idx}.self_attn.v_proj.lora_A.weight"
v_lora_b_key = f"base_model.model.model.layers.{config.target_layer_idx}.self_attn.v_proj.lora_B.weight"

tokenizer = AutoTokenizer.from_pretrained(config.model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    config.model_id, torch_dtype=torch.float16, device_map="auto"
)

loaded_adapters = []
for path in config.adapter_paths:
    w = load_file(path)
    # 필요한 것만 추려서 GPU로 이동
    loaded_adapters.append({
        'qa': w[q_lora_a_key].to(device=base_model.device, dtype=base_model.dtype),
        'qb': w[q_lora_b_key].to(device=base_model.device, dtype=base_model.dtype),
        'va': w[v_lora_a_key].to(device=base_model.device, dtype=base_model.dtype),
        'vb': w[v_lora_b_key].to(device=base_model.device, dtype=base_model.dtype)
    })


# 4. 훅 등록
controller = LoGO_controller(adapters_list=loaded_adapters, top_k = config.top_k,scaling=64/32, device = base_model.device, dtype=base_model.dtype)

attn_module = base_model.model.layers[config.target_layer_idx].self_attn
q_proj = attn_module.q_proj
v_proj = attn_module.v_proj

attn_module.register_forward_pre_hook(controller.controller_pre_hook, with_kwargs=True)
q_proj.register_forward_hook(partial(controller.apply_adapter_hook, mode='q'))
v_proj.register_forward_hook(partial(controller.apply_adapter_hook, mode='v'))



from lm_eval.models.huggingface import HFLM
lm_obj = HFLM(pretrained=base_model, tokenizer=tokenizer) # 이미 훅이 걸린 모델을 래핑

for i in [10,20]:
    config.top_k=i
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        # model="hf",
        # model_args={
        #     "pretrained": "meta-llama/Llama-3.1-8B-Instruct",
        #     "peft": "./lora_adapters_fine_tuned/6_1/cos_e_v1.11_aligned_with_common_sense.json", # Path to your LoRA adapter
        #     "device": "cuda:0",
        # },
        tasks= config.task_name,           # Evaluation tasks (list format)
        num_fewshot=0,                # Number of few-shot examples (0-5 is standard)
        batch_size=4,                 # Adjust based on your GPU VRAM
        # limit=10                    # Uncomment this line for a quick "sanity check" (only 10 samples)
    )

    # 2. Output Results
    # Print the results in a formatted table as seen in the CLI version
    table_results = make_table(results)
    task_metrics = results["results"].get(task_name, {})
    score = task_metrics.get("acc,none") or task_metrics.get("acc") or "N/A"

    with open(config.output_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Task: {config.task_name} | Top_K: {config.top_k} | Batch Size: 1\n")
        f.write(f"Final Score (Acc): {score}\n")
        f.write(f"{'-'*60}\n")
        f.write(table_results)
        f.write(f"\n{'='*60}\n")

    print(f"✅ {config.task_name} Score: {score} - Saved to {config.output_file}")

    # Optional: Clean up GPU memory after evaluation
    gc.collect()
    torch.cuda.empty_cache()
