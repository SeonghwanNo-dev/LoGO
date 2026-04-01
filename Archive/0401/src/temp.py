import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# 1. 모델 및 토크나이저 로드 (메모리 효율을 위해 4bit/8bit 양자화 권장)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# 2. 테스트용 입력 문장
prompt = "인공지능의 미래에 대해 한 문장으로 말해줘."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 3. 분석을 위한 함수 정의
def check_sequence_length(input_ids):
    seq_len = input_ids.shape[1]
    if seq_len > 1:
        print(f"[Prefill Phase] Prompt 처리 중... Sequence Length: {seq_len}")
    else:
        print(f"[Decoding Phase] 토큰 생성 중... Sequence Length: {seq_len}")

# 4. 수동 루프를 통한 생성 과정 시뮬레이션
print("--- 생성 시작 ---")
generated_ids = inputs["input_ids"]
past_key_values = None

# 최대 5개의 토큰만 생성하며 확인
for i in range(5):
    # 현재 입력의 seq_len 확인
    # 실제 generate() 내부에서도 첫 실행은 prompt 전체, 이후는 마지막 토큰 1개만 처리함
    current_input = generated_ids if past_key_values is None else generated_ids[:, -1:]
    
    check_sequence_length(current_input)
    
    with torch.no_grad():
        outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
        
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    
    # 다음 스텝을 위해 데이터 업데이트
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    past_key_values = outputs.past_key_values

print("--- 생성 완료 ---")