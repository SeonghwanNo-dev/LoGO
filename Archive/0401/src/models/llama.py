import transformers
import torch

model_id = "meta-llama/Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"dtype": torch.bfloat16}, device_map="auto"
)

pipeline("Hey how are you doing today?")

outputs = pipeline("Hello, how are you?")
print(outputs[0]['generated_text'])

'''
- huggingface-cli scan-cache: 어떤 모델이 ~/.cache에 저장되어 있는지 나옴
다 사용한 뒤에 안 사용하는 모델들은 지운 뒤 서버 반납하기.

'''