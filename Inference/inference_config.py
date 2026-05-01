class LogoConfig(object):
  def __init__(self):
    self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    self.target_layer_idx = 31
    self.task_name = "arc_easy"
    self.top_k = 2
    self.adapter_paths = ["./Train/lora_adapters_fine_tuned/6_1/adversarial_qa_dbert_answer_the_following_q.json/adapter_model.safetensors",
                          "./Train/lora_adapters_fine_tuned/6_1/adversarial_qa_dbert_generate_question.json/adapter_model.safetensors",
                         ]
                         