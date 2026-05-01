import glob
import os

class LogoConfig(object):
  def __init__(self):
    self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    self.target_layer_idx = 31
    self.task_name = ["leaderboard_bbh", "arc_challenge", "hellaswag", "arc_easy", "anli_r1", "piqa"]
    self.top_k = 10 #20
    self.adapter_paths = glob.glob(f"./Train/lora_adapters_fine_tuned/6_1/*/adapter_model.safetensors")[:102]
    self.output_file = "logo_result.txt"

    # Result Check
    print(f"Total {len(self.adapter_paths)} adapters")
                         
if __name__ == "__main__":
  config = LogoConfig()
