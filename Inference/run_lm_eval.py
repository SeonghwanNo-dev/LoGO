import lm_eval
import torch
import gc
from lm_eval.utils import make_table

# 1. Benchmark Execution Settings
# Using 'simple_evaluate' to run the benchmark programmatically
results = lm_eval.simple_evaluate(
    model="hf",
    model_args={
        "pretrained": "meta-llama/Llama-3.1-8B-Instruct",
        "peft": "./lora_adapters_fine_tuned/backup_6_3_11", # Path to your LoRA adapter
        "device": "cuda:0",
        "use_chat_template": True,         # Enable chat template (recommended for Instruct models)
        "fewshot_as_multiturn": True,      # Treat few-shot examples as a multi-turn conversation
    },
    tasks=["arc_easy"],           # Evaluation tasks (list format)
    num_fewshot=5,                # Number of few-shot examples (0-5 is standard)
    batch_size=4,                 # Adjust based on your GPU VRAM
    # limit=10                    # Uncomment this line for a quick "sanity check" (only 10 samples)
)

# 2. Output Results
# Print the results in a formatted table as seen in the CLI version
print(make_table(results))


# 3. Extract Specific Metrics
# Example: Extracting the accuracy score for a specific task
task_name = "arc_easy"
if task_name in results["results"]:
    # Accessing the 'acc,none' metric (standard accuracy)
    score = results["results"][task_name].get("acc,none") or results["results"][task_name].get("acc")
    print(f"✨ {task_name.upper()} Final Score: {score:.4f}")

# Optional: Clean up GPU memory after evaluation
gc.collect()
torch.cuda.empty_cache()