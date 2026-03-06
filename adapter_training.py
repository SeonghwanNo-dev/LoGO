import os
import sys
from typing import List

import torch
import transformers
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from accelerate import Accelerator
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import pandas as pd
import wandb
import gc

def compute_loss_func(outputs, labels, num_items_in_batch=None):
    logits = outputs.logits
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, labels)
    return loss


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    model_type: str = "LLaMA",
    data_path: str = "dataset2/local_flan_v2/name.csv",
    output_dir: str = "./lora_adapters_fine_tuned/data_name",
    adapter_name: str = "lora",
    # training hyperparams
    batch_size: int = 32,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 2e-4,
    cutoff_len: int = 1024,
    val_set_size: int = 100,
    use_gradient_checkpointing: bool = False,
    eval_step: int = 50,
    save_step: int = 50,
    # lora hyperparams
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ['q_proj', 'k_proj', 'v_proj', 'down_proj', 'up_proj'],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "LoGo Adapters",
    wandb_run_name: str = "dataset_name",
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "false",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"model_type: {model_type}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert base_model, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    accelerator = Accelerator()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
        
    # 
    if use_wandb:
        os.environ["WANDB_MODE"] = "online"
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "model": base_model,
            },
            reinit=True
        )
    # 1. Model & Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map = "auto",
        dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    model.config.use_cache = False
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    
    # 2. Data Processing <- 고치는 중
    full_dataset = load_from_disk(data_path)
    dataset = full_dataset["train"]
    
    # Split the DataFrame (0.8 / 0.1 / 0.1)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_data = train_test_split["train"]
    test_eval_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)
    eval_data = test_eval_split["train"]
    test_data = test_eval_split["test"]

    # Generate prompts for training and evaluation data
    def apply_prompt(example):
        return {"text": f"Instruction: {example['inputs']}\nResponse: {example['targets']}"}
    train_data = train_data.map(apply_prompt)
    eval_data = eval_data.map(apply_prompt)
    test_data = test_data.map(apply_prompt)

    # 3. LoRA, Trainer Config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    training_arguments = SFTConfig( # object name changed
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        eval_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_set_size > 0 else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_steps=eval_step,
        save_steps=save_step,
        output_dir=output_dir,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
        report_to="wandb",
        run_name=wandb_run_name,
        max_length=cutoff_len, # arg name changed
        dataset_text_field="text",
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer, # arg name changed
        args=training_arguments,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
    )
    # trainer.compute_loss_func = compute_loss_func
    # trainer.accelerator.print(f"{trainer.model}")

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # 4. Train
    trainer.train()
    trainer.save_model(output_dir)
    if use_wandb:
        print(f"Finishing W&B run for {wandb_run_name}...")
        wandb.finish()

    # # 5. Test Logic
    # if True:    # temp
    #     model.config.use_cache = True
    #     model.eval()

    #     results = []
            
    #     with torch.no_grad():
    #         for step, batch in tqdm(enumerate(test_loader)):
    #             input_ids = batch['input_ids'].to(model.device)
    #             attention_mask = batch['attention_mask'].to(model.device)

    #             outputs = model.generate(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 max_new_tokens=128, 
    #                 do_sample=True,
    #                 temperature=0.7,
    #                 top_p=0.9,
    #                 eos_token_id=tokenizer.eos_token_id,
    #                 pad_token_id=tokenizer.pad_token_id
    #             )
                
    #             decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
    #             print(f"\n--- Test Case {step} ---")
    #             print(f"Generated: {decoded_output}")
    #             results.append(decoded_output)
                
    #     print("Accuracy:")
    #     for key in accuracy:
    #         print(f"{key}: {accuracy[key][0] / accuracy[key][1]}")
        
        
if __name__ == "__main__":

    # # Dataset Loading and Validation
    # dataset = load_from_disk("./dataset_2/local_flan_v2/adversarial_qa_dbert_answer_the_following_q.json")
    # print(dataset)
    # print("-" * 50)
    # for i in range(3):
    #     print(f"Sample {i} Input: {dataset['train'][i]['inputs']}")
    #     print(f"Sample {i} Target: {dataset['train'][i]['targets']}")
    #     print("-" * 50)
        

    base_path = "./dataset_2/local_flan_v2/"
    task_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    done_path = "./lora_adapters_fine_tuned/5/"
    done_task_folders = [f for f in os.listdir(done_path) if os.path.isdir(os.path.join(done_path, f))]
    print(done_task_folders)

    for task in task_folders:
        
        full_path = os.path.join(base_path, task)
        
        if task in done_task_folders:
            print(f"{task} is done")
        else:
            print(f"\n" + "="*50)
            print(f"Current Task: {task}")
            print("="*50 + "\n")
            train(
                base_model = "meta-llama/Llama-3.1-8B-Instruct",
                model_type = "LLaMA",
                data_path = full_path,
                output_dir = f"./lora_adapters_fine_tuned/5/{task}",
                adapter_name = "lora",
                wandb_project = "LoGo Adapters_5",
                wandb_run_name = task,
                wandb_watch = "false",  # options: false | gradients | all
                wandb_log_model = "false",  # options: false | true
                resume_from_checkpoint = None,  # either training checkpoint or final adapter
                # num_epochs = 1,
                eval_step = 50,
                save_step = 50,
            )

        # GPU Memory Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n Finished {task}. GPU Memory Cleared.\n")
    
    '''
    DatasetDict({
        train: Dataset({
            features: ['inputs', 'targets', 'task_source', 'task_name', 'template_type'],
            num_rows: 1502
        })
    })
    Sample 0 Input: Given the following passage  "A generation later, the Irish Anglican bishop, George Berkeley (1685–1753), determined that Locke's view immediately opened a door that would lead to eventual atheism. In response to Locke, he put forth in his Treatise Concerning the Principles of Human Knowledge (1710) an important challenge to empiricism in which things only exist either as a result of their being perceived, or by virtue of the fact that they are an entity doing the perceiving. (For Berkeley, God fills in for humans by doing the perceiving whenever humans are not around to do it.) In his text Alciphron, Berkeley maintained that any order humans may see in nature is the language or handwriting of God. Berkeley's approach to empiricism would later come to be called subjective idealism.",  answer the following question. Note that the answer is present within the text.  Question: what concept is mentioned last?
    Sample 0 Target: subjective idealism
    ''' 