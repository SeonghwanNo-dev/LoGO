class AdapterTrainerConfig_1(object):
  def __init__(self):
    self.base_path = "./Data/dataset_2/local_flan_v2_1"
    self.base_model = "meta-llama/Llama-3.1-8B-Instruct",
    self.model_type = "LLaMA",
    self.data_path = full_path,
    self.output_dir = "./lora_adapters_fine_tuned/6_2",
    self.adapter_name = "lora",
    self.wandb_project = "LoGo Adapters_5",
    self.wandb_run_name = task,
    self.wandb_watch = "false",  # options: false | gradients | all
    self.wandb_log_model = "false",  # options: false | true
    self.resume_from_checkpoint = None,  # either training checkpoint or final adapter
    self.num_epochs = 10,
    self.eval_step = 50,
    self.save_step = 50,

class AdapterTrainerConfig_2(object):
  def __init__(self):
    self.base_path = "./Data/dataset_2/local_flan_v2_2"
    self.base_model = "meta-llama/Llama-3.1-8B-Instruct",
    self.model_type = "LLaMA",
    self.data_path = full_path,
    self.output_dir = "./lora_adapters_fine_tuned/6_2",
    self.adapter_name = "lora",
    self.wandb_project = "LoGo Adapters_5",
    self.wandb_run_name = task,
    self.wandb_watch = "false",  # options: false | gradients | all
    self.wandb_log_model = "false",  # options: false | true
    self.resume_from_checkpoint = None,  # either training checkpoint or final adapter
    self.num_epochs = 10,
    self.eval_step = 50,
    self.save_step = 50,


class GoogleUploadConfig(object):
    def __init__(self):
        self.token_file= './token.json'                        # Name of the authentication token file
        self.cred_file = './credentials.json'                  # Name of the credential file from Google Console
        self.target_file = './backup_6_3_10.zip'             # File to be uploaded
        self.google_drive_folder_id = '1MnADBZTRqtblNiFxQWJjS3AMJMID1qHw' # Google Drive Folder ID

class ObserverConfig_1(object):
    def __init__(self):
        self.google_drive_folder_id = '1MnADBZTRqtblNiFxQWJjS3AMJMID1qHw'
        self.watch_dir = "./6_1/"
        self.batch_size = 5
        self.stable_time_threshold = 300 

class ObserverConfig_2(object):
    def __init__(self):
        self.google_drive_folder_id = '1MnADBZTRqtblNiFxQWJjS3AMJMID1qHw'
        self.watch_dir = "./6_2/"
        self.batch_size = 5
        self.stable_time_threshold = 300 