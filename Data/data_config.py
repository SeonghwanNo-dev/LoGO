class Dataset_2_Config(object):
  def __init__(self):
    self.target_data_txt = "./Data/dataset_2/dataset_temp.txt"
    self.huggingface_dataset_ID = "lorahub/flanv2"
    self.save_directory = "./Data/dataset_2/local_flan_v2"


class DataAllocationConfig(object):
  def __init__(self):
    self.base_path = "./Data/dataset_2/local_flan_v2"
    self.split_num = 2
    self.target_1_path = "./Data/dataset_2/local_flan_v2_3"
    self.target_2_path = "./Data/dataset_2/local_flan_v2_4"