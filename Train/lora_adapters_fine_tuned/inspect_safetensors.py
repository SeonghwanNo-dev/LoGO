import os
import sys
from safetensors import safe_open

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from train_config import InspectSafetensorsConfig


if __name__ == "__main__":
    config = InspectSafetensorsConfig()

    with safe_open(config.file_path, framework="pt", device="cpu") as f, open(config.save_file, "w") as out:
        out.write(f"--- [Tensors in {config.file_path}] ---\n")
        
        # Retrieve all tensor names (keys) from the file
        for key in f.keys():
            tensor = f.get_tensor(key)
            out.write(f"Name: {key}\n")
            out.write(f"  Shape: {list(tensor.shape)}\n")
            out.write(f"  Dtype: {tensor.dtype}\n")
            
            # Uncomment the lines below to record actual weight values (e.g., first 5 elements)
            # out.write(f"  Sample: {tensor.flatten()[:5].tolist()}\n")
            out.write("-" * 30 + "\n")

print(f"Inspection complete. Results saved to: {config.save_file}")