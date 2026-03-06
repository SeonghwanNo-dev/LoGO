from safetensors import safe_open

# Path to the safetensors file you want to inspect
file_path = "./5/wiki_qa_Jeopardy_style.json/checkpoint-40/adapter_model.safetensors"
# Path to the output text file
output_file = "saved_file_test_result.txt"

with safe_open(file_path, framework="pt", device="cpu") as f, open(output_file, "w") as out:
    out.write(f"--- [Tensors in {file_path}] ---\n")
    
    # Retrieve all tensor names (keys) from the file
    for key in f.keys():
        tensor = f.get_tensor(key)
        out.write(f"Name: {key}\n")
        out.write(f"  Shape: {list(tensor.shape)}\n")
        out.write(f"  Dtype: {tensor.dtype}\n")
        
        # Uncomment the lines below to record actual weight values (e.g., first 5 elements)
        # out.write(f"  Sample: {tensor.flatten()[:5].tolist()}\n")
        out.write("-" * 30 + "\n")

print(f"Inspection complete. Results saved to: {output_file}")