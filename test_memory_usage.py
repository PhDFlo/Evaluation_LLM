import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.quantization import quantize_dynamic
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Load model and tokenizer
#model_name = "mistralai/Mistral-7B-v0.1"
model_name = "google-bert/bert-base-uncased"
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#model_name = "gpt2"  # Replace with your model of choice
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Print initial memory usage
print("Initial memory usage:")
print_memory_usage()

# List of torch types for quantization
list_torch_types = [torch.float16, torch.qint8]

# Quantize model and print memory usage
for torch_type in list_torch_types:
    print(f"\nMemory usage for {torch_type}:")
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch_type
    )
    print_memory_usage()

# Test text generation
inputs = tokenizer("What is distillation for a LLM?", return_tensors="pt")

# Generate text with original model
print("\nOriginal model output:")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Generate text with qint8 quantized model
print("\nQuantized model (qint8) output:")
quantized_model_qint8 = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
outputs = quantized_model_qint8.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Generate text with float16 quantized model
print("\nQuantized model (float16) output:")
quantized_model_fp16 = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.float16)
outputs = quantized_model_fp16.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))