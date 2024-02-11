from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

path = 'openbmb/MiniCPM-2B-dpo-fp32'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.float32, device_map='cuda', trust_remote_code=True)

while True:
    inp = input(">>> ")    
    responds, history = model.chat(
        tokenizer, inp, temperature=0.8, top_p=0.8, pad_token_id=2)
    print(responds)
