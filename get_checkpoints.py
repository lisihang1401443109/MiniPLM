from transformers import AutoTokenizer, AutoModelForCausalLM

name = "gpt2-xl"
save_name = "gpt2/xlarge"

# tokenizer = AutoTokenizer.from_pretrained(name, proxies={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})
# model = AutoModelForCausalLM.from_pretrained(name, proxies={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

# tokenizer.save_pretrained(f"checkpoints/{save_name}/")
model.save_pretrained(f"checkpoints/{save_name}/", safe_serialization=False)