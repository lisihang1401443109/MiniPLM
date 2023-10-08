from transformers import AutoTokenizer, AutoModelForCausalLM

name = "gpt2-xl"
save_name = "gpt2-xlarge"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

tokenizer.save_pretrained(f"checkpoints/{name}/")
model.save_pretrained(f"checkpoints/{save_name}/")