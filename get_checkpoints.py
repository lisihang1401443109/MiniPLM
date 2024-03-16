from transformers import AutoTokenizer, AutoModelForCausalLM

name = "KoboldAI/fairseq-dense-125M"
save_name = "fairseq/125M"

tokenizer = AutoTokenizer.from_pretrained(name, proxies={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})
model = AutoModelForCausalLM.from_pretrained(name, proxies={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})

tokenizer.save_pretrained(f"checkpoints/{save_name}/")
model.save_pretrained(f"checkpoints/{save_name}/", safe_serialization=False)