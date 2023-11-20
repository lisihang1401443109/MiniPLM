from transformers import AutoTokenizer, AutoModelForCausalLM

name = "KoboldAI/fairseq-dense-1.3b"
save_name = "fairseq/1.3B"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

tokenizer.save_pretrained(f"checkpoints/{save_name}/")
model.save_pretrained(f"checkpoints/{save_name}/", safe_serialization=False)