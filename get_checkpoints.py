from transformers import AutoTokenizer, AutoModelForCausalLM

name = "KoboldAI/fairseq-dense-125M"
save_name = "fairseq/125M"

tokenizer = AutoTokenizer.from_pretrained(name)
# model = AutoModelForCausalLM.from_pretrained(name)

tokenizer.save_pretrained(f"checkpoints/{save_name}/")
# model.save_pretrained(f"checkpoints/{save_name}/")