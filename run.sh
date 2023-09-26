python generate.py  --method speculative \
                    --prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
                    --max_new_tokens 64 \
                    --target_model checkpoints/opt-13B \
                    --draft_model checkpoints/opt-1.3B \
                    --temperature 0.5