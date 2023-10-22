import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from schedulers import WarmupCosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup

model = nn.Linear(10, 10)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# scheduler = WarmupCosineAnnealingLR(optimizer, T_max=100, warmup_steps=0, eta_min=1e-5)
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

for i in range(100):
    scheduler.step()
    print(scheduler.get_last_lr()[0])