# Pretrain Perf.
Vanilla (eager): step time 0.47
sdpa: step time 0.34
xops: step time 0.31

# Notation
Global Step: number of update (starts from 1)
Step/Model/Ckpt Index: model states at specific time step (starts from 0) 

loss mask 当输入有 pad 的时候有 bug，已修复，需要合并到 main commit: efd4e9e599047c2a3aa5471d764ceac3abed0fd2

preprocess 的时候不要加 pad !!

mistral tokenizer:

(3.3 + 0.783 + 0.328 + 0.083 + 0.085 + 0.092 + 0.078)GB size -> 1.4TB tokens