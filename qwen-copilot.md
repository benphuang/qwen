User's plan to self-host Qwen3.8-27B on a remote 8x RTX A4000 server and use it via VSCode Remote SSH + Copilot BYOK

Details
Thinking of setting up a local agent with Qwen3.8-27B on a remote server
Codes on that server via VSCode Remote SSH; the Copilot extension already works there, so considering BYOK (bring your own key/model) for Copilot
Server hardware: 8x RTX A4000 (16GB VRAM each); P2P/NCCL disabled so GPU traffic routes via host
Asked to be interviewed to nail down the plan
Usage: just User, one Copilot session at a time
Priority: longest possible context per request
Quantization: open, wants to benchmark and decide (BF16 vs FP8 vs 4-bit)
Context target: native 262K for now, thinking of extending later
Server access: shared, no root (conda/venv only, maybe Slurm)
Copilot usage: a mix of agent mode and chat
GPUs: basically nobody else uses them, but no guarantees
Processes can run persistently (tmux/screen or user service, no time limit)
NVIDIA driver/CUDA: 13.0
Copilot plan: personal (not org-managed)
Wants to benchmark hosting on 2, 4, and all 8 GPUs (in addition to precision variants); wants install + launch scripts drafted