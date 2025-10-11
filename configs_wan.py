import torch


model_id = '/home/m2v_intern/yangzhen/checkpoint/Wan2.1-T2V-1.3B-Diffusers'

num_inference_steps = 50

# WAN-1.3B
res_min, res_max = (480, 832), (960, 1664)
N = 2
cfg_min, cfg_max, M_cfg = 5, 10, 1
T_min, T_max, M_T = 30, num_inference_steps, 1

seed = 2025
generator = torch.Generator("cuda").manual_seed(seed)
It_base_path = f"results_wan"

prompts = [""]


