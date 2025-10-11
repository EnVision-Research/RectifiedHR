import torch


model_id = '/home/m2v_intern/yangzhen/checkpoint/stable-diffusion-xl-base-1.0'

num_inference_steps = 50

# SDXL
# For baseline
# res_min, res_max = (1024, 1024), (1024, 1024)
# N = 2
# cfg_min, cfg_max, M_cfg = 5, 5, 1
# T_min, T_max, M_T = 49, num_inference_steps, 1

# For 2048 x 2048
res_min, res_max = (1024, 1024), (2048, 2048)
N = 2
cfg_min, cfg_max, M_cfg = 5, 30, 1
T_min, T_max, M_T = 30, num_inference_steps, 1

# For 2048 x 4096 
# res_min, res_max = (1536, 768), (4096, 2048)
# M_cfg, M_T, T_max, cfg_min = 1, 1, num_inference_steps, 5
# N = 3
# cfg_max = 50
# T_min = 40

# # For 4096 x 4096
# res_min, res_max = (1024, 1024), (4096, 4096)
# N = 3
# cfg_min, cfg_max, M_cfg = 5, 50, 0.5
# T_min, T_max, M_T = 40, num_inference_steps, 0.5


seed = 2025
generator = torch.Generator("cuda").manual_seed(seed)
It_base_path = f"results"

prompts = [""]


