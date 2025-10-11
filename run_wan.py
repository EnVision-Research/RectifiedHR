import torch, os, cv2
import numpy as np
from pipelines.scheduling_flow_match_euler_discrete import FMEDScheduler
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from pipelines.pipeline_wan import WanPipeline
from diffusers.utils.torch_utils import randn_tensor
from utils.preprocess import latent2video, video2latent
from utils.main_tools import quantic_HW, quantic_cfg, quantic_step
from configs_wan import (
    model_id, prompts, generator, It_base_path, num_inference_steps,
    N, cfg_min, cfg_max, M_cfg, T_min, T_max, M_T, res_min, res_max,
)



def main():
    # 1. init
    num_frames = 81
    refresh_step_list = quantic_step(T_min, T_max, N, M_T)
    cfg_list = quantic_cfg(cfg_min, cfg_max, N, M_cfg)
    resolution_list = quantic_HW(res_min, res_max, N)
    os.makedirs(It_base_path, exist_ok=True)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    # pipe.vae.enable_tiling()
    # pipe.enable_model_cpu_offload()
    pipe.scheduler = FMEDScheduler.from_config(pipe.scheduler.config)
    MEMORY = {}

    # 2. run
    for idx, prompt in enumerate(prompts):
        MEMORY.update({
            'predict_x0_list': [],
        })
        start_latent = randn_tensor((1, 16, (num_frames - 1) // pipe.vae_scale_factor_temporal + 1, res_min[0] // pipe.vae_scale_factor_spatial, res_min[1] // pipe.vae_scale_factor_spatial), dtype=pipe.dtype, device=pipe.device, generator=generator)
        for i in range(len(resolution_list)):
            
            if i == 0 and resolution_list[0] == resolution_list[1]:
                strength, denoising_end = 1, 1
                hr_output = pipe(
                        prompt=prompt,
                        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        generator=generator,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=cfg_list[i],
                        latents=start_latent,

                        MEMORY=MEMORY,
                        strength=1,
                        denoising_end=1,
                    )
            else:
                if i == 0:
                    predict_x0_latent_noisey = start_latent
                else:
                    if resolution_list[i][1] == resolution_list[i - 1][1]:
                        predict_x0_latent = MEMORY['predict_x0_list'][-1]
                    else:
                        video_frames = [(frame * 255).astype(np.uint8) for frame in latent2video(MEMORY['predict_x0_list'][-1], pipe)][0]
                        resize_video_frames = np.zeros((video_frames.shape[0], resolution_list[i][0], resolution_list[i][1], video_frames.shape[-1]), dtype=video_frames.dtype)
                        for j in range(video_frames.shape[0]):
                            resize_video_frames[j] = cv2.resize(video_frames[j], (resolution_list[i][1], resolution_list[i][0]), interpolation=cv2.INTER_LINEAR)
                        torch.cuda.empty_cache()
                        predict_x0_latent = video2latent(resize_video_frames, pipe)
                    noise_ = randn_tensor(predict_x0_latent.shape, dtype=predict_x0_latent.dtype, device=predict_x0_latent.device, generator=generator)
                    predict_x0_latent_noisey = pipe.scheduler.scale_noise(predict_x0_latent, pipe.scheduler.timesteps[len(MEMORY['predict_x0_list']) - num_inference_steps].unsqueeze(0), noise_)
                hr_output = pipe(
                        prompt=prompt,
                        negative_prompt="",
                        generator=generator,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=cfg_list[i],
                        latents=predict_x0_latent_noisey,

                        MEMORY=MEMORY,
                        strength=(num_inference_steps - len(MEMORY['predict_x0_list'])) / num_inference_steps,
                        denoising_end=refresh_step_list[i + 1] / num_inference_steps,
                    )
                torch.cuda.empty_cache()
        export_to_video(hr_output.frames[0], os.path.join(It_base_path, f'{idx}_{cfg_max}_{res_min}_{res_max}.mp4'), fps=15)



if __name__ == '__main__':
    main()



