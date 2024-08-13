import os 
import sys
if './' not in sys.path:
	sys.path.append('./')
from utils.share import *
import utils.config as config

import cv2
import einops
import gradio as gr
import numpy as np

import torch
from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3
from annotator.content import ContentDetector

from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler
from annotator.content import ContentDetector

apply_content = ContentDetector()


model = create_model('./configs/uni_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./ckpt/flare3.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)




prompt = ''

num_samples = 2     #how many images will be  generated
image_resolution = 512 


strength = 1.0  
global_strength = 1

high_threshold = 200
low_threshold = 100

value_threshold = 0.1
distance_threshold = 0.1
alpha = 6.2

ddim_steps = 50
scale = 9.0
seed = 22
eta = 0.0

a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

''' test path '''
test_path = '/export/lianjz/workspace/control/Uni-ControlNet/data/conditions/lq'
test_names = os.listdir(test_path)


def process(lq_image, mask_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength):
    
    seed_everything(seed)

    if lq_image is not None:
        anchor_image = lq_image
    elif mlsd_image is not None:
        anchor_image = mask_image
    else:
        anchor_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)
    H, W, C = resize_image(HWC3(anchor_image), image_resolution).shape

    with torch.no_grad():
        if lq_image is not None:
            lq_image = cv2.resize(lq_image, (W, H))
            lq_detected_map = HWC3(lq_image)
        else:
            lq_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if mask_image is not None:
            mask_image = cv2.resize(mask_image, (W, H))
            mask_detected_map = HWC3(mask_image)
        # if content_image is not None:
        #     content_emb = apply_content(content_image)
        #     content_emb.seek(0)
        # else:
        content_emb = np.zeros((768))
        
        # global_condition = np.load(content_emb, allow_pickle=True)
        global_condition = content_emb
        detected_maps_list = [lq_image, 
                               mask_image                          
                             ]
        detected_maps = np.concatenate(detected_maps_list, axis=2)

        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
        global_control = torch.from_numpy(global_condition.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)
        cond = {"local_control": [local_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 'global_control': [global_control]}
        un_cond = {"local_control": [uc_local_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], 'global_control': [uc_global_control]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, global_strength=global_strength)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]

    return [results, detected_maps_list]

for test_name in test_names[ : 3 ]:
    content_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)
    # content_image = cv2.imread('./data/conditions/content/' + test_name )
    lq_image = cv2.imread('./data/conditions/lq/' + test_name )
    
    mask_image = cv2.imread('./data/conditions/mask/' + test_name)
    
    # content_image = cv2.imread('./data/imags/' + test_name)
    print(f"========== test {test_name}==========")
    [image_gallery, cond_gallery] = process(lq_image, mask_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength)
    for image in image_gallery:
        for   i in range(0,len(image_gallery)):
            image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)
            path1 = f'./data/output/{test_name}'.replace("input", "output").replace(".png",f"_{i:02d}.png")
            path2 = './data/output/{}'.format(test_name).replace("input", "output").replace(".png",f"_{i:02d}.png")
            print(f'path {path1, path2}')
            
            cv2.imwrite(path1, image)
        

