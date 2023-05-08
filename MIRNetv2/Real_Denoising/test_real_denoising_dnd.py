## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/


import numpy as np
import argparse

import torch

from MIRNetv2.basicsr.models.archs.mirnet_v2_arch import MIRNet_v2

yaml_file = 'MIRNetv2/Real_Denoising/Options/RealDenoising_MIRNet_v2.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')

def main(img: np.ndarray) -> np.ndarray:
    model_restoration = MIRNet_v2(**x['network_g'])
    checkpoint = torch.load('MIRNetv2/Real_Denoising/pretrained_models/real_denoising.pth')
    model_restoration.load_state_dict(checkpoint['params'])
    model_restoration.cuda()
    model_restoration.eval()
    torch.save(model_restoration.half().state_dict(), 'MIRNetv2/Real_Denoising/pretrained_models/real_denoising_half.pth')

    noisy_patch = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).cuda().half()
    restored_patch = model_restoration(noisy_patch)
    restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    return restored_patch
