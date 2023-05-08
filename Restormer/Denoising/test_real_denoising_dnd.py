## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np

import torch

from Restormer.basicsr.models.archs.restormer_arch import Restormer

yaml_file = 'Restormer/Denoising/Options/RealDenoising_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')

def main(img: np.ndarray) -> np.ndarray:
    model_restoration = Restormer(**x['network_g'])
    checkpoint = torch.load('Restormer/Denoising/pretrained_models/real_denoising.pth')
    model_restoration.load_state_dict(checkpoint['params'])
    model_restoration.cuda()
    model_restoration.eval()
    torch.save(model_restoration.half().state_dict(), 'Restormer/Denoising/pretrained_models/real_denoising_half.pth')

    noisy_patch = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).cuda().half()
    restored_patch = model_restoration(noisy_patch)
    restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    return restored_patch
