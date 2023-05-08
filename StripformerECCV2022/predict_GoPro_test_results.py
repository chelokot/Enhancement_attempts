from __future__ import print_function
import numpy as np
import torch
import cv2
import yaml
import os
from torch.autograd import Variable
from StripformerECCV2022.models.networks import get_generator
import torchvision

GO_PRO_MODE = 'GoPro'
REAL_BLUR_R_MODE = 'RealBlur_R'
REAL_BLUR_J_MODE = 'RealBlur_J'

def main(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == GO_PRO_MODE:
        with open('StripformerECCV2022/config/config_Stripformer_gopro.yaml') as cfg:
            config = yaml.safe_load(cfg)
        model = get_generator(config['model'])
        model.load_state_dict(torch.load('StripformerECCV2022/Stripformer_gopro.pth'))
        model = model.cuda()
        torch.save(model.half().state_dict(), "StripformerECCV2022/Stripformer_gopro_half.pth")
    elif mode == REAL_BLUR_R_MODE:
        with open('StripformerECCV2022/config/config_Stripformer_gopro.yaml') as cfg:
            config = yaml.safe_load(cfg)
        model = get_generator(config['model'])
        model.load_state_dict(torch.load('StripformerECCV2022/Stripformer_realblur_R.pth'))
        model = model.cuda()
        torch.save(model.half().state_dict(), "StripformerECCV2022/Stripformer_realblur_R_half.pth")
    elif mode == REAL_BLUR_J_MODE:
        with open('StripformerECCV2022/config/config_Stripformer_gopro.yaml') as cfg:
            config = yaml.safe_load(cfg)
        model = get_generator(config['model'])
        model.load_state_dict(torch.load('StripformerECCV2022/Stripformer_realblur_J.pth'))
        model = model.cuda()
        torch.save(model.half().state_dict(), "StripformerECCV2022/Stripformer_realblur_J_half.pth")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img = cv2.resize(img, ((img.shape[1]//4) * 4, (img.shape[0]//4) * 4))
    img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float16')) - 0.5
    with torch.no_grad():
        img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()

        result_image = model(img_tensor)
        result_image = result_image + 0.5
        result_image = result_image.cpu().numpy()
        result_image = np.transpose(result_image[0], (1, 2, 0))
        result_image = (result_image * 255).clip(0, 255)
        result_image = result_image.astype('uint8')
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        result_image = cv2.resize(result_image, (original_size[1], original_size[0]))
        return result_image
