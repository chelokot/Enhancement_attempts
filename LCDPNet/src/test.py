# -*- coding: utf-8 -*-

import numpy as np
import torch
import hydra

from LCDPNet.src.globalenv import *
from LCDPNet.src.utils.util import parse_config

OURS_MODE = "ours"
MSEC_MODE = "msec"

model = None

@hydra.main(config_path='config', config_name="config")
def get_model_ours(opt):
    global model
    opt = parse_config(opt, TEST)
    from LCDPNet.src.model.lcdpnet import LitModel as ModelClass
    ckpt = "LCDPNet/trained_on_ours.ckpt"
    model = ModelClass.load_from_checkpoint(ckpt, opt = opt)


@hydra.main(config_path='config', config_name="config")
def get_model_msec(opt):
    global model
    opt = parse_config(opt, TEST)
    from LCDPNet.src.model.lcdpnet import LitModel as ModelClass
    ckpt = "LCDPNet/trained_on_MSEC.ckpt"
    model = ModelClass.load_from_checkpoint(ckpt, opt = opt)

def main(img: np.ndarray, mode: str) -> np.ndarray:
    global model
    if mode == OURS_MODE:
        get_model_ours()
    elif mode == MSEC_MODE:
        get_model_msec()
    else:
        raise ValueError("Unknown mode")
    img_tensor = torch.tensor(img, dtype = torch.float32).unsqueeze(0).transpose(1, 3).transpose(2, 3) / 255
    pred = model(img_tensor).clip(0, 1) * 255
    return pred.transpose(2, 3).transpose(1, 3).squeeze(0).detach().cpu().numpy()
