import torch
from torch.backends import cudnn
from mimo_unet.models.MIMOUNet import build_net
from mimo_unet.eval import _eval
import numpy as np

# python3.9 mimo_unet/main.py --model_name "MIMO-UNetPlus" --mode "test" --data_dir "test_images" --test_model "mimo_unet/MIMO-UNetPlus.pkl" --save_image True
# python3.9 mimo_unet/main.py --model_name "MIMO-UNet" --mode "test" --data_dir "test_images" --test_model "mimo_unet/MIMO-UNet.pkl" --save_image True
# in 384x384!
# cuda out of memory!

def main(input_image: np.ndarray) -> np.ndarray:
    cudnn.benchmark = True

    model = build_net("MIMO-UNet")
    if torch.cuda.is_available():
        model.cuda()

    return _eval(model, input_image)