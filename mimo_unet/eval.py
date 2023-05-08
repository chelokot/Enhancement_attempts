import torch
import numpy as np
import cv2

def _eval(model, input_img) -> np.ndarray:
    state_dict = torch.load("mimo_unet/MIMO-UNet.pkl")
    model.load_state_dict(state_dict['model'])
    torch.save(model.half().state_dict(), "mimo_unet/MIMO-UNet_half.pkl")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        input_shape = (input_img.shape[1], input_img.shape[0])

        input_img = cv2.resize(input_img, ((input_img.shape[1]//4)*4, (input_img.shape[0]//4)*4))
        input_img = torch.tensor(input_img, device = device, dtype=torch.float16) / 255
        input_img = input_img.unsqueeze(0).transpose(1, 3).transpose(2, 3)
        pred = model(input_img)[2]
        pred_clip = (torch.clamp(pred, 0, 1) * 255)
        pred_numpy = pred_clip.transpose(2, 3).transpose(1, 3).squeeze(0).cpu().numpy()
        pred_numpy = pred_numpy.astype('uint8')
        pred_numpy = cv2.resize(pred_numpy, input_shape)

    return pred_numpy
