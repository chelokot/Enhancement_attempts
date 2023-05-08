# python3.9 -m pip install git+https://github.com/arsenyinfo/EnlightenGAN-inference
from TestModel import ProcessImage, TestModel, ENLIGHTNING, MOTION_DEBLUR, DENOISE, DEHAZING
import numpy as np
import os

def get_pretty_file_size(path_to_file: str) -> str:
    size = os.path.getsize(path_to_file)
    if size < 1024:
        return f'{size} B'
    elif size < 1024 * 1024:
        return f'{size / 1024:.1f} KB'
    elif size < 1024 * 1024 * 1024:
        return f'{size / 1024 / 1024:.1f} MB'
    else:
        return f'{size / 1024 / 1024 / 1024:.1f} GB'

from enlighten_inference import EnlightenOnnxModel
class EnlightenOnnxModelProcess(ProcessImage):
    def process(self, image: np.ndarray) -> np.ndarray:
        model = EnlightenOnnxModel()
        return model.predict(image)

    def tasks(self) -> list[str]:
        return [ENLIGHTNING]

    def model_size(self) -> str:
        return "33 mb" #https://github.com/arsenyinfo/EnlightenGAN-inference/blob/main/enlighten_inference/enlighten.onnx



from mimo_unet.main import main as mimo_unet_main
class MimoUnetProcess(ProcessImage):
    def process(self, image: np.ndarray) -> np.ndarray:
        return mimo_unet_main(image)

    def tasks(self) -> list[str]:
        return [MOTION_DEBLUR]

    def model_size(self) -> str:
        return get_pretty_file_size('mimo_unet/MIMO-UNet_half.pkl')


from StripformerECCV2022.predict_GoPro_test_results import main, GO_PRO_MODE, REAL_BLUR_R_MODE, REAL_BLUR_J_MODE
class StripformerProcessGopro(ProcessImage):
    def process(self, image: np.ndarray) -> np.ndarray:
        return main(image, GO_PRO_MODE)

    def tasks(self) -> list[str]:
        return [MOTION_DEBLUR]

    def model_size(self) -> str:
        return get_pretty_file_size('StripformerECCV2022/Stripformer_gopro_half.pth')


class StripformerProcessRealBlurR(ProcessImage):
    def process(self, image: np.ndarray) -> np.ndarray:
        return main(image, REAL_BLUR_R_MODE)

    def tasks(self) -> list[str]:
        return [MOTION_DEBLUR]

    def model_size(self) -> str:
        return get_pretty_file_size('StripformerECCV2022/Stripformer_realblur_R_half.pth')


class StripformerProcessRealBlurJ(ProcessImage):
    def process(self, image: np.ndarray) -> np.ndarray:
        return main(image, REAL_BLUR_J_MODE)

    def tasks(self) -> list[str]:
        return [MOTION_DEBLUR]

    def model_size(self) -> str:
        return get_pretty_file_size('StripformerECCV2022/Stripformer_realblur_J_half.pth')


from LCDPNet.src.test import main as lcdpnet_main, OURS_MODE, MSEC_MODE
class LcdpnetProcessOurs(ProcessImage):
    def process(self, image: np.ndarray) -> np.ndarray:
        return lcdpnet_main(image, OURS_MODE)

    def tasks(self) -> list[str]:
        return [ENLIGHTNING]

    def model_size(self) -> str:
        return get_pretty_file_size('LCDPNet/trained_on_ours.ckpt')

class LcdpnetProcessMsec(ProcessImage):
    def process(self, image: np.ndarray) -> np.ndarray:
        return lcdpnet_main(image, MSEC_MODE)

    def tasks(self) -> list[str]:
        return [ENLIGHTNING, DEHAZING]

    def model_size(self) -> str:
        return get_pretty_file_size('LCDPNet/trained_on_MSEC.ckpt')


from MIRNetv2.Real_Denoising.test_real_denoising_dnd import main as mirnet_main
class MirnetProcess(ProcessImage):
    def process(self, image: np.ndarray) -> np.ndarray:
        return mirnet_main(image)

    def tasks(self) -> list[str]:
        return [DENOISE]

    def model_size(self) -> str:
        return 'TODO'


from Restormer.Denoising.test_real_denoising_dnd import main as restormer_main
class RestormerProcess(ProcessImage):
    def process(self, image: np.ndarray) -> np.ndarray:
        return restormer_main(image)

    def tasks(self) -> list[str]:
        return [DENOISE]

    def model_size(self) -> str:
        return 'TODO'

models = [
    TestModel(EnlightenOnnxModelProcess(),   'EnlightenOnnxModel',   test_size = 768),

    TestModel(MimoUnetProcess(),             'MimoUnet',             test_size = 768),

    TestModel(StripformerProcessGopro(),     'StripformerGopro',     test_size = 500),
    TestModel(StripformerProcessRealBlurR(), 'StripformerRealBlurR', test_size = 500),
    TestModel(StripformerProcessRealBlurJ(), 'StripformerRealBlurJ', test_size = 500),

    TestModel(LcdpnetProcessOurs(),          'LcdpnetOurs',          test_size = 1024),
    TestModel(LcdpnetProcessMsec(),          'LcdpnetMsec',          test_size = 1024),

    # TestModel(MirnetProcess(),            'MirnetDnd',            test_size = 512), #судя по их коду нужен размер 512, но у меня на нем out of memory
    # TestModel(RestormerProcess(),            'RestormerDnd',         test_size = 256), #это падает даже на 256
]