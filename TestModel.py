import cv2
import numpy as np
import os

import time

PROCESSED_FOLDER = 'processed'

MOTION_DEBLUR = 'motion_deblur'
ENLIGHTNING = 'enlightning'
DENOISE = 'denoise'
DEHAZING = 'dehazing'

tasks = {
    MOTION_DEBLUR: [
        'test_images/motionblur1.jpeg', 'test_images/motionblur2.jpeg', 'test_images/motionblur3.png',
        'test_images/motionblur4.png',  'test_images/motionblur5.png', 'test_images/motionblur6.png',
        'test_images/motionblur7.png',
    ],
    ENLIGHTNING: [
        'test_images/dark1.jpg', 'test_images/dark2.png', 'test_images/dark3.png', 'test_images/dark4.png',
        'test_images/dark5.png',
    ],
    DENOISE: [
        'test_images/шум1.png', 'test_images/шум2.png', 'test_images/шум3.png',
        'test_images/шум4.png', 'test_images/шум5.png', 'test_images/шум6.png',
        'test_images/шум7.png', 'test_images/шум8.png', 'test_images/шум9.png', 'test_images/шум10.png',
    ],
    DEHAZING: [
        'test_images/бледно1.jpg', 'test_images/бледно2.jpg', 'test_images/бледно3.png',
        'test_images/бледно4.jpg', 'test_images/бледно5.jpg',
    ]
}

class ProcessImage:
    def process(self, image: np.ndarray) -> np.ndarray:
        pass

    def tasks(self) -> list[str]:
        pass

    def is_test(self, image_path: str) -> bool:
        return image_path in sum([tasks[task] for task in self.tasks()], [])

    def model_size(self) -> str:
        pass


class TestModel:
    def __init__(self, process: ProcessImage, model_name: str, test_size: int):
        self._process = process
        self._model_name = model_name
        self.test_size = test_size

    def get_model_name(self) -> str:
        return self._model_name

    def process_image(self, image: np.ndarray) -> np.ndarray:
        return self._process.process(image)

    def is_test(self, image_path: str) -> bool:
        return self._process.is_test(image_path)

    def tasks(self) -> list[str]:
        return self._process.tasks()

    def model_size(self) -> int:
        return self._process.model_size()


def compare_to_original(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    w, h = original.shape[0], original.shape[1]
    white_img = 255 * np.ones((int(1.25 * w), int(2.5 * h), 3), dtype=np.uint8)

    # put original image on the left
    white_img[w // 8:w + w // 8, h // 8:h + h // 8, :] = original
    # put processed image on the right
    white_img[w // 8:w + w // 8, h + h // 4:2 * h + h // 4, :] = processed

    return white_img


def test_model(model: TestModel, path_to_image: str, image_size: int) -> float:
    start_time = time.time()
    original_image = cv2.imread(path_to_image)
    original_image_bigger_dim = max(original_image.shape[0], original_image.shape[1])
    if original_image_bigger_dim > image_size:
        original_image = cv2.resize(original_image, (
            original_image.shape[1] * image_size // original_image_bigger_dim, # cv2 wtf???
            original_image.shape[0] * image_size // original_image_bigger_dim
        ))
    processed_image = model.process_image(original_image)
    compared_image = compare_to_original(original_image, processed_image)
    try:
        os.mkdir(PROCESSED_FOLDER)
    except:
        pass
    try:
        os.mkdir(os.path.join(PROCESSED_FOLDER, model.get_model_name()))
    except:
        pass

    cv2.imwrite(os.path.join(PROCESSED_FOLDER, model.get_model_name(), os.path.basename(path_to_image)), compared_image)
    return time.time() - start_time

