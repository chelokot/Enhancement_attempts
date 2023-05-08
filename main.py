import glob
from Zoo import models
from TestModel import test_model, PROCESSED_FOLDER

TEST_IMAGES_FOLDER = 'test_images'

if __name__ == '__main__':
    for model in models:
        time_strings = []
        times = []
        for img_path in glob.glob(f'{TEST_IMAGES_FOLDER}/*.*'):
            if not model.is_test(img_path):
                continue
            time = test_model(model, img_path, model.test_size)
            time_strings.append(f"{img_path:30s}: {time:.3f} sec, Size: {model.test_size}\n")
            times.append(time)
        with open(f'{PROCESSED_FOLDER}/{model.get_model_name()}/times.txt', 'w') as f:
            f.write(''.join(time_strings))
            f.write(f"Average time for {model.get_model_name()}: {sum(times) / len(times):.3f} sec")
        print(f"Average time for {model.get_model_name():20s}: {sum(times) / len(times):.3f} sec  When bigger dim is {model.test_size:4d}  Model size: {model.model_size():10s} Tasks: {', '.join(model.tasks())}")
