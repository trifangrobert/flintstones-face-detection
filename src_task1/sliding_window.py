import pickle
import cv2 as cv
import time

def load_config():
    with open("proposed_aspect_ratios.pkl", "rb") as f:
        aspect_ratios = pickle.load(f)

    with open("proposed_widths.pkl", "rb") as f:
        widths = pickle.load(f)

    return aspect_ratios, widths

def get_patch_sizes(aspect_ratios, widths):
    patch_sizes = []
    for aspect_ratio in aspect_ratios:
        for width in widths:
            height = width / aspect_ratio
            height = round(height)
            width = round(width)
            patch_sizes.append((height, width))
    
    patch_sizes = list(set(patch_sizes))
    return patch_sizes

def sliding_window(image, patch_size, step_size=1):
    # image should be in the format (channels, height, width)
    # print("From sliding window")
    # print("Image shape: ", image.shape)
    # print("Patch size: ", patch_size)
    # print(f"height {len(range(0, image.shape[1] - patch_size[0], step_size))}")
    # print(f"width {len(range(0, image.shape[2] - patch_size[1], step_size))}")
    for x in range(0, image.shape[1] - patch_size[0], step_size):
        for y in range(0, image.shape[2] - patch_size[1], step_size):
            yield (x, y, image[:, x:x + patch_size[0], y:y + patch_size[1]])


if __name__ == "__main__":
    aspect_ratios, widths = load_config()
    patch_sizes = get_patch_sizes(aspect_ratios, widths)

    image = cv.imread("../train_positive/barney/0003.jpg")

    start_time = time.time()
    for patch_size in patch_sizes:
        for (x, y, window) in sliding_window(image, patch_size, step_size=10):
            # window = window.numpy()
            pass
    
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)


    