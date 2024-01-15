import torch
from torch.utils.data import Dataset
import os
import cv2 as cv

from utils import sliding_window

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastPatchDataset(Dataset):
    def __init__(self, dataset_path: str, patch_shape: tuple = (64, 64), stride: int = 10):
        super(FastPatchDataset, self).__init__()
        self.dataset_path = dataset_path
        self.patch_shape = patch_shape
        self.stride = stride
        
        # calculate the number of patches per image 
        self.patches_per_height = (360 - self.patch_shape[0]) // stride
        self.patches_per_width = (480 - self.patch_shape[1]) // stride

        self.num_patches_per_image = self.patches_per_width * self.patches_per_height
        self.image_paths = [os.path.join(dataset_path, image_name) for image_name in os.listdir(dataset_path)]
        self.image_paths.sort()
        self.num_images = len(self.image_paths)

        print(f"Patches per width: {self.patches_per_width}")
        print(f"Patches per height: {self.patches_per_height}")
        print(f"Total number of patches per image: {self.num_patches_per_image}")

        self.last_image_index = -1
        self.curr_image = None
        self.curr_image_name = None

        self.curr_patches = None

    def __len__(self):
        return self.num_images * self.num_patches_per_image
    
    def __getitem__(self, index):
        expected_image_index = index // self.num_patches_per_image
        if expected_image_index != self.last_image_index:
            self.last_image_index = expected_image_index
            image_path = self.image_paths[self.last_image_index]
            self.curr_image = cv.imread(image_path)
            self.curr_image = self.curr_image.transpose(2, 0, 1)
            self.curr_image = torch.from_numpy(self.curr_image)
            self.curr_image_name = os.path.basename(image_path)

            self.curr_patches = []
            for (x, y, window) in sliding_window(self.curr_image, self.patch_shape, self.stride):
                self.curr_patches.append((window, (x, y)))

        patch_index = index % self.num_patches_per_image

        patch = self.curr_patches[patch_index][0]
        x, y = self.curr_patches[patch_index][1]
        patch = patch.to(torch.float32)
        patch = patch / 255.0

        return patch, (x, y), self.curr_image_name
