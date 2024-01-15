import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2 as cv
from tqdm import tqdm
import numpy as np
import pickle

from sliding_window import sliding_window

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


if __name__ == "__main__":
    face_threshold = 1.0
    # dataset_path = "../val_images/validare"
    actor = "betty"
    dataset_path = f"../val_train/{actor}"
    clusters = 30
    # with open(f"top_{clusters}_representative_patches.pkl", "rb") as f:
    with open(f"top_{clusters}_representative_patches_fixed.pkl", "rb") as f:
        patch_shapes = pickle.load(f)

    # model_path = "../saved_models/20/epoch_13_loss_0.00000.pth" # model_resize1_1
    # model_path = "../saved_models/26/epoch_19_loss_0.00005.pth" # model_resize_2
    model_path = "../saved_models/30/epoch_15_loss_0.00000.pth" # model_resize_3
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    print(model)

    files = sorted(os.listdir(dataset_path))
    proposals = {k: [] for k in files}

    for index, patch_shape in enumerate(patch_shapes):
        dataset = FastPatchDataset(dataset_path, patch_shape=patch_shape, stride=10)
        print(f"Processing {index + 1}/{len(patch_shapes)} patch shape {patch_shape} with {dataset.num_patches_per_image} patches per image")

        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        for batch in tqdm(dataloader, desc="Processing batches"):
            patches, top_left_corners, image_names = batch
            top_left_corners = [(x.item(), y.item()) for x, y in zip(top_left_corners[0], top_left_corners[1])]
            patches = patches.to(device)
            output = model(patches)
            output = output.cpu().detach().numpy()
            output = output.squeeze(1)

            for i in range(len(output)):
                face_prob = output[i]
                patch = patches[i]
                top_left_corner = top_left_corners[i][0], top_left_corners[i][1]
                image_name = image_names[i]
                # print(f"top_left_corner: {top_left_corner}")
                # print(f"image_name: {image_name}")
                # print(f"patch_shape: {patch_shape}")
                # print(f"face_prob: {face_prob}")
                if face_prob >= face_threshold:
                    # add to proposals[image_name] list tuple of shape x, y, width, height
                    x = top_left_corner[0]
                    y = top_left_corner[1]
                    height = patch_shape[0]
                    width = patch_shape[1]
                    proposals[image_name].append((x, y, height, width, face_prob))
    print(f"Saved to proposals_{clusters}_{face_threshold}_train_{actor}.pkl")
    with open(f"proposals_{clusters}_{face_threshold}_train_{actor}.pkl", "wb") as f:
        pickle.dump(proposals, f)