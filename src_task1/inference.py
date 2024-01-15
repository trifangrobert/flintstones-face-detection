import torch 
import pickle
from tqdm import tqdm
import numpy as np  
import time
import os
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

from sliding_window import sliding_window

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class PatchDataset(Dataset):
    def __init__(self, images_path: str, clusters_patches: str, stride: int = 10):
        super(PatchDataset, self).__init__()
        self.images_path = images_path
        self.image_paths = [os.path.join(images_path, image_name) for image_name in os.listdir(images_path)]
        self.image_paths.sort()
        self.last_image_index = -1
        self.curr_image = None
        self.curr_image_name = None

        self.patch_sizes = []
        with open(f"top_{clusters_patches}_representative_patches.pkl", "rb") as f:
            self.patch_sizes = pickle.load(f)

        self.stride = stride

    def __len__(self):
        return len(self.patch_sizes) * len(self.image_paths)
    
    def __getitem__(self, index):
        expected_image_index = index // len(self.patch_sizes)
        if expected_image_index != self.last_image_index:
            self.last_image_index = expected_image_index
            image_path = self.image_paths[self.last_image_index]
            self.curr_image = cv.imread(image_path)
            self.curr_image = self.curr_image.transpose(2, 0, 1)
            self.curr_image = torch.from_numpy(self.curr_image)
            self.curr_image_name = os.path.basename(image_path)

        patch_index = index % len(self.patch_sizes)
        patch_size = self.patch_sizes[patch_index]

        # Extract patches and their top-left corners
        patches, top_left_corners = self._extract_patches_with_coords(self.curr_image, patch_size)

        return self.curr_image, patches, top_left_corners, self.curr_image_name, patch_size

    def _extract_patches_with_coords(self, image, patch_size):
        # return some random dummy data
        # patches = torch.rand((100, 3, patch_size[0], patch_size[1]))
        
        patches = image.unfold(1, patch_size[0], self.stride).unfold(2, patch_size[1], self.stride)
        num_patches_h = patches.size(1)
        num_patches_w = patches.size(2)
        patches = patches.contiguous().view(-1, 3, patch_size[0], patch_size[1])
        patches = patches.to(torch.float32)
        patches = patches / 255.0

        # compute the top-left corner of each patch as tensor of shape (num_patches, 2)
        top_left_corners = torch.zeros((num_patches_h * num_patches_w, 2), dtype=torch.int)
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                top_left_corners[i * num_patches_w + j, 0] = i * self.stride
                top_left_corners[i * num_patches_w + j, 1] = j * self.stride
        
        top_left_corners = top_left_corners.squeeze(1)
        return patches, top_left_corners

if __name__ == "__main__":
    dataset = PatchDataset("../val_images/validare", clusters_patches=10, stride=5)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # model_path = "../saved_models/10/accuracy_0.01075.pth"
    model_path = "../saved_models/17/accuracy_0.00000.pth"
    if os.path.exists(model_path) and os.path.isfile(model_path):
        model = torch.load(model_path)
    else:
        print("Model not found")
        exit(1)
    # model = torch.load("../saved_models/10/accuracy_0.01075.pth")
    # model = torch.load("../saved_models/13/accuracy_0.00069.pth")
    # model = torch.load("../saved_models/17/accuracy_0.00000_1.pth")
    model = model.to(device)
    model.eval()

    proposals = dict()
    steps = 10
    start_time = time.time()
    K = 4
    for image, patches, top_left_corners, image_name, patch_size in tqdm(dataloader, desc="Sliding window"):
        image = cv.imread(f"../val_images/validare/{image_name[0]}")

        top_left_corners = top_left_corners.squeeze()
        patches = patches.squeeze()

        # empty probs
        probs = []

        chunks = torch.chunk(patches, K, dim=0)
        for chunk in chunks:
            chunk = chunk.to(device)
            labels = model(chunk)
            labels = labels.squeeze()
            labels = labels.cpu()
            probs.append(labels)

        probs = torch.cat(probs, dim=0)
            
        image_shape = (image.shape[0], image.shape[1])
        
        if image_name[0] not in proposals:
            proposals[image_name[0]] = torch.zeros(image_shape, dtype=torch.uint8)
        
        # draw the bounding boxes
        for p in range(probs.size(0)):
            if probs[p] > 0.999:
                x_min = top_left_corners[p, 0].item()
                y_min = top_left_corners[p, 1].item()
                x_max = x_min + patch_size[0].item()
                y_max = y_min + patch_size[1].item()
                # patch_image = patches[p].cpu().numpy()
                # patch_image = patch_image.transpose(1, 2, 0)
                # print(patch_image.dtype, patch_image.shape, type(patch_image))
                aux_image = image.copy()
                # cv.rectangle(aux_image, (y_min, x_min), (y_max, x_max), (0, 255, 0), 1)
                # cv.imshow("image", aux_image)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                proposals[image_name[0]][x_min:x_max, y_min:y_max] += 1

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds")

    # show the proposals
    for image_name, image in proposals.items():
        original_image = cv.imread(f"../val_images/validare/{image_name}")
        image = image.numpy()
        image = image / image.max()

        # threshold = 0.3
        # image[image < threshold] = 0.0
        # image[image >= threshold] = 1.0

        image = (image * 255).astype(np.uint8)
        
        cv.imshow("original image", original_image)
        cv.imshow("proposals", image)
        cv.waitKey(0)
        cv.destroyAllWindows()


    