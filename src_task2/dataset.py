import numpy as np
import os
from torch.utils.data import Dataset
import cv2 as cv
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import torch

from utils import horizontal_flip, gaussian_noise

class FacialDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str, dataset_type: str, transform_type: str = None, patch_shape: tuple = (64, 64)):
        self.images_path = images_path
        self.patch_shape = patch_shape
        self.labels_path = labels_path
        self.dataset_type = dataset_type
        
        self.transform = None
        if transform_type == "horizontal_flip":
            self.transform = horizontal_flip
        elif transform_type == "gaussian_noise":
            self.transform = gaussian_noise
        
        print(f"Applied transformations: {transform_type}")
        if dataset_type == "train":
            self.actors = ["barney", "betty", "fred", "wilma"]
        elif dataset_type == "validation":
            self.actors = ["validare"]

        self.actor_to_label = {
            "unknown": 0,
            "barney": 1,
            "betty": 2,
            "fred": 3,
            "wilma": 4
        }

        self.actor_images = {actor: dict() for actor in self.actor_to_label.keys()}

        self.data = []
        self._load_images()
        self._load_labels()

    def _load_actor_images(self, actor):
        actor_images = {}
        actor_path = os.path.join(self.images_path, actor)
        for file in os.listdir(actor_path):
            image = cv.imread(os.path.join(actor_path, file))
            actor_images[file] = image
        return actor, actor_images

    def _load_images(self):
        results = process_map(
            self._load_actor_images, self.actors, max_workers=4, chunksize=1, desc=f"Loading images for {self.dataset_type} dataset"
        )
        for actor, images in results:
            self.actor_images[actor] = images


    def _load_labels(self):
        for actor in tqdm(self.actors, desc=f"Loading labels for {self.dataset_type} dataset"):
            actor_annotations = os.path.join(self.labels_path, actor + "_annotations.txt")
            with open(actor_annotations) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split()
                    xmin, ymin, xmax, ymax = (
                        int(line[1]),
                        int(line[2]),
                        int(line[3]),
                        int(line[4]),
                    )
                    image_name = line[0]
                    label = self.actor_to_label[line[5]]

                    image = self.actor_images[actor][image_name]
                    patch = image[ymin:ymax, xmin:xmax]
                    # convert patch to torch tensor of shape channels x height x width
                    patch = patch.transpose(2, 0, 1)
                    patch = torch.from_numpy(patch)

                    patch = torch.nn.functional.interpolate(patch.unsqueeze(0), size=self.patch_shape).squeeze(0)
                    patch = patch.float() / 255.0

                    # label should be one hot encoded
                    label = torch.eye(5)[label]
                    self.data.append((patch, label))

                    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
    
if __name__ == "__main__":
    dataset = FacialDataset(images_path="../train_images", labels_path="../train_positive")
    print(len(dataset))

    for index, (image, label) in enumerate(dataset):
        cv.imshow(f"Label {label}", image)
        cv.waitKey(0)
        if index >= 50:
            break

    cv.destroyAllWindows()
