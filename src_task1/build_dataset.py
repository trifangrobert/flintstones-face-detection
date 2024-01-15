import numpy as np
import os
from torch.utils.data import Dataset
import cv2 as cv
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import torch

from utils import horizontal_flip, gaussian_noise

class FacialDataset(Dataset):
    def __init__(self, images_path: str, dataset_type: str = "train", positive_path: str = None, negative_path: str = None, transform_type: str = None, patch_shape: tuple = (64, 64)):
        super(FacialDataset, self).__init__()
        self.positive_path = positive_path
        self.negative_path = negative_path
        self.images_path = images_path
        self.patch_shape = patch_shape
        self.dataset_type = dataset_type

        self.data = []
        self.transform = None
        if transform_type == "horizontal_flip":
            self.transform = horizontal_flip
        elif transform_type == "gaussian_noise":
            self.transform = gaussian_noise

        if dataset_type == "train":
            self.actors = ["barney", "betty", "fred", "wilma"]
        elif dataset_type == "validation":
            self.actors = ["validare"]
        self.actor_images = {actor: dict() for actor in self.actors}

        print(f"Loading {dataset_type} data")
        self._load_all_images()
        self._load_data()

    def _load_data(self):
        for label, path in enumerate([self.negative_path, self.positive_path]):
            for actor in tqdm(self.actors, desc=f"Loading {'positive' if label else 'negative'} data"):
                actor_annotations = os.path.join(path, actor + "_annotations.txt")
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
                        image = self.actor_images[actor][image_name]
                        patch = image[ymin:ymax, xmin:xmax]
                        # convert patch to torch tensor of shape channels x height x width
                        patch = patch.transpose(2, 0, 1)
                        patch = torch.from_numpy(patch)

                        # resize the patch to the desired shape REMOVE THIS FOR SPP

                        if self.dataset_type == "train" and label == 1:
                            height = patch.shape[1]
                            width = patch.shape[2]
                            split_height = 1
                            split_width = 1
                            if height >= 128:
                                split_height = 2
                            if width >= 128:
                                split_width = 2
                            if height >= 192:
                                split_height = 3
                            if width >= 192:
                                split_width = 3
                            for i in range(split_height):
                                for j in range(split_width):
                                    sub_patch = patch[:, i * height // split_height: (i + 1) * height // split_height, j * width // split_width: (j + 1) * width // split_width]
                                    sub_patch = torch.nn.functional.interpolate(sub_patch.unsqueeze(0), size=self.patch_shape).squeeze(0)
                                    sub_patch = sub_patch.float() / 255.0
                                    self.data.append((sub_patch, label))

                        patch = torch.nn.functional.interpolate(patch.unsqueeze(0), size=self.patch_shape).squeeze(0)
                        patch = patch.float() / 255.0
                        self.data.append((patch, label))
                        

    def _load_actor_images(self, actor):
        actor_images = {}
        actor_path = os.path.join(self.images_path, actor)
        for file in os.listdir(actor_path):
            image = cv.imread(os.path.join(actor_path, file))
            actor_images[file] = image
        return actor, actor_images

    def _load_all_images(self):
        results = process_map(
            self._load_actor_images, self.actors, max_workers=4, chunksize=1, desc="Loading images"
        )
        for actor, images in results:
            self.actor_images[actor] = images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
        

if __name__ == "__main__":
    # positive_path = "../train_positive"
    # negative_path = "../train_negative"
    # images_path = "../train_images"

    positive_path = "../val_positive"
    negative_path = "../val_negative"
    images_path = "../val_images"
    dataset = FacialDataset(images_path=images_path, positive_path=positive_path, negative_path=negative_path, train_or_test="validation")

    # test the dataset
    for i in range(10):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        # show the image and the label
        cv.imshow(f"Label {label}", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
