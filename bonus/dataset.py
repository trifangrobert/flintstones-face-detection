import torch
from torch import nn
from torch.utils.data import Dataset
import os
import cv2 as cv
from tqdm import tqdm


class FaceDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str, dataset_type: str, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.dataset_type = dataset_type
        self.transform = transform

        self.actor_to_label = {
            "unknown": 0,
            "barney": 1,
            "betty": 2,
            "fred": 3,
            "wilma": 4
        }

        if self.dataset_type == "train":
            self.actors = ["barney", "betty", "fred", "wilma"]
        else:
            self.actors = ["validare"]

        self.data = []
        self._load_data()

    def _load_data(self):
        for actor in tqdm(self.actors, desc=f"Loading {self.dataset_type} data"):
            
            # actor data is a dictionary with keys the image names and values a tuple of shape (image, bounding_boxes_list, labels_list)
            actor_data = {}
            image_actor_path = os.path.join(self.images_path, actor)
            for file in os.listdir(image_actor_path):
                image_path = os.path.join(image_actor_path, file)
                actor_data[file] = (image_path, [], [])
                # image = cv.imread(os.path.join(image_actor_path, file))
                # image = image.transpose(2, 0, 1)
                # image = torch.tensor(image, dtype=torch.float32)
                # image = image / 255.0
                # actor_data[file] = (image, [], [])

            actor_annotations = os.path.join(self.labels_path, actor + "_annotations.txt")
            with open(actor_annotations) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split()
                    xmin, ymin, xmax, ymax = (
                        int(line[1]),
                        int(line[2]),
                        int(line[3]),
                        int(line[4])
                    )
                    image_name = line[0]
                    actor_name = line[5]
                    label = self.actor_to_label[actor_name]
                    actor_data[image_name][1].append((xmin, ymin, xmax, ymax))
                    actor_data[image_name][2].append(label)

            for image_name, (image_path, bounding_boxes, labels) in actor_data.items():
                self.data.append((image_path, bounding_boxes, labels))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path, bounding_boxes, labels = self.data[index]
        image = cv.imread(image_path)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        image = image / 255.0

        targets = {}
        targets["boxes"] = torch.tensor(bounding_boxes, dtype=torch.float32)
        targets["labels"] = torch.tensor(labels, dtype=torch.int64)

        return image, targets
    
class FaceDatasetTest(Dataset):
    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform

        self.image_paths = sorted([os.path.join(dataset_path, image_name) for image_name in os.listdir(dataset_path)])
        self.num_images = len(self.image_paths)

    def __len__(self):
        return self.num_images
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv.imread(image_path)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        image = image / 255.0

        return image, os.path.basename(image_path)

        

if __name__ == "__main__":
    dataset = FaceDataset(images_path="../train_images", labels_path="../train_positive", dataset_type="train")
    print(len(dataset))
    # for index, (image, bounding_boxes, labels) in enumerate(dataset):
    #     cv.imshow("image", image)
    #     cv.waitKey(0)

    #     print(image.shape)
    #     print(bounding_boxes)
    #     print(labels)
    #     if index == 10:
    #         break
    # cv.destroyAllWindows()