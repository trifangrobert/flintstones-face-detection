import numpy as np
import os
from tqdm import tqdm

import pickle
from utils import calculate_intersection

actors = ["barney", "betty", "fred", "wilma"]
positive_dataset_path = "../train_positive"
negative_dataset_path = "../train_negative"
# actors = ["validare"]
# positive_dataset_path = "../val_positive"
# negative_dataset_path = "../val_negative"

if __name__ == "__main__":
    clusters = 100
    with open(f"top_{clusters}_representative_patches.pkl", "rb") as f:
        patch_sizes = pickle.load(f)

    for actor in actors:
        actor_annotations = os.path.join(positive_dataset_path, actor + "_annotations.txt")
        negatives = []
        all_boxes = dict()
        with open(actor_annotations) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                xmin, ymin, xmax, ymax = int(line[1]), int(line[2]), int(line[3]), int(line[4])
                image_name = line[0]
                all_boxes[image_name] = all_boxes.get(image_name, []) + [(xmin, ymin, xmax, ymax)]

        for negative_count in range(4):
            for k, v in tqdm(all_boxes.items(), desc=actor):
                # choose a random patch from patch_sizes
                found = False
                xmin, ymin, xmax, ymax = 0, 0, 0, 0
                while not found:
                    # choose a random location in the image
                    patch_size = patch_sizes[np.random.randint(len(patch_sizes))]
                    area = patch_size[0] * patch_size[1]
                    while negative_count * 200 <= area <= (negative_count + 1) * 200:
                        patch_size = patch_sizes[np.random.randint(len(patch_sizes))]

                    x = np.random.randint(0, 480 - patch_size[0])
                    y = np.random.randint(0, 360 - patch_size[1])

                    # check if the patch is inside any of the bounding boxes
                    bad_patch = False
                    for box1 in v:
                        box2 = (x, y, x + patch_size[0], y + patch_size[1])
                        intersection_area = calculate_intersection(box1, box2)
                        # intersection_area = round(intersection_area)
                        if intersection_area > 0.0:
                            bad_patch = True
                            break
                    
                    if not bad_patch:
                        xmin, ymin, xmax, ymax = x, y, x + patch_size[0], y + patch_size[1]
                        found = True 

                # this is a negative patch
                negative_annotation = f"{k} {xmin} {ymin} {xmax} {ymax}\n"
                negatives.append(negative_annotation)

        with open(os.path.join(negative_dataset_path, actor + "_annotations.txt"), "w") as f:
            f.writelines(negatives)
            
    
