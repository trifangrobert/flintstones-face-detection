import torch
from torch.utils.data import Dataset
import os
import cv2 as cv
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import pickle
from scipy import ndimage


from sliding_window import  sliding_window

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_sliding_window_patch(patch_size, image, step_size=10):
    print(f"Worker {os.getpid()} processing patch size {patch_size}")
    model = torch.load("../saved_models/10/accuracy_0.01075.pth")  # Load model in each worker
    res = torch.zeros_like(image[0])

    for (x, y, window) in sliding_window(image, patch_size, step_size):
        output = model(window.unsqueeze(0)).squeeze(0)

        if output > 0.99:
            res[x:x + patch_size[0], y:y + patch_size[1]] += 1

    print(f"Worker {os.getpid()} finished processing patch size {patch_size}")
    return res


def find_islands_and_bounding_boxes(array):
    # Label the connected components
    labeled_array, num_features = ndimage.label(array)

    # Find the bounding boxes
    bounding_boxes = []
    for label in range(1, num_features + 1):
        # Find the coordinates of the pixels with this label
        positions = np.argwhere(labeled_array == label)

        # Find the bounding box
        xmin = np.min(positions[:, 0])
        xmax = np.max(positions[:, 0])
        ymin = np.min(positions[:, 1])
        ymax = np.max(positions[:, 1])

        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        
        bounding_box = (tuple(top_left), tuple(bottom_right))
        bounding_boxes.append(bounding_box)

    return labeled_array, bounding_boxes


if __name__ == "__main__":
    clusters = 10
    with open(f"top_{clusters}_representative_patches.pkl", "rb") as f:
        patch_sizes = pickle.load(f)

    image_name = "0001.jpg"
    gt_bounding_boxes = []
    with open(f"../val/validare_annotations.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if line[0] == image_name:
                gt_bounding_boxes.append((int(line[1]), int(line[2]), int(line[3]), int(line[4])))


    model = torch.load("../saved_models/10/accuracy_0.01075.pth")
    model = model.to(device)
    model.eval()

    for index, image_name in enumerate(os.listdir("../val_images/validare")):

        image = cv.imread(f"../val_images/validare/{image_name}")
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image.float() / 255.0

        proposals = torch.zeros_like(image[0])


        for patch_size in tqdm(patch_sizes, desc="Sliding window"):
            for (x, y, window) in sliding_window(image, patch_size, step_size=10):
                input = window.unsqueeze(0).to(device)
                output = model(input).squeeze(0)

                if output > 0.95:
                    # add one to the bounding box of the proposal
                    proposals[x:x + patch_size[0], y:y + patch_size[1]] += 1    

        proposals = proposals / proposals.max()
        proposals = proposals.numpy()
        proposals = (proposals * 255).astype(np.uint8)

        image = image.numpy()
        image = image.transpose(1, 2, 0)

        cv.imshow("image", image)
        cv.imshow("proposals", proposals)
        cv.waitKey(0)
        cv.destroyAllWindows()

        if index == 10:
            break


    # threshold = 0.7
    
    # labeled_array, bounding_boxes = find_islands_and_bounding_boxes(proposals)

    # # with open("proposals.pkl", "wb") as f:
    # #     pickle.dump(bounding_boxes, f)

    # # with open("proposals.pkl", "rb") as f:
    # #     bounding_boxes = pickle.load(f)
    
    # # show the image
    # image = image.numpy()
    # image = image.transpose(1, 2, 0)

    # print(gt_bounding_boxes)

    # # show the proposals
    # for bounding_box in bounding_boxes:
    #     top_left, bottom_right = bounding_box
    #     top_left = tuple(top_left[::-1])
    #     bottom_right = tuple(bottom_right[::-1])
    #     cv.rectangle(image, top_left, bottom_right, color=(0, 0, 255), thickness=2)

    # # show the ground truth bounding boxes
    # for bounding_box in gt_bounding_boxes:
    #     top_left, bottom_right = bounding_box[:2], bounding_box[2:]
    #     top_left = tuple(top_left)
    #     bottom_right = tuple(bottom_right)
    #     cv.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)


    # cv.imshow("proposals", proposals)
    # cv.imshow("image", image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

                       
    