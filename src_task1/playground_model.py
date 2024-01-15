import torch
import torch.nn as nn
import pickle
import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

from sliding_window import sliding_window

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # model_path = "../saved_models/19/epoch_18_loss_0.00000.pth"
    model_path = "../saved_models/20/epoch_13_loss_0.00000.pth"
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    
    clusters = 20
    with open(f"top_{clusters}_representative_patches.pkl", "rb") as f:
        patch_sizes = pickle.load(f)

    dataset_path = "../val_images/validare"

    files = sorted(os.listdir(dataset_path))

    for file in files[20:30]:
        image = cv.imread(f"{dataset_path}/{file}")
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image.float() / 255.0

        proposals = torch.zeros_like(image[0])


        for patch_size in tqdm(patch_sizes):
            for (x, y, window) in sliding_window(image, patch_size, step_size=10):
                input = window.unsqueeze(0)
                input = input.to(device)
                output = model(input).squeeze(0)
                output = output.cpu()

                aux_image = image.numpy()
                aux_image = aux_image.transpose(1, 2, 0)
                aux_image = aux_image.copy()


                if output > 0.999:
                    # cv.rectangle(aux_image, (x, y), (x + patch_size[0], y + patch_size[1]), (0, 0, 255), 2)
                    # cv.imshow(f"Probability {output}", aux_image)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
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