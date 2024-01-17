import torch
import numpy as np
import cv2 as cv
import pickle
from tqdm import tqdm
import os
import argparse
import time
import datetime
from torchvision.models import resnet50
import torch.nn as nn
from torch.utils.data import DataLoader


from fast_inference import FastPatchDataset
from utils import nms_based_on_overlap
from model_cnn import FaceDetector

from checker import evaluate_results_task1, evaluate_results_task2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceRecognizer:
    def __init__(
        self,
        dataset_path: str,
        patch_shapes_path: str,
        model_localize_path: str,
        process_config_path: str,
        model_classify_path: str,
        face_threshold: float = 1.0,
        save_path: str = None,
        ground_truth_path: str = None,
    ):
        self.dataset_path = dataset_path
        self.patch_shapes_path = patch_shapes_path
        self.model_localize_path = model_localize_path
        self.model_classify_path = model_classify_path
        self.face_threshold = face_threshold
        self.process_config_path = process_config_path
        self.save_path = save_path
        self.ground_truth_path = ground_truth_path
        self.face_bounding_boxes = []

        self.actors = ["barney", "betty", "fred", "wilma"]
        self.actor_to_label = {
            "unknown": 0,
            "barney": 1,
            "betty": 2,
            "fred": 3,
            "wilma": 4
        }
        self.label_to_actor = {v: k for k, v in self.actor_to_label.items()}

        print(f"Dataset path: {self.dataset_path}")
        print(f"Patch shapes path: {self.patch_shapes_path}")
        print(f"Model for localization path: {self.model_localize_path}")
        print(f"Model for classification path: {self.model_classify_path}")
        print(f"Process config path: {self.process_config_path}")
        print(f"Save path: {self.save_path}")
        print(f"Ground truth path: {self.ground_truth_path}")

    def _sliding_window_proposals(self):
        with open(self.patch_shapes_path, "rb") as f:
            self.patch_shapes = pickle.load(f)
        self.patch_shapes = self.patch_shapes[:5]

        self.model_localize = FaceDetector()
        self.model_localize.load_state_dict(torch.load(self.model_localize_path, map_location=device))
        self.model_localize = self.model_localize.to(device)
        self.model_localize.eval()

        self.files = sorted(os.listdir(self.dataset_path))
        self.proposals = {k: [] for k in self.files}

        for index, patch_shape in enumerate(self.patch_shapes):
            dataset = FastPatchDataset(
                self.dataset_path, patch_shape=patch_shape, stride=10
            )
            print(
                f"Processing {index + 1}/{len(self.patch_shapes)} patch shape {patch_shape} with {dataset.num_patches_per_image} patches per image"
            )

            dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

            for batch in tqdm(dataloader, desc="Processing batches"):
                patches, top_left_corners, image_names = batch
                top_left_corners = [
                    (x.item(), y.item())
                    for x, y in zip(top_left_corners[0], top_left_corners[1])
                ]
                patches = patches.to(device)
                output = self.model_localize(patches)
                output = output.cpu().detach().numpy()
                output = output.squeeze(1)

                for i in range(len(output)):
                    face_prob = output[i]
                    patch = patches[i]
                    top_left_corner = top_left_corners[i][0], top_left_corners[i][1]
                    image_name = image_names[i]
                    if face_prob >= self.face_threshold:
                        x = top_left_corner[0]
                        y = top_left_corner[1]
                        height = patch_shape[0]
                        width = patch_shape[1]
                        self.proposals[image_name].append(
                            (x, y, height, width, face_prob)
                        )

    def _detect_faces_from_heatmap(
        self, heatmap, threshold_value, model_bounding_boxes, image_name
    ):
        # Apply threshold
        props_before = heatmap.copy()
        _, binary_heatmap = cv.threshold(
            heatmap, threshold_value, 255, cv.THRESH_BINARY
        )
        bin_heatmap = binary_heatmap.copy()
        bin_heatmap[bin_heatmap == 255] = 1
        props_after = props_before * bin_heatmap

        # Find contours or connected regions
        contours, _ = cv.findContours(
            binary_heatmap.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # Convert contours to bounding boxes
        bounding_boxes = [cv.boundingRect(contour) for contour in contours]

        bounding_boxes = [
            (x, y, h, w)
            for x, y, h, w in bounding_boxes
            if h * w >= self.config["min_face_area"]
        ]  # remove small boxes

        bounding_boxes = nms_based_on_overlap(
            bounding_boxes, iou_threshold=self.config["iou_threshold"]
        )
        bounding_boxes_with_scores = []
        image = cv.imread(f"{self.dataset_path}/{image_name}")

        self.model_localize.eval()

        for bbox in bounding_boxes:
            x, y, h, w = bbox
            patch = image[y : y + w, x : x + h]
            patch = patch.transpose(2, 0, 1)
            patch = torch.from_numpy(patch)
            patch = patch.float() / 255.0
            patch = torch.nn.functional.interpolate(
                patch.unsqueeze(0), size=(64, 64)
            ).squeeze(0)
            patch = patch.unsqueeze(0).to(device)
            score = self.model_localize(patch).item()
            if score == 0.0:
                continue
            bounding_boxes_with_scores.append((x, y, h, w, score))

        return image, bounding_boxes_with_scores

    def _process_proposals(self):
        with open(self.process_config_path, "rb") as f:
            self.config = pickle.load(f)

        self.predictions_file_names = []
        self.predictions_bounding_boxes = []
        self.predictions_scores = []

        for image_name, boxes in tqdm(
            self.proposals.items(), desc="Processing proposals"
        ):
            boxes = [
                (x, y, height, width, score)
                for x, y, height, width, score in boxes
                if height * width <= self.config["max_patch_area"]
            ]
            props = np.zeros((360, 480))

            for box in boxes:
                x, y, height, width, score = box
                props[x : x + height, y : y + width] += score

            props = props / props.max()
            props = (props * 255).astype(np.uint8)

            image, bounding_boxes = self._detect_faces_from_heatmap(
                props, self.config["heatmap_threshold"], boxes, image_name
            )
            for bbox in bounding_boxes:
                x, y, h, w, score = bbox
                self.predictions_file_names.append(image_name)
                self.predictions_bounding_boxes.append((x, y, x + h, y + w))
                self.predictions_scores.append(score)
                self.face_bounding_boxes.append((image_name, (x, y, x + h, y + w), score))

    def _classify_faces(self):
        self.model_classify = resnet50(weights=None)
        num_ftrs = self.model_classify.fc.in_features
        num_classes = 5
        self.model_classify.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1),
        )

        # self.model_classify = ResNet50(num_classes=5)
        self.model_classify.load_state_dict(torch.load(self.model_classify_path))
        self.model_classify = self.model_classify.to(device)
        self.model_classify.eval()

        self.predictions = {actor : {"file_names": [], "bounding_boxes": [], "scores": []} for actor in self.actors}
        expand = 0

        for image_name, bbox, _ in tqdm(self.face_bounding_boxes, desc="Classifying faces"):
            xmin, ymin, xmax, ymax = bbox
            image = cv.imread(f"{self.dataset_path}/{image_name}")
            
            patch = image[ymin - expand:ymax + expand, xmin - expand:xmax + expand]
            patch = patch.transpose(2, 0, 1)
            patch = torch.from_numpy(patch)
            patch = patch.float() / 255.0
            patch = torch.nn.functional.interpolate(patch.unsqueeze(0), size=(224, 224)).squeeze(0)
            patch = patch.unsqueeze(0).to(device)
            output = self.model_classify(patch)
            output = output.cpu().detach().numpy()
            output = output.squeeze(0)
            actor = self.label_to_actor[output.argmax()]
            if actor == "unknown":
                continue
            self.predictions[actor]["file_names"].append(image_name)
            self.predictions[actor]["bounding_boxes"].append(bbox)
            score = output.max() 
            self.predictions[actor]["scores"].append(score)

    def _save_results_task1(self):
        SAVE_PATH = self.save_path
        SAVE_PATH = os.path.join(SAVE_PATH, "task1")
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"Saving results to {SAVE_PATH}")

        with open(f"{SAVE_PATH}/detections_all_faces.npy", "wb") as f:
            np.save(f, self.predictions_bounding_boxes, allow_pickle=True)

        with open(f"{SAVE_PATH}/scores_all_faces.npy", "wb") as f:
            np.save(f, self.predictions_scores, allow_pickle=True)

        with open(f"{SAVE_PATH}/file_names_all_faces.npy", "wb") as f:
            np.save(f, self.predictions_file_names, allow_pickle=True)

    def _save_results_task2(self):
        SAVE_PATH = self.save_path
        SAVE_PATH = os.path.join(SAVE_PATH, "task2")
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"Saving results to {SAVE_PATH}")

        for actor in self.actors:
            with open(f"{SAVE_PATH}/detections_{actor}.npy", "wb") as f:
                np.save(f, self.predictions[actor]["bounding_boxes"], allow_pickle=True)

            with open(f"{SAVE_PATH}/scores_{actor}.npy", "wb") as f:
                np.save(f, self.predictions[actor]["scores"], allow_pickle=True)

            with open(f"{SAVE_PATH}/file_names_{actor}.npy", "wb") as f:
                np.save(f, self.predictions[actor]["file_names"], allow_pickle=True)


    def _check_task1(self):
        solution_path_root = self.save_path
        ground_truth_path_root = self.ground_truth_path

        solution_path = solution_path_root + "task1/"
        ground_truth_path = ground_truth_path_root + "task1_gt_validare.txt"

        evaluate_results_task1(solution_path, ground_truth_path, verbose=0)

    def _check_task2(self):
        solution_path_root = self.save_path
        ground_truth_path_root = self.ground_truth_path

        solution_path = solution_path_root + "task2/"
        
        ground_truth_path = ground_truth_path_root + "task2_fred_gt_validare.txt"
        evaluate_results_task2(solution_path, ground_truth_path, "fred", verbose=0)

        ground_truth_path = ground_truth_path_root + "task2_barney_gt_validare.txt"
        evaluate_results_task2(solution_path, ground_truth_path, "barney", verbose=0)

        ground_truth_path = ground_truth_path_root + "task2_betty_gt_validare.txt"
        evaluate_results_task2(solution_path, ground_truth_path, "betty", verbose=0)

        ground_truth_path = ground_truth_path_root + "task2_wilma_gt_validare.txt"
        evaluate_results_task2(solution_path, ground_truth_path, "wilma", verbose=0)

    def run(self):
        start_time = time.time()
        # task 1
        print("\n\nRunning task 1\n\n")
        self._sliding_window_proposals()
        self._process_proposals()

        # task 2
        print("\n\nRunning task 2\n\n")
        self._classify_faces()

        # save results
        if self.save_path is not None:
            print("Saving results")
            self._save_results_task1()                
            self._save_results_task2()

            if self.ground_truth_path is not None:
                self._check_task1()
                self._check_task2()
        
        finish_time = time.time()
        elapsed_time = str(datetime.timedelta(seconds=finish_time - start_time))
        print(f"Elapsed time: {elapsed_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("-p", "--patch_shapes_path", type=str, required=True, help="Path to patch shapes")
    parser.add_argument("-ml", "--model_localization_path", type=str, required=True, help="Path to model for localization")
    parser.add_argument("-mc", "--model_classification_path", type=str, required=True, help="Path to model for classification")
    parser.add_argument("-c", "--process_config_path", type=str, required=True, help="Path to process config")

    parser.add_argument("-s", "--save_path", type=str, default=None, help="Path to save results")
    parser.add_argument("-g", "--ground_truth_path", type=str, default=None, help="Path to ground truth")
    args = parser.parse_args()

    face_localizer = FaceRecognizer(
        dataset_path=args.dataset_path,
        patch_shapes_path=args.patch_shapes_path,
        model_localize_path=args.model_localization_path,
        model_classify_path=args.model_classification_path,
        process_config_path=args.process_config_path,
        save_path=args.save_path,
        ground_truth_path=args.ground_truth_path
    )
    face_localizer.run()
