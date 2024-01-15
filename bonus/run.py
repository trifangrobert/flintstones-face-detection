from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import torch
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import mobilenet_v2

from checker import evaluate_results_task1, evaluate_results_task2
from dataset import FaceDatasetTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def non_max_suppression(predictions, iou_threshold=0.5):
    keep = []

    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    scores = predictions[:, 4]

    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.float32)
    x2 = x2.astype(np.float32)
    y2 = y2.astype(np.float32)

    area = (x2 - x1) * (y2 - y1)
    indices = np.argsort(scores)[::-1]

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        xx1 = np.maximum(x1[current], x1[indices[1:]])
        yy1 = np.maximum(y1[current], y1[indices[1:]])
        xx2 = np.minimum(x2[current], x2[indices[1:]])
        yy2 = np.minimum(y2[current], y2[indices[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        intersection = w * h
        union = area[current] + area[indices[1:]] - intersection
        iou = intersection / union

        indices = indices[np.where(iou <= iou_threshold)[0] + 1]

    return keep

class FaceRecognizer:
    def __init__(self, dataset_path: str, model_path: str, save_path: str = None, ground_truth_path: str = None):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.save_path = save_path
        self.ground_truth_path = ground_truth_path
        num_classes = 5

        # backbone = mobilenet_v2(weights=None).features
        # backbone.out_channels = 1280
        # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        # self.model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

        backbone = resnet_fpn_backbone('resnet18', weights=None)
        self.model = FasterRCNN(backbone=backbone, num_classes=num_classes)        
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.actor_to_label = {
            "unknown": 0,
            "barney": 1,
            "betty": 2,
            "fred": 3,
            "wilma": 4
        }
        self.label_to_actor = {v: k for k, v in self.actor_to_label.items()}

        self.predictions_bounding_boxes = []
        self.predictions_scores = []
        self.predictions_file_names = []
        self.actors = ["barney", "betty", "fred", "wilma"]
        self.predictions = {actor : {"file_names": [], "bounding_boxes": [], "scores": []} for actor in self.actors}

    def _load_data(self):
        self.dataset = FaceDatasetTest(dataset_path=self.dataset_path)
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False)


    def _inference(self):
        threshold = 0.7
        self.model.eval()
        for images, image_names in tqdm(self.dataloader):
            images = [image.to(device) for image in images]

            with torch.no_grad():
                outputs = self.model(images)

            for output, image_name in zip(outputs, image_names):
                image_name = image_name
                pred_bounding_boxes = output["boxes"].cpu().numpy().astype(int)
                pred_labels = output["labels"].cpu().numpy()
                pred_labels = [self.label_to_actor[x] for x in pred_labels]
                pred_scores = output["scores"].cpu().numpy()

                pred_labels = np.array(pred_labels).reshape(-1, 1)
                pred_scores = np.array(pred_scores).reshape(-1, 1)

                pred_scores = pred_scores.astype(np.float32)
                pred_bounding_boxes = np.concatenate((pred_bounding_boxes, pred_scores, pred_labels), axis=1)

                pred_bounding_boxes = pred_bounding_boxes[non_max_suppression(pred_bounding_boxes)]

                pred_labels = pred_bounding_boxes[:, 5]
                pred_scores = pred_bounding_boxes[:, 4]
                pred_bounding_boxes = pred_bounding_boxes[:, :4]

                for label, score, bounding_box in zip(pred_labels, pred_scores, pred_bounding_boxes):
                    xmin, ymin, xmax, ymax = bounding_box
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)
                    actor_name = str(label)
                    score = float(score)
                    if score > threshold:
                        self.predictions_bounding_boxes.append((xmin, ymin, xmax, ymax))
                        self.predictions_scores.append(score)
                        self.predictions_file_names.append(image_name)
                        if actor_name in self.actors:
                            self.predictions[actor_name]["file_names"].append(image_name)
                            self.predictions[actor_name]["bounding_boxes"].append((xmin, ymin, xmax, ymax))
                            self.predictions[actor_name]["scores"].append(score)

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
        # load data
        print("Loading data")
        self._load_data()

        # task 1 + task 2
        print("Running inference")
        self._inference()

        # save results
        if self.save_path is not None:
            print("Saving results")
            self._save_results_task1()                
            self._save_results_task2()

            if self.ground_truth_path is not None:
                print("Checking results")
                self._check_task1()
                self._check_task2()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--dataset_path", type=str, required=True)
    argparser.add_argument("-m", "--model_path", type=str, required=True)
    argparser.add_argument("-s", "--save_path", type=str)
    argparser.add_argument("-g", "--ground_truth_path", type=str)
    args = argparser.parse_args()

    face_recognizer = FaceRecognizer(dataset_path=args.dataset_path, model_path=args.model_path, save_path=args.save_path, ground_truth_path=args.ground_truth_path)
    face_recognizer.run()