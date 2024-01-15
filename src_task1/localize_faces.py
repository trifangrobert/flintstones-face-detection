import torch
import pickle
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from model_resize_3 import FaceDetector
from utils import calculate_iou

save = False
steps = 50
config = {
    "clusters": 30,
    "face_threshold": 1.0,
    "iou_threshold": 0.9,
    "max_patch_area": 9000,
    "heatmap_threshold": 90,
    "min_face_area": 1000
}

# gt_path = "../val/validare_annotations.txt"
# data_path = "../val/validare"

actor = "barney"
gt_path = f"../val_train/{actor}_annotations.txt"
data_path = f"../val_train/{actor}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_gt_boxes(image_name):
    gt_boxes = []
    with open(f"{gt_path}", "r") as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(" ")
            if line[0] == image_name:
                gt_boxes.append((int(line[1]), int(line[2]), int(line[3]), int(line[4])))
    return gt_boxes

def nms_based_on_overlap(boxes, iou_threshold):
    # Convert to [(x1, y1, x2, y2), ...] format
    boxes = [(x, y, x + height, y + width) for x, y, height, width in boxes]
    boxes = sorted(boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
    selected_boxes = []
    while boxes:
        current_box = boxes.pop(0)
        boxes = [box for box in boxes if calculate_iou(current_box, box) < iou_threshold]
        selected_boxes.append(current_box)
    
    selected_boxes = [(x, y, x_max - x, y_max - y) for x, y, x_max, y_max in selected_boxes]
    return selected_boxes

def dbscan_clustering(heatmap):
    # Standardize the data
    x_coords, y_coords = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
    points = np.stack([x_coords.ravel(), y_coords.ravel(), heatmap.ravel()], axis=1)

    scaler = StandardScaler()
    heatmap_scaled = scaler.fit_transform(points)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.4, min_samples=25)
    clusters = dbscan.fit_predict(heatmap_scaled)
    
    unique_clusters = np.unique(clusters)
    # print(unique_clusters)
    # print(heatmap.shape, clusters.shape)

    # Dictionary to hold bounding boxes for each cluster
    bounding_boxes = []

    for cluster in unique_clusters:
        if cluster != -1:  # Ignoring noise points, which are labeled as -1
            # Extract points belonging to the current cluster
            cluster_points = points[clusters == cluster, :2]  # Get only x and y coordinates

            # Find minimum and maximum coordinates
            min_x, min_y = np.min(cluster_points, axis=0)
            max_x, max_y = np.max(cluster_points, axis=0)

            # Define bounding box (min_x, min_y, max_x, max_y)
            bounding_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

    return bounding_boxes

def solve_close_faces_dbscan(bounding_boxes, props_after):
    bounding_boxes_to_add = []
    for bbox in bounding_boxes:
        x, y, h, w = bbox
        if h / w > 1.4:
            # aux_image = image.copy()
            # cv.rectangle(aux_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
            bounding_boxes.remove(bbox)
            # this means that there are multiple faces in the bounding box
            # try to cluster them
            props_for_patch = props_after[y:y+w, x:x+h]

            # remove small values
            props_for_patch[props_for_patch < 130] = 0

            # do erosion
            kernel = np.ones((3, 3), np.uint8)
            props_for_patch = cv.erode(props_for_patch, kernel, iterations=10)

            # do dilation
            kernel = np.ones((3, 3), np.uint8)
            props_for_patch = cv.dilate(props_for_patch, kernel, iterations=5)
            
            # cluster with DBSCAN
            props_for_patch = props_for_patch / props_for_patch.max()
            props_for_patch = (props_for_patch * 255).astype(np.uint8)
            
            # cv.imshow("props_for_patch", props_for_patch)

            split_bounding_boxes = dbscan_clustering(props_for_patch)
            # remove the bounding box with the biggest area
            split_bounding_boxes = sorted(split_bounding_boxes, key=lambda x: x[2] * x[3], reverse=True)
            split_bounding_boxes = split_bounding_boxes[1:]
            
            # print(len(split_bounding_boxes))
            if len(split_bounding_boxes) <= 1:
                continue
            for split_bbox in split_bounding_boxes:
                x_min, y_min, height, width = split_bbox
                x_min = x_min + x
                y_min = y_min + y
                mean_value = props_after[y_min:y_min+width, x_min:x_min+height].mean()
                # print(x_min, y_min, height, width, mean_value)
                if mean_value < 50:
                    continue

                # cv.rectangle(aux_image, (x_min, y_min), (x_min + height, y_min + width), (0, 0, 255), 2)
                bounding_boxes_to_add.append((x_min, y_min, height, width))
            # cv.imshow(f"{image_name}", aux_image)
            # cv.waitKey(0)
            
    bounding_boxes.extend(bounding_boxes_to_add)
    return bounding_boxes

def solve_close_faces_peaks(bounding_boxes, props_after, model_bounding_boxes, image_name):
    bounding_boxes_to_add = []
    for bbox in bounding_boxes:
        bbox_x, bbox_y, bbox_h, bbox_w = bbox
        if (bbox_h / bbox_w > 1.3 or bbox_w / bbox_h > 1.3) or bbox_h * bbox_w > 10000:
            # find the highest peak
            bounding_boxes.remove(bbox)
            tries = 10
            while True:
                tries -= 1
                if tries == 0:
                    break
                mx_peak_value = props_after[bbox_y:bbox_y+bbox_w, bbox_x:bbox_x+bbox_h].max()
                if mx_peak_value == 0:
                    break
                mx_peak = np.where(props_after[bbox_y:bbox_y+bbox_w, bbox_x:bbox_x+bbox_h] == mx_peak_value)

                # cv.imshow(f"{mx_peak_value} at {mx_peak[1][0]} {mx_peak[0][0]}", props_after[bbox_y:bbox_y+bbox_w, bbox_x:bbox_x+bbox_h])
                # cv.waitKey(0)
                # cv.destroyAllWindows()

                #threshold of face detection
                threshold = 90

                # look for values that are close to the maximum peak and have a value greater than the mx_peak - threshold
                start_x = mx_peak[1][0] + bbox_x
                start_y = mx_peak[0][0] + bbox_y
                # print(start_x, start_y, mx_peak[1][0], mx_peak[0][0], bbox_x, bbox_y)
                # img = cv.imread(f"{data_path}/{image_name}")
                # cv.rectangle(img, (start_x, start_y), (start_x + 10, start_y + 10), (0, 0, 255), 2)
                # cv.imshow(f"Image {start_x} {start_y}", img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

                stk = [(start_x, start_y)]
                visited = np.zeros(props_after.shape)
                visited[start_y, start_x] = 1

                # print(start_x, start_y, props_after[start_y, start_x])
                face_bbox = (start_x, start_y, start_x, start_y)
                while stk:
                    prev_x, prev_y = stk.pop()
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            new_x, new_y = prev_x + i, prev_y + j
                            if i == 0 and j == 0:
                                continue
                            if new_x < 0 or new_x >= props_after.shape[1] or new_y < 0 or new_y >= props_after.shape[0]:
                                continue
                            if visited[new_y, new_x] == 1:
                                continue
                            # print(x + i, new_y, props_after[new_y, x + i])
                            if props_after[new_y, new_x] > mx_peak_value - threshold:
                                stk.append((new_x, new_y))
                                face_bbox = (min(face_bbox[0], new_x), min(face_bbox[1], new_y), max(face_bbox[2], new_x), max(face_bbox[3], new_y))
                                visited[new_y, new_x] = 1

                # cv.rectangle(img, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0, 255, 0), 2)
                # cv.imshow("img", img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

                # remove all bounding boxes that intersect with the face_bbox
                for bbox in model_bounding_boxes:
                    x, y, h, w, score = bbox
                    # aux_image = img.copy()
                    model_bbox = (y, x, y + w, x + h)
                    if calculate_iou(face_bbox, model_bbox) > 0.2:
                        # cv.rectangle(aux_image, (y, x), (y + w, x + h), (0, 255, 0), 2)
                        # cv.imshow("aux_image", aux_image)
                        # cv.waitKey(0)
                        # cv.destroyAllWindows()
                        props_after[x:x+h, y:y+w] = 0

                # add the face_bbox
                face_bbox = (face_bbox[0], face_bbox[1], face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1])
                if face_bbox[2] * face_bbox[3] >= 300 * 400:
                    break
                bounding_boxes_to_add.append(face_bbox)
                
    bounding_boxes.extend(bounding_boxes_to_add)
    # img = cv.imread(f"{data_path}/{image_name}")
    # for bbox in bounding_boxes:
    #     x, y, h, w = bbox
    #     print(x, y, h, w)
    #     cv.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
    #     cv.imshow("img", img)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
    return bounding_boxes


def detect_faces_from_heatmap(heatmap, threshold_value, model_bounding_boxes, image_name):
    # Apply threshold
    props_before = heatmap.copy()
    _, binary_heatmap = cv.threshold(heatmap, threshold_value, 255, cv.THRESH_BINARY)
    bin_heatmap = binary_heatmap.copy()
    bin_heatmap[bin_heatmap == 255] = 1
    props_after = props_before * bin_heatmap

    # Find contours or connected regions
    contours, _ = cv.findContours(binary_heatmap.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Convert contours to bounding boxes
    bounding_boxes = [cv.boundingRect(contour) for contour in contours]
    
    bounding_boxes = [(x, y, h, w) for x, y, h, w in bounding_boxes if h * w >= config["min_face_area"]] # remove small boxes

    bounding_boxes = nms_based_on_overlap(bounding_boxes, iou_threshold=config["iou_threshold"])
    bounding_boxes_with_scores = []
    image = cv.imread(f"{data_path}/{image_name}")

    model_path = "../saved_models_detection/37_cnn/epoch_26_loss_0.00001.pth"
    model = FaceDetector()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # special case for close faces
    # bounding_boxes = solve_close_faces_dbscan(bounding_boxes, props_after)
    # bounding_boxes = solve_close_faces_peaks(bounding_boxes, props_after, model_bounding_boxes, image_name)

    for bbox in bounding_boxes:
        x, y, h, w = bbox
        patch = image[y:y+w, x:x+h]
        patch = patch.transpose(2, 0, 1)
        patch = torch.from_numpy(patch)
        patch = patch.float() / 255.0
        patch = torch.nn.functional.interpolate(patch.unsqueeze(0), size=(64, 64)).squeeze(0)
        patch = patch.unsqueeze(0).to(device)
        score = model(patch).item()
        if round(score, 3) == 0.0:
            continue
        bounding_boxes_with_scores.append((x, y, h, w, score))
        # cv.imshow("patch", patch.cpu().squeeze(0).numpy().transpose(1, 2, 0))
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        cv.putText(image, f"{score:.3f}", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.rectangle(image, (x, y), (x + h, y + w), (0, 0, 255), 2)

    # draw ground truth boxes
    gt_boxes = get_gt_boxes(image_name)
    # print(gt_boxes)
    for box in gt_boxes:
        x_min, y_min, x_max, y_max = box
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    if not save:
        cv.imshow("props_before", props_before)
        # cv.imshow("binary_heatmap", binary_heatmap)
        cv.imshow("props_after", props_after)
        cv.imshow(f"{image_name}", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return image, bounding_boxes_with_scores

if __name__ == "__main__":
    proposals = None
    print(os.getcwd())
    with open(f"proposals_{config['clusters']}_{config['face_threshold']}_train_{actor}.pkl", "rb") as f:
        proposals = pickle.load(f)
    # print(f"Loaded {len(proposals)} proposals")
    # print(proposals["0001.jpg"])
    # suspect_images = ["0059.jpg", "0100.jpg", "0111.jpg"]
    # suspect_images = ["0072.jpg", "0073.jpg", "0084.jpg"]
    reward = 1
    predictions_file_names = []
    predictions_bounding_boxes = []
    predictions_scores = []
    for k, v in tqdm(proposals.items(), desc="Processing proposals"):
        image_name = k
        boxes = v
        boxes = [(x, y, height, width, score) for x, y, height, width, score in boxes if height * width <= config["max_patch_area"]]
        props = np.zeros((360, 480))

        # if image_name not in suspect_images:
        #     continue

        for box in boxes:
            x, y, height, width, score = box
            # value = reward / (height * width)
            value = 1
            props[x:x+height, y:y+width] += value

        props = props / props.max()
        props = (props * 255).astype(np.uint8)

        image, bounding_boxes = detect_faces_from_heatmap(props, config["heatmap_threshold"], boxes, image_name)
        for bbox in bounding_boxes:
            x, y, h, w, score = bbox
            predictions_file_names.append(image_name)
            predictions_bounding_boxes.append((x, y, x + h, y + w))
            predictions_scores.append(score)

        # image = dbscan_clustering(boxes, image_name)
        

        if not save:
            # cv.imshow("image", image)
            # cv.imshow("proposals", props)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            steps -= 1
            if steps == 0:
                break
            
    if save:
        result_index = 0
        RESULTS_PATH = "./results"

        for path in os.listdir(f"{RESULTS_PATH}"):
            if os.path.isdir(os.path.join(f"{RESULTS_PATH}", path)):
                index = int(path)
                if index > result_index:
                    result_index = index

        result_index += 1
            
        SAVE_PATH = f"{RESULTS_PATH}/{result_index}"

        # create the directory
        os.mkdir(SAVE_PATH)
        print(f"Saving results to {SAVE_PATH}")

        with open(f"{SAVE_PATH}/config.pkl", "wb") as f:
            pickle.dump(config, f)

        with open(f"{SAVE_PATH}/detections_all_faces.npy", "wb") as f:
            np.save(f, predictions_bounding_boxes, allow_pickle=True)
            
        with open(f"{SAVE_PATH}/scores_all_faces.npy", "wb") as f:
            np.save(f, predictions_scores, allow_pickle=True)
        
        with open(f"{SAVE_PATH}/file_names_all_faces.npy", "wb") as f:
            np.save(f, predictions_file_names, allow_pickle=True)
