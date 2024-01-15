def sliding_window(image, patch_size, step_size=1):
    # image should be in the format (channels, height, width)
    # print("From sliding window")
    # print("Image shape: ", image.shape)
    # print("Patch size: ", patch_size)
    # print(f"height {len(range(0, image.shape[1] - patch_size[0], step_size))}")
    # print(f"width {len(range(0, image.shape[2] - patch_size[1], step_size))}")
    for x in range(0, image.shape[1] - patch_size[0], step_size):
        for y in range(0, image.shape[2] - patch_size[1], step_size):
            yield (x, y, image[:, x:x + patch_size[0], y:y + patch_size[1]])

def calculate_intersection(box1, box2):
    x_left = max(box1[0], box2[0])
    x_right = min(box1[2], box2[2])
    y_top = max(box1[1], box2[1])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    return intersection_area

def calculate_union(box1, box2):
    intersection_area = calculate_intersection(box1, box2)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - intersection_area

    return union_area

def calculate_iou(box1, box2):
    intersection_area = calculate_intersection(box1, box2)
    union_area = calculate_union(box1, box2)
    
    iou = intersection_area / union_area
    return iou
    
def nms_based_on_overlap(boxes, iou_threshold):
        # Convert to [(x_min, y_min, x_max, y_max), ...] format
        boxes = [(x, y, x + height, y + width) for x, y, height, width in boxes]
        boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
        selected_boxes = []
        while boxes:
            current_box = boxes.pop(0)
            boxes = [box for box in boxes if calculate_iou(current_box, box) < iou_threshold]
            selected_boxes.append(current_box)
        
        selected_boxes = [(x, y, x_max - x, y_max - y) for x, y, x_max, y_max in selected_boxes]
        return selected_boxes

