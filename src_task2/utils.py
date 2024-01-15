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
    

import torch
def horizontal_flip(tensor):
    return torch.flip(tensor, dims=[2])
    
def gaussian_noise(tensor, mean=0, std=0.1):
    noise = torch.randn(tensor.shape) * std + mean
    return tensor + noise