import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import mobilenet_v2


import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import numpy as np
import wandb
import os

from dataset import FaceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = {
    "batch_size": 4,
    "num_epochs": 30,
    "learning_rate": 0.0001,
    "patience": 3
}

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs):
    wandb.init(project="facial-recognition", config=hparams)
    BASE_PATH = "../saved_models_recognition"

    save_index = 0
    for path in os.listdir(f"{BASE_PATH}"):
        if os.path.isdir(os.path.join(f"{BASE_PATH}", path)):
            index = int(path)
            if index > save_index:
                save_index = index

    save_index += 1
        
    SAVE_PATH = f"{BASE_PATH}/{save_index}"

    # create the directory
    os.mkdir(SAVE_PATH)
    print(f"Saving model to {SAVE_PATH}")

    model = model.to(device)
    print(f"Using {device} for training")
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
        train_loss = running_loss / len(train_dataloader)

        model.eval()
        running_val_loss = 0.0
        for images, targets in tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            running_val_loss += losses.item()
        val_loss = running_val_loss / len(val_dataloader)

        scheduler.step(val_loss)

        if val_loss < best_loss and epoch + 1 >= 5:
            best_loss = val_loss
            print(f"Saved model with validation loss {val_loss:.5f}")
            model_name = f"model_{epoch + 1}_{val_loss:.5f}.pth"
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, model_name))

        print(f"Epoch {epoch + 1}/{num_epochs} train_loss: {train_loss:.5f} val_loss: {val_loss:.5f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})


if __name__ == "__main__":
    original_dataset = FaceDataset(images_path="../train_images", labels_path="../train_positive", dataset_type="train")
    val_dataset = FaceDataset(images_path="../val_images", labels_path="../val_positive", dataset_type="val")

    train_dataloader = DataLoader(original_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn)

    num_classes = 5

    backbone = mobilenet_v2(weights=None).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator)
    
    # backbone = resnet_fpn_backbone('resnet18', weights=None)
    # model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    optimizer = Adam(model.parameters(), lr=hparams["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, patience=hparams["patience"], verbose=True)
    criterion = CrossEntropyLoss()

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, hparams["num_epochs"])



