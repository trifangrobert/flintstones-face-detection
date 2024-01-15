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

from torchvision.models import resnet18, resnet34, resnet50

from model import FaceClassifier
from dataset import FacialDataset
from model_resnet import ResNet18, ResNet34
from model_resnet_big import ResNet50

hparams = {
    "batch_size": 4,
    "num_epochs": 30,
    "learning_rate": 0.00001,
    "patience": 3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs):
    wandb.init(project="facial-classifier", config=hparams)
    BASE_PATH = "../saved_models_classification"

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
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Move data to the appropriate device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        running_loss = 0.0
        for inputs, labels in tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            
        epoch_val_loss = running_loss / len(val_dataloader)
        scheduler.step(epoch_val_loss)

        if epoch + 1 >= 5 and epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            model_name = f"model_{epoch + 1}_{epoch_val_loss:.5f}.pth"
            torch.save(model.state_dict(), f"{SAVE_PATH}/{model_name}")
            print(f"Model saved at {SAVE_PATH}/model.pth")

        print(f"Epoch {epoch + 1}/{num_epochs} train_loss: {epoch_loss:.5f} val_loss: {epoch_val_loss:.5f}")
        wandb.log({"epoch_loss": epoch_loss, "epoch_val_loss": epoch_val_loss})


if __name__ == "__main__":
    # load the dataset
    resize_shape = (224, 224)

    original_dataset = FacialDataset(images_path="../train_images", labels_path="../train_positive", dataset_type="train", patch_shape=resize_shape)
    horizontal_flip_dataset = FacialDataset(images_path="../train_images", labels_path="../train_positive", dataset_type="train", transform_type="horizontal_flip", patch_shape=resize_shape)
    train_dataset = ConcatDataset([original_dataset, horizontal_flip_dataset])
    val_dataset = FacialDataset(images_path="../val_images", labels_path="../val_positive", dataset_type="validation", patch_shape=resize_shape)

    # create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

    # create the model
    # model = FaceClassifier(num_classes=5, patch_shape=resize_shape)
    # model = ResNet50(num_classes=5)
    model = resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    num_classes = 5
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=hparams["learning_rate"])  
    scheduler = ReduceLROnPlateau(optimizer, patience=hparams["patience"], verbose=True)

    # train the model
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, hparams["num_epochs"])