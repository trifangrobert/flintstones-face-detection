import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
import wandb

from torchvision.models import resnet18, resnet34, resnet50, inception_v3

from model_resize_3 import FaceDetector
from build_dataset import FacialDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = {
    "batch_size": 8,
    "num_epochs": 20,
    "learning_rate": 0.0001,
    "resize_shape": (64, 64),
    "patience": 3
}


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs):
    wandb.init(project="facial-detection", config=hparams)
    BASE_PATH = "../saved_models_detection"

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
           
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)

        # validate the model
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in tqdm(val_dataloader, desc="Validating"):
                # Move data to the appropriate device (GPU or CPU)
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

            val_loss /= len(val_dataloader)
            if epoch + 1 >= 5 and (val_loss < best_loss or val_loss < 0.0001):
                best_loss = epoch_loss
                model_name = f"epoch_{epoch + 1}_loss_{val_loss:.5f}.pth"
                torch.save(model.state_dict(), f"{SAVE_PATH}/{model_name}")

            scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, training loss: {epoch_loss:.6f} validation loss: {val_loss:.6f}')
        wandb.log({"training_loss": epoch_loss, "validation_loss": val_loss})

    print('Training complete')  


if __name__ == "__main__":
    model = FaceDetector()
    # model = resnet18(weights=None)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 1),
    #     nn.Sigmoid()
    # )

    train_positive = "../train_positive"
    train_negative = "../train_negative"
    train_images = "../train_images"

    val_positive = "../val_positive"
    val_negative = "../val_negative"
    val_images = "../val_images"

    original_dataset = FacialDataset(images_path=train_images, positive_path=train_positive, negative_path=train_negative, dataset_type="train")
    horizontal_flip_dataset = FacialDataset(images_path=train_images, positive_path=train_positive, negative_path=train_negative, dataset_type="train", transform_type="horizontal_flip")
    gaussian_noise_dataset = FacialDataset(images_path=train_images, positive_path=train_positive, negative_path=train_negative, dataset_type="train", transform_type="gaussian_noise")
    train_dataset = ConcatDataset([original_dataset, horizontal_flip_dataset, gaussian_noise_dataset])
    train_dataloader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=4)

    val_dataset = FacialDataset(images_path=val_images, positive_path=val_positive, negative_path=val_negative, dataset_type="validation")
    val_dataloader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers=4)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=hparams["patience"], verbose=True)

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, hparams["num_epochs"])
