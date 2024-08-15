from losses import init_loss
from engine import train_step, test_step
from data_setup import CustomDataset
import pandas as pd
import numpy as np
from utilies import sample_class_weights
import torch
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from utilies import sample_class_weights, create_transform, init_model, init_scheduler, initialize_weights
from tqdm.auto import tqdm

# Setup hyperparameters
NUM_EPOCHS = 100
LR_MIN = 1e-6


hyperparameter_defaults = dict(
    batch_size = 16,
    lr = 0.001,
    optimizer = "Adam",
    loss_fn = "CCE",
    augmentation = "method10",
    model = "resnet",
    image_size = 260,
    focalLoss_gamma = 2,
    scheduler = "cosine",
    scheduler_factor = 0.1,
    scheduler_patience = 5,
    cosine_T_mult = 1,
    pretrained = False,
    imbalance = "weighted_random",
    clip_norm = 2,
    weight_decay = 1e-2,
    l1_param = 1e-2
    )

wandb.init(project="Classification_stenosis", entity="elbadawy990-sejong-university", job_type="training", config=hyperparameter_defaults)
config = wandb.config


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Augmentations
transform_train, transform_test = create_transform(config.augmentation, config.image_size)


# Importing data and setting up data loaders
data = pd.read_csv("dataset.csv")

train_data = data[data["Split"] == "train"].reset_index(drop=True)
test_data = data[data["Split"] == "test"].reset_index(drop=True)

loss_weights, sample_weights = sample_class_weights(train_data)


sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data))

if config.imbalance == "weighted_random":
    loss_weights = None
else:
    sampler = None

 
train_data = CustomDataset(train_data, transform_train, config.pretrained)

test_data = CustomDataset(test_data, transform_test, config.pretrained)

train_loader = DataLoader(train_data,batch_size= config.batch_size, sampler=sampler)

test_loader = DataLoader(test_data, batch_size= config.batch_size)

# Model

model = init_model(config.model , config.pretrained)

if not config.pretrained:

    model.apply(initialize_weights)


wandb.watch(model)

optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), config.lr, weight_decay = config.weight_decay)



scheduler = init_scheduler(optimizer=optimizer, scheduler_flag=config.scheduler, lr_min=LR_MIN, plateau_factor=config.scheduler_factor, plateau_patience=config.scheduler_patience, 
                           cosine_t0=config.scheduler_patience, cosine_t_mult=config.cosine_T_mult)


scaler = torch.cuda.amp.GradScaler()

criterion = init_loss(loss_fn = config.loss_fn, focalLoss_gamma=config.focalLoss_gamma , loss_weights=loss_weights, device=device)


########################### Train #################################

# Create empty results dictionary
results = {"train_loss": 0,
            "train_acc": 0,
            "test_loss": 0,
            "test_acc":  0
}

# Make sure model on target device
model.to(device)

# Loop through training and testing steps for a number of epochs
for epoch in tqdm(range(NUM_EPOCHS)):
    train_loss, train_acc = train_step(model=model,
                                        dataloader=train_loader,
                                        loss_fn=criterion,
                                        optimizer=optimizer,
                                        scaler=scaler,
                                        max_norm=config.clip_norm,
                                        l1_param = config.l1_param,
                                        device=device)
    test_loss, test_acc = test_step(model=model,
        dataloader=test_loader,
        loss_fn=criterion,
        device=device)
    
    scheduler.step(test_loss)


    # Print out what's happening
    # print(
    #     f"Epoch: {epoch+1} | "
    #     f"train_loss: {train_loss:.4f} | "
    #     f"train_acc: {train_acc:.4f} | "
    #     f"test_loss: {test_loss:.4f} | "
    #     f"test_acc: {test_acc:.4f}"
    # )

    # Update results dictionary
    results["train_loss"] = train_loss
    results["train_acc"] = train_acc
    results["test_loss"] = test_loss
    results["test_acc"] = test_acc

    wandb.log(results)

