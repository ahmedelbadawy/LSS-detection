import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import torch
import torch.nn as nn

def sample_class_weights(train_data):

    class_counts = train_data.Label.value_counts()

    class_weights = 1 / class_counts

    sample_weights = np.array([class_weights[i] for i in train_data.Label.values])

    loss_weights = (sum(class_counts) / class_counts).sort_index().values

    return loss_weights, sample_weights

def create_transform(method, size):
    # these methods are from "Medical image data augmentation: techniques, comparisons and interpretations" paper
    # print(size,"##################################################")
    transform_test = A.Compose([
    A.Resize(size, size),
    A.Normalize(mean= [0.305238, 0.305238, 0.305238], std = [0.20949449, 0.20949449, 0.20949449]),
    ToTensorV2(),
    ])
    if method == "method10":

        transform_train = A.Compose([
        A.Resize(size, size),
        A.Affine(translate_px = {'x': (-15, 15) , 'y':(-15, 15)}, shear = {'x': (-15, 15) , 'y':(-15, 15)}, rotate = (-25, 25)) ,
        A.Normalize(mean= [0.305238, 0.305238, 0.305238], std = [0.20949449, 0.20949449, 0.20949449]),
        ToTensorV2(),
        ])
    elif method == "method9":

        transform_train = A.Compose([
        A.Resize(size, size),
        A.Affine(translate_px = {'x': (-15, 15) , 'y':(-15, 15)}, shear = {'x': (-15, 15) , 'y':(-15, 15)}) ,
        A.Normalize(mean= [0.305238, 0.305238, 0.305238], std = [0.20949449, 0.20949449, 0.20949449]),
        ToTensorV2(),
        ])

    elif method == "method8":

        transform_train = A.Compose([
        A.Resize(size, size),
        A.Affine(translate_px = {'x': (-15, 15) , 'y':(-15, 15)}, rotate = (-25, 25)) ,
        A.Normalize(mean= [0.305238, 0.305238, 0.305238], std = [0.20949449, 0.20949449, 0.20949449]),
        ToTensorV2(),
        ])


    elif method == "method7":

        transform_train = A.Compose([
        A.Resize(size, size),
        A.GaussNoise(var_limit = 0.03) ,
        A.Affine(rotate = (-25, 25)),
        A.Normalize(mean= [0.305238, 0.305238, 0.305238], std = [0.20949449, 0.20949449, 0.20949449]),
        ToTensorV2(),
        ])
    return transform_train, transform_test

def init_model(model_name , pretrained):
    if model_name == "convnext":
        if pretrained:
            model = torchvision.models.convnext_tiny(weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            # model.classifier[2] = torch.nn.Linear( in_features=1024, out_features=4, bias=True)
        else:
            model = torchvision.models.convnext_tiny()
            # model.features[0][0] = torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
            # model.classifier[2] = torch.nn.Linear( in_features=1024, out_features=4, bias=True)
        model.classifier[2] = nn.Sequential(
        nn.Linear(768, 256),  # Additional linear layer with 256 output features
        nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
        nn.Dropout(0.5),               # Dropout layer with 50% probability
        nn.Linear(256, 4)    # Final prediction fc layer
                                        )

    elif model_name == "efficientnet":
        if pretrained:
            model = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            # model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=4, bias=True)
        else:
            model = torchvision.models.efficientnet_v2_s()
            # model.features[0][0] = torch.nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=4, bias=True)
        model.classifier[1] = nn.Sequential(
        nn.Linear(1280, 256),  # Additional linear layer with 256 output features
        nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
        nn.Dropout(0.5),               # Dropout layer with 50% probability
        nn.Linear(256, 4)    # Final prediction fc layer
    )

    elif model_name == "regnet_y_16gf":
        if pretrained:
            model = torchvision.models.regnet_y_400mf(weights = torchvision.models.RegNet_Y_400MF_Weights.IMAGENET1K_V1)
            # model.fc = torch.nn.Linear(in_features=3024, out_features=4, bias=True)
        else:
            model = torchvision.models.regnet_y_400mf()
            # model.stem[0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # model.fc = torch.nn.Linear(in_features=3024, out_features=4, bias=True)
        model.fc = nn.Sequential(
        nn.Linear(440, 256),  # Additional linear layer with 256 output features
        nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
        nn.Dropout(0.5),               # Dropout layer with 50% probability
        nn.Linear(256, 4)    # Final prediction fc layer
    )

    elif model_name == "resnet":
        if pretrained:
            model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            # model.fc = torch.nn.Linear(in_features=2048, out_features=4, bias=True)
        else:
            model = torchvision.models.resnet50()
            # model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # model.fc = torch.nn.Linear(in_features=2048, out_features=4, bias=True)
        
        
        model.fc = nn.Sequential(
            nn.Linear(2048, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 4)    # Final prediction fc layer
        )

    return model


def init_scheduler(optimizer, scheduler_flag, lr_min, plateau_factor, plateau_patience, cosine_t0, cosine_t_mult):
    if scheduler_flag == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = cosine_t0, T_mult = cosine_t_mult, eta_min= lr_min)

    elif scheduler_flag == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=plateau_factor, patience=plateau_patience, threshold=lr_min)
    return scheduler

def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)