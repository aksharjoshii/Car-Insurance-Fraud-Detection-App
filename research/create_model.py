from torchvision import models 
import torch.optim as optim
import torch.nn as nn


def get_model_optimizer(device:str,
                        config):
    
    weights = models.EfficientNet_B0_Weights.DEFAULT
    efnet_b0 = models.efficientnet_b0(weights=weights, progress=True)
    for param in efnet_b0.parameters():
        param.requires_grad = False
        efnet_b0.classifier= nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1280, out_features= 2, bias=True)  
         ).to(device)
        optimizer = optim.Adam(params=efnet_b0.parameters(), lr=config.lr)

        return efnet_b0, optimizer
    