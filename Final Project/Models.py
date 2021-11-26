from Hyper_parameters import HyperParams
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)


class CNN_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.extractor = nn.Sequential(
            # 1 128 128
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 128 128
            nn.MaxPool2d(kernel_size=2),
            # 64 64 64

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 64 64
            nn.MaxPool2d(kernel_size=2),
            # 128 32 32

            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 256 32 32
            nn.MaxPool2d(kernel_size=4),
            # 256 8 8

            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 512 8 8
            nn.MaxPool2d(kernel_size=4)
            # 512 2 2
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=len(HyperParams.genres))
        )

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        features = self.extractor(x)
        features = features.reshape((features.shape[0], -1))
        ret = self.classifier(features)

        # get label
        index = np.argmax(ret.cpu().numpy())
        label = HyperParams.genres[index]
        return label

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, dim=1)
        features = self.extractor(x)
        features = features.reshape((features.shape[0], -1))
        ret = self.classifier(features)
        loss = self.loss_function(ret, y)
        self.log(loss.mean().item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, dim=1)
        features = self.extractor(x)
        features = features.reshape((features.shape[0], -1))
        ret = self.classifier(features)

        ret = ret.max(1)[1]
        acc = (ret == y).float().mean()
        self.log("acc", acc)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=HyperParams.learning_rate,)
