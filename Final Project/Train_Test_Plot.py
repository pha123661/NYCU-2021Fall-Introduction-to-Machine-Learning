from Hyper_parameters import HyperParams
from myDataLoader import get_loaders
from Models import CNN_Model
import numpy as np
import pytorch_lightning as pl

def main():
    # CNN
    CNN_model = CNN_Model()
    trainer = pl.Trainer(gpus=1)
    train_loader, test_loader = get_loaders()
    trainer.fit(CNN_model, train_loader, test_loader)


if __name__ == '__main__':
    main()