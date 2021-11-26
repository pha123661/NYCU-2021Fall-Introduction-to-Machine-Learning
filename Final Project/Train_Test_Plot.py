from myDataLoader import get_loaders
from Models import CNN_Model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    # CNN
    CNN_model = CNN_Model()
    train_loader, test_loader = get_loaders()
    checkpoint_callback = ModelCheckpoint(
        monitor='acc',
        dirpath='./model_weights',
        filename='CNN_{epoch:02d}_{acc:.2f}',
        save_top_k=2,
        mode='max',
    )
    trainer = pl.Trainer(gpus=1, callbacks=[checkpoint_callback])
    trainer.fit(CNN_model, train_loader, test_loader)


if __name__ == '__main__':
    main()