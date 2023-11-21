import os
import sys
sys.path.append("./")

import numpy as np
import torch
from datargs import parse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.datamodules import CIFAR10DataModule
import torchvision

from rapp.args import Args
from rapp.model import OSP_Model
from rapp.models.resnet import ResNetDiscriminator, ResNetGenerator
from rapp.models.model32 import Discriminator32, Generator32



def main():
    # Configs
    args: Args = parse(Args)
    args.setup()

    # Logging
    if args.log_checkpoint_every_n:
        checkpoint_callbacks = [ModelCheckpoint(
            dirpath=args.out_dir,
            filename='checkpoint_{epoch:02d}',
            period=args.log_checkpoint_every_n)]
    else:
        checkpoint_callbacks = []
    
    wandb_logger = WandbLogger(
        tags=args.wandb_tags,
        project=args.wandb_project, 
        entity=args.wandb_entity, 
        name=args.wandb_name, 
        id=args.wandb_id,
        config=args,
    )

    # Folder and seed
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Log arguments for easy reexecution
    print("Script command:")
    print(" ".join(sys.argv))

    # Dataloader
    if args.dataset == 'cifar10':
        # Change normalization to make eval with torch-fidelity accurate:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        dm = CIFAR10DataModule(
            args.data_root, 
            num_workers=args.workers,
            normalize=True,
            shuffle=True,
            batch_size=args.batch_size,
            train_transforms=transform,
            val_transforms=transform,
            test_transforms=transform,
        )

    # Define the model
    if args.dataset in ['cifar10']:
        if args.model == 'resnet':
            netD = ResNetDiscriminator(spectral_norm=args.spectral_norm, size=args.ndf*2)
            netG = ResNetGenerator(z_dim=args.nz, size=args.ngf*2)
        else:
            netG = Generator32(z_dim=args.nz, h_filters=args.ngf)
            netD = Discriminator32(spectral_norm=args.spectral_norm, h_filters=args.ndf)

    # Pytorch lightning Trainer
    model = OSP_Model(netG, netD, args)       
    trainer = Trainer(
        logger=wandb_logger, 
        gpus=args.gpus, 
        max_epochs=args.epochs, 
        # progress_bar_refresh_rate=5, 
        callbacks=checkpoint_callbacks,
    )
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
