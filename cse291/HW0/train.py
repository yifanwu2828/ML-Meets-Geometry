import argparse
import pathlib

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.optim as optim

from tqdm import tqdm

from cse291.commons.utils import set_random_seed
import cse291.commons.pytorch_util as ptu
from cse291.HW0.Unet_model import UNET
from cse291.HW0.pipline import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

try:
    from icecream import install  # noqa
    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 320  


def train(dataloader, model, optimizer, loss_fn, scaler, device):
    pbar = tqdm(dataloader)
    
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        targets = targets.float().unsqueeze(1).to(device)
        
        with torch.cuda.amp.autocast():
            preds = model(data)
            loss = loss_fn(preds, targets)
        
        # In general  will have lower memory footprint, and can modestly improve performance.
        # However, it changes certain behaviors.
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_postfix(loss=loss.item())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_epoch", "-n", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    
    device = ptu.init_gpu(use_gpu=args.cuda, gpu_id=0, verbose=True)
    set_random_seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.benchmark = True
    
    file_dir = pathlib.Path(__file__).resolve().parent
    data_dir = file_dir.joinpath("data")

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    valid_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    train_loader, valid_loader = get_loaders(
        train_dir= data_dir / "train.npz",
        valid_dir= data_dir / "train.npz",
        batch_size=args.batch_size,
        train_transform=train_transform,
        valid_transform=valid_transform,
        
    )
    if args.test:
        load_checkpoint(torch.load("my_checkpoint.pth"), model)
        check_accuracy(valid_loader, model, device=device)
    

    for epoch in range(args.num_epoch):
        print(f"iteration: {epoch+1}")
        train(train_loader, model, optimizer, loss_fn, scaler, device)
        check_accuracy(valid_loader, model, device=device)
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            filename="my_checkpoint.pth",
        )
        
        save_predictions_as_imgs(
            valid_loader, model, folder="saved_images/", device=device
        )
    

if __name__ == '__main__':
    main()