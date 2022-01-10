
import numpy as np
import torch

import matplotlib.pyplot as plt

from cse291.HW0.model import UNET
from cse291.HW0.pipline import (
    load_checkpoint,
    save_predictions_as_imgs,
)

try:
    from icecream import install  # noqa
    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def predict(model, test_images, save_dir, device):
    # model.eval()
    
    plt.figure(figsize=(10, 10))
    
    x = np.moveaxis(test_images, -1, 1)
    x = torch.from_numpy(x).float().to(device)
    preds = torch.sigmoid(model(x))
    preds = (preds > 0.5).squeeze(1).float().cpu().numpy()
    
    for i in range(len(test_images)):
        plt.subplot(4, 2, i * 2 + 1)
        plt.imshow(test_images[i])
        plt.subplot(4, 2, i * 2 + 2)
        plt.imshow(preds[i])
    plt.savefig(save_dir)
    

if __name__ == '__main__':
    test_data = np.load("./data/test.npz")
    test_images = test_data["images"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET(in_channels=3, out_channels=1).to(device)
    load_checkpoint(torch.load("my_checkpoint.pth"), model)
    predict(model, test_images, "test_result.png", device)
    
    

