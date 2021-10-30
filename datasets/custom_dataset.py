from torch.utils.data import Dataset
import os
from PIL import Image
from natsort import natsorted

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        all_imgs = natsorted(all_imgs)
        print(f"total images to be process: {len(all_imgs)} images")
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        fname = self.total_imgs[idx]
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        img = self.transform(image)
        return img
