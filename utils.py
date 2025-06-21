
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

class SaliencyDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.imgs[idx].replace('.jpg', '.png'))
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        return self.transform(img), self.transform(mask)
