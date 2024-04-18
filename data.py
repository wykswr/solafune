import tifffile
from torch.utils.data import Dataset, DataLoader
import torch
import lightning as L
from datasets import load_dataset, Image, ClassLabel
import torchvision.transforms as T
from pathlib import Path



def load_tiff(path):
    img = tifffile.imread(path)
    return img[..., [1, 2, 3, 8, 10, 11]]

img_norm_cfg = dict(
    mean=[0.0935, 0.1196, 0.1377, 0.2674, 0.2361, 0.1804],
    std=[0.0931, 0.0967, 0.1217, 0.1180, 0.1453, 0.1394],
)  # change the mean and std of all the bands


class TiffDataset(Dataset):
    def __init__(self, dataset, aug=False):
        self.dataset = dataset
        if aug:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(**img_norm_cfg),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
                T.Resize(224, antialias=True),
        ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(**img_norm_cfg),
                T.Resize(224, antialias=True),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = load_tiff(self.dataset[idx]['image']['path'])
        label = self.dataset[idx]['label']
        img = self.transform(img).unsqueeze(1)
        return img, torch.tensor(label).long()
    
class InferSet(Dataset):
    def __init__(self, img_folder, aug=True):
        self.with_aug = aug
        img_folder = Path(img_folder)
        # all tiif files in the folder
        imgs = list(img_folder.glob("*.tif"))
        ids = [int(img.stem.split('_')[1]) for img in imgs]
        self.imgs = [img for _, img in sorted(zip(ids, imgs))]
        if aug:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(**img_norm_cfg),
                T.Resize(224, antialias=True),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = load_tiff(self.imgs[idx])
        if self.with_aug:
            img = self.transform(img).unsqueeze(1)
        else:
            img = self.transform(img)
        return img

    
def collate(batch) -> dict[str, torch.Tensor]:
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return {
        'image': imgs,
        'label': labels
    }

def infer_collate(batch) -> dict[str, torch.Tensor]:
    imgs = batch
    imgs = torch.stack(imgs)
    return {
        'image': imgs,
    }

class TiffDataModule(L.LightningDataModule):
    def __init__(self, data_folder, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        if stage == "fit":
            dataset = load_dataset('imagefolder', data_dir=self.data_folder, split="train").cast_column("image", Image(decode=False))
            class_label = ClassLabel(names=['no', 'yes'])
            features = dataset.features
            features['label'] = class_label
            dataset = dataset.cast(features)
            splitted = dataset.train_test_split(test_size=0.2, seed=22, stratify_by_column='label')
            self.train_dataset = TiffDataset(splitted['train'], aug=True)
            self.val_dataset = TiffDataset(splitted['test'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=collate,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=collate,
                          shuffle=False)
