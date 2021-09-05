from torchvision import transforms

# map-style dataset for DataLoader:
# https://github.com/rosinality/style-based-gan-pytorch/blob/master/dataset.py

from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(path, max_readers=32, readonly=True, lock=False,
                             readahead=False, meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(6)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


out_path = "C:/Users/Bene/PycharmProjects/StyleGAN/lmdb_corgis/"
test = MultiResolutionDataset(out_path, transform= transforms.RandomHorizontalFlip(), resolution=128)
test[0].show()

# everything seems to work so far, lets look at the dataloader