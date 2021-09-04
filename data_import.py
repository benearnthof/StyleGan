from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset

# map-style dataset for DataLoader:
# https://github.com/rosinality/style-based-gan-pytorch/blob/master/dataset.py

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution):
        self.env = lmdb.open(path, max_readers = 32, readonly = True,
                             lock = False, readahead = False, meminit = False)
        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write = False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    # map-style datasets need to implement __getitem__() and __len__()
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write = False) as txn:
            # compare to line 58 in dataset.py
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        with BytesIO() as buffer:
            img = Image.open(buffer)
            img = self.transform(img)

        return img
