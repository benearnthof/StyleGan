# preprocessing the data
from PIL import Image
from torchvision.transforms import functional as F
from io import BytesIO
import lmdb
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# img_path = "C:/Users/Bene/PycharmProjects/StyleGAN/dataset_corgis/corgis_25k/5h8XwMyMdr.jpeg"

dataset_path = "D:/Downloads/crop1024/"
out_path = "C:/Users/Bene/PycharmProjects/StyleGAN/lmdb_corgis/"


# function to convert pixel number inputs to their corresponding powers of 2
def power(x):
    assert x % 2 == 0
    if x <= 2:
        return 1
    return 1 + power(x / 2)


# we need multiple sizes so we write to memory first before converting to jpgs
# all at once
# https://stackoverflow.com/questions/646286/how-to-write-png-image-to-string-with-the-pil
def resize(img_path, max_size=128, min_size=8, quality=100):
    image = Image.open(img_path)
    # range(x) starts at 0 and ends at x-1
    sizes = list(map(lambda x: 2 ** x, range(power(max_size) + 1)))
    sizes = [x for x in sizes if x >= min_size]
    ret = []

    for size in sizes:
        img = F.resize(image, size)
        with BytesIO() as buffer:
            img.save(buffer, format="jpeg", quality=quality)
            contents = buffer.getvalue()
        ret.append(contents)

    return sizes, ret


# sz, bts = resize(img_path, max_size=128)
# test = BytesIO(bts[0])
# test = Image.open(test)
# test.show()

# number of datapoints: 25000

def to_lmdb(transaction, dataset, max_size=128, min_size=8):
    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    for i in tqdm(range(len(files))):
        sizes, res = resize(files[i][1], max_size, min_size)
        for size, img in zip(sizes, res):
            key = f'{size}-{str(i).zfill(6)}'.encode('utf-8')
            transaction.put(key, img)

        total = total + 1

    transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))


dataset = ImageFolder(root=dataset_path)
maxsize = 128
minsize = 8
# how to adjust the needed size of the database dynamically?
# trial and error: starting off with 25k images and a map_size of
# 128 ** 4 * 3 leads to 24958 elements being written.
# for all 136255 images we need approx:
# 128 ** 4 * (137000 / (24950/3)) = 128 ** 4 ** 17

mapsize = (maxsize ** 4) * 17 # black magic
# the final database will require about 4.5 GB of disk space

with lmdb.open(path=out_path, map_size=mapsize, readahead=False) as env:
    with env.begin(write=True) as txn:
        to_lmdb(txn, dataset, maxsize, minsize)
