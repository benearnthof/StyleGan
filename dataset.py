# preprocessing the data
from PIL import Image
from torchvision.transforms import functional as F
from io import BytesIO

img = Image.open("C:/Users/Bene/PycharmProjects/StyleGAN/dataset_corgis/corgis_25k/5h8XwMyMdr.jpeg")
img.show()

path = "C:/Users/Bene/PycharmProjects/StyleGAN/dataset_corgis/resized"


# function to convert pixel number inputs to their corresponding powers of 2
def power(x):
    assert x % 2 == 0
    if x <= 2:
        return 1
    return 1 + power(x / 2)


# we need multiple sizes so we write to memory first before converting to jpgs all at once
# https://stackoverflow.com/questions/646286/how-to-write-png-image-to-string-with-the-pil
def resize(image, max_size=128, min_size=8, quality=100):
    sizes = list(map(lambda x: 2 ** x, range(power(max_size) + 1)))  # range starts at 0
    sizes = [x for x in sizes if x >= min_size]
    ret = []

    for size in sizes:
        img = F.resize(image, size)
        with BytesIO() as buffer:
            img.save(buffer, format="jpeg", quality=quality)
            contents = buffer.getvalue()
        ret.append(contents)

    return ret


bts = resize(img, max_size=128)
test = BytesIO(bts[0])
test = Image.open(test)
test.show()
