# preprocessing the data
from PIL import Image
from torchvision.transforms import functional as F
from io import BytesIO

img = Image.open("C:/Users/Bene/PycharmProjects/StyleGAN/dataset_corgis/corgis_25k/5h8XwMyMdr.jpeg")
img.show()

path = "C:/Users/Bene/PycharmProjects/StyleGAN/dataset_corgis/resized"

# we need multiple sizes so we write to memory first before converting to jpgs all at once
# https://stackoverflow.com/questions/646286/how-to-write-png-image-to-string-with-the-pil
def resize(image, size, quality = 100):
    img = F.resize(image, size)
    with BytesIO() as buffer:
        img.save(buffer, format = "jpeg", quality = quality)
        contents = buffer.getvalue()
    return contents

