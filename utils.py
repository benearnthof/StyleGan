import torchvision.transforms.functional as visF
def display_tensor(x):
    img = visF.to_pil_image(x.squeeze(0))
    img.show()