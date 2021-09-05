from data_import import MultiResolutionDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

out_path = "C:/Users/Bene/PycharmProjects/StyleGAN/lmdb_corgis/"

# best way of adding data augmentation
# through the data loader
transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # data loader needs tensors, arrays etc.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

dataset = MultiResolutionDataset(out_path, transform=transform, resolution=128)
# F.to_pil_image(dataset[0]).show()

def dataloader(dataset, batch_size, resolution = 128):
    dataset.resolution = resolution
    loader = DataLoader(dataset, shuffle = True, batch_size = batch_size,
                        num_workers=0, drop_last=True)
    # num workers set to 0 for running on windows 10
    # https://github.com/pytorch/examples/issues/526#issuecomment-605450664
    data_loader = iter(loader)
    return data_loader

loader = dataloader(dataset, 1, 128)

def getimg(data_loader):
    img = next(data_loader)
    img = F.to_pil_image(img.squeeze(0))
    return img

# getimg(loader).show()
