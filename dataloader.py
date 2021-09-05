from data_import import MultiResolutionDataset
from torchvision.transforms import transforms

out_path = "C:/Users/Bene/PycharmProjects/StyleGAN/lmdb_corgis/"
test = MultiResolutionDataset(out_path, transform= transforms.RandomHorizontalFlip(), resolution=128)
test[0].show()