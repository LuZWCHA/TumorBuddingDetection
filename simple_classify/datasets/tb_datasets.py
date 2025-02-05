import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize

size = (512, 512)

normalize = Normalize(mean=0, std=1)
_transforms = Compose([Resize(size), ToTensor(), normalize])
# _transforms_val = Compose([Resize(size), ToTensor(), normalize])

tb_cls2_dataset_train = dset.ImageFolder(root="data/train", transform=_transforms)
tb_cls2_dataset_val = dset.ImageFolder(root="data/val", transform=_transforms)
