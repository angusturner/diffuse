import numpy as np
from torchvision import transforms, datasets


class MNIST:
    mean = 0.1307
    std = 0.3081
    default_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((mean,), (std,))
        ]
    )

    def __init__(self, data_dir, train: bool = True):
        """
        Return continuous MNIST digits, scaled in [-1, 1]
        :param data_dir: location to save
        :param train:
        """
        self.dataset = datasets.MNIST(data_dir, train=train, transform=MNIST.default_transforms, download=True)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def denormalize(im):
        """
        :param im: (B, C, H, W)
        """
        return (im + 1) / 2.0
        # return (im * MNIST.std) + MNIST.mean

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = np.array(x).astype(np.float32)
        x = (x * 2.0) - 1

        return x, y
