import os
from torchvision import datasets, transforms

class CIFAR10:
    def __init__(self, data_path='./data_cifar10'):
        """
        Initializes the CIFAR10Loader with a specified data path.
        Downloads data if it doesn't already exist in the directory.
        """
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
            self._download_data()
        else:
            print(f"Using existing data directory: {self.data_path}")

    def _download_data(self):
        """
        Downloads the CIFAR-10 dataset to the specified data path.
        """
        datasets.CIFAR10(root=self.data_path, train=True, download=True, transform=self.transform)
        datasets.CIFAR10(root=self.data_path, train=False, download=True, transform=self.transform)

    def get_data(self, train=True):
        """
        Returns the CIFAR-10 dataset (train or test set).
        """
        return datasets.CIFAR10(root=self.data_path, train=train, download=False, transform=self.transform)
