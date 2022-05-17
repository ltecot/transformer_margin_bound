import torch
import random
import torchvision
from torchvision import transforms, datasets

def tiny_imagenet_dataset():
    subdir_train = 'datasets/tiny-imagenet-200/train'
    subdir_test = 'datasets/tiny-imagenet-200/val/images'
    transform = transforms.Compose([
        # transforms.Resize(64),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(subdir_train, transform=transform)
    return train_dataset

def imagenet_dataset():
    subdir = "../rand_smoothing_indep_vars/datasets/imagenet"  # CUSTOM: Change filepath
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(subdir, transform)
    return dataset

def cifar10_dataset():
    train_transform = transforms.Compose(
        [transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    return trainset

def cifar100_dataset():
    train_transform = transforms.Compose(
        [transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=train_transform)
    return trainset

def mnist_dataset():
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=train_transform)
    return trainset

# Randomize labels (deterministically from set seed for reproducability between testing)
class RandomizedDataset(torch.utils.data.Dataset):
  def __init__(self, dataset):
    super(RandomizedDataset, self).__init__()
    self.orig_data = dataset
    num_classes = len(dataset.classes)
    random.seed(42)
    self.rand_map = [random.randrange(0, num_classes) for _ in range(dataset.__len__())]
    random.seed()

  def __getitem__(self, index):
    x, _ = self.orig_data[index]  # get the original item
    y = self.rand_map[index]
    return x, y

  def __len__(self):
    return self.orig_data.__len__()