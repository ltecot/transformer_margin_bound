import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from datetime import datetime
import wandb
import torchvision.models as models
from torchvision.models import resnet50
from vit import ViT
import matplotlib.pyplot as plt
import math 

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

# rand labels
# target_transform=lambda y: torch.randint(0, 10, (1,)).item(),

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    config = {
        'dataset' : 'CIFAR10', # 'tiny_imagenet', # 'CIFAR100', # 'CIFAR10' 'MNIST'
        'margin_file_name' : 'cifar10_margins',
        'model_name' : 'models/net_epoch_20.pth',
        'run_path' : 'ltecot/transformer_margin/runs/2z5oeixr', # mahwlam4 mnist, 2z5oeixr cifar10
        'weight_decay' : 0,  
        # 'num_classes' : 10, # 10, 100, 200
        'batch_size' : 1,
        'epochs' : 1000,
        'lr' : 1e-4,
        # 'gamma' : 0.99,
        'print_interval' : 10,
        'save_interval' : 10,
        # 'image_size' : 28, # 32, 64, 28
        'patch_size' : 2,
        'dim' : 2048,
        'depth' : 2,
        'heads' : 1, # 9, # 12,
        'mlp_dim' : 2048,
        'dropout' : 0.1,
        'emb_dropout' : 0.1,
        'num_workers' : 4,
    }

    if config['dataset'] == 'Imagenet':
        trainset = imagenet_dataset()
    elif config['dataset'] == 'tiny_imagenet':
        trainset = tiny_imagenet_dataset()
        config['image_size'] = 32
        config['num_classes'] = 200
        config['channels'] = 3
    elif config['dataset'] == 'CIFAR10':
        trainset = cifar10_dataset()
        config['image_size'] = 28
        config['num_classes'] = 10
        config['channels'] = 3
    elif config['dataset'] == 'CIFAR100':
        trainset = cifar100_dataset()
        config['image_size'] = 28
        config['num_classes'] = 100
        config['channels'] = 3
    elif config['dataset'] == 'MNIST':
        trainset = mnist_dataset()
        config['image_size'] = 28
        config['num_classes'] = 10
        config['channels'] = 1

    net = ViT(
        image_size = config['image_size'],
        patch_size = config['patch_size'],
        num_classes = config['num_classes'],
        dim = config['dim'],
        depth = config['depth'],
        heads = config['heads'],
        mlp_dim = config['mlp_dim'],
        dropout = config['dropout'],
        emb_dropout = config['emb_dropout'],
        channels = config['channels']
    ).to(device)

    loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                         shuffle=True, num_workers=config['num_workers'])
    model_weights = wandb.restore(config['model_name'], run_path=config['run_path'])
    # net.load_weights(model_weights.name)
    net.load_state_dict(torch.load(model_weights.name))
    net.eval()
    net.update_spectral_terms()
    margins = []
    spectral_complexity = 0
    print(len(loader))
    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            label_prob = outputs.data[:, labels]  # label probs
            outputs.data[:, labels] = float("-Inf")
            next_prob, _ = torch.max(outputs.data, 1)
            m = label_prob - next_prob
            margins.append(m)
            spectral_complexity = max(net.spectral_complexity(images), spectral_complexity)
            print(str(i))
    margins = torch.cat(margins, 0)
    n = len(loader) 
    w = max(config['dim'], config['mlp_dim'], (config['image_size'] / config['patch_size'])**2)
    spectral_complexity = spectral_complexity * math.log(n) * w * math.log(w) / n
    margins /= spectral_complexity

    # ax = sns.kdeplot(margins, shade=True, color="r")
    # plt.show()

    torch.save(margins, 'margin_files/' + config['margin_file_name'] + '.pt')

if __name__ == "__main__":
    main()


# # restore the model file "model.h5" from a specific run by user "lavanyashukla"
# # in project "save_and_restore" from run "10pr4joa"
# best_model = wandb.restore('model.h5', run_path="lavanyashukla/save_and_restore/10pr4joa")

# # use the "name" attribute of the returned object if your framework expects a filename, e.g. as in Keras
# model.load_weights(best_model.name)