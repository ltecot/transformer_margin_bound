import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from datetime import datetime
import wandb
# from vit_pytorch import ViT
# from vit_pytorch.vit_for_small_dataset import ViT
import torchvision.models as models
from torchvision.models import resnet50
# from vit_pytorch.distill import DistillableViT, DistillWrapper
from vit import ViT

def tiny_imagenet_dataset():
    subdir_train = 'datasets/tiny-imagenet-200/train'
    subdir_test = 'datasets/tiny-imagenet-200/val/images'
    transform = transforms.Compose([
        transforms.Resize(64),
        # transforms.CenterCrop(56),
        transforms.RandomCrop(56),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(subdir_train, transform=transform)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(56),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(subdir_test, transform=transform)
    return train_dataset, test_dataset

def imagenet_dataset():
    subdir = "../rand_smoothing_indep_vars/datasets/imagenet"  # CUSTOM: Change filepath
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(subdir, transform)
    # train_set, test_set, other_set = torch.utils.data.random_split(dataset, [10000, 1000, 1270167]) # Total size is 1281167
    train_set, test_set = torch.utils.data.random_split(dataset, [1153051, 128116]) # Total size is 1281167
    return train_set, test_set

def cifar10_dataset():
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomErasing(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    return trainset, testset

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    config = {
        'dataset' : 'CIFAR10', # 'tiny_imagenet', # 'Imagenet', # 'CIFAR10'
        'batch_size' : 128,
        'epochs' : 1000,
        'lr' : 1e-4,
        # 'gamma' : 0.99,
        'print_interval' : 10,
        'save_interval' : 100,
        'image_size' : 32, # 56,
        'patch_size' : 2,
        'num_classes' : 10, # 200,
        'dim' : 256,
        'depth' : 6,
        # 'weight_decay' : 0.005,
        # First depth elements are for layers, first to last. Last element is for all other parameters
        # 'weight_decay' : [5e-1, 5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7, 5e-8, 5e-9, 5e-10], 
        # 'weight_decay' : [0.75, 0.25, 0.075, 0.025, 0.0075, 0.0025, 0.00075, 0.00075, 0.000025, 5e-2], 
        'weight_decay' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        # 'weight_decay' : [5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3],  
        # 'weight_decay' : [5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2],  
        'heads' : 9, # 12,
        'mlp_dim' : 256,
        'dropout' : 0,
        'emb_dropout' : 0,
        'spectral_norm_frequency' : 0,  # if less than or equal to 0, no normalization
        'spectral_norm_caps' : [0.95, 0.96, 0.97, 0.98, 0.99, 1],
        'num_workers' : 4,
    }

    net = ViT(
        image_size = config['image_size'],
        patch_size = config['patch_size'],
        num_classes = config['num_classes'],
        dim = config['dim'],
        depth = config['depth'],
        heads = config['heads'],
        mlp_dim = config['mlp_dim'],
        dropout = config['dropout'],
        emb_dropout = config['emb_dropout']
    ).to(device)

    # teacher = resnet50(pretrained = True)

    # net = DistillableViT(
    #     image_size = config['image_size'],
    #     patch_size = config['patch_size'],
    #     num_classes = config['num_classes'],
    #     dim = config['dim'],
    #     depth = config['depth'],
    #     heads = config['heads'],
    #     mlp_dim = config['mlp_dim'],
    #     dropout = config['dropout'],
    #     emb_dropout = config['emb_dropout']
    # ).to(device)

    # distiller = DistillWrapper(
    #     student = net,
    #     teacher = teacher,
    #     temperature = 3,           # temperature of distillation
    #     alpha = 0.5,               # trade between main loss and distillation loss
    #     hard = False               # whether to use soft or hard distillation
    # ).to(device)

    if config['dataset'] == 'Imagenet':
        trainset, testset = imagenet_dataset()
    elif config['dataset'] == 'tiny_imagenet':
        trainset, testset = tiny_imagenet_dataset()
    elif config['dataset'] == 'CIFAR10':
        trainset, testset = cifar10_dataset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                            shuffle=True, num_workers=config['num_workers'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'],
                                            shuffle=False, num_workers=config['num_workers'])

    # Decays
    # https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/6
    all_params = set(net.parameters())
    layer_params = [set() for _ in range(config['depth'])]
    # wd_params = set()
    for i, layer in enumerate(net.transformer.layers):
    # for i, layer in enumerate(distiller.student.transformer.layers):
        for p in layer.parameters():
            layer_params[i].add(p)
        # for m in layer.modules():
        #     layer_params[i].add(m.weight)
    other_params = all_params - set.union(*layer_params)
    layer_params.append(other_params)
    optim_list = []
    for i, lp in enumerate(layer_params):
        optim_list.append({'params': list(lp), 'weight_decay': config['weight_decay'][i]})

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer = optim.Adam(optim_list, lr=config['lr'])
    # optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    # scheduler = StepLR(optimizer, step_size=1, gamma=config['gamma'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # for m in net.modules():
    #     print(m)
    #     # if isinstance(m, (nn.Linear, nn.Conv2d, ...)):
    #     #     decay.append(m.weight)
    #     #     no_decay.append(m.bias)
    #     # elif hasattr(m, 'weight'):
    #     #     no_decay.append(m.weight)
    #     # elif hasattr(m, 'bias'):
    #     #     no_decay.append(m.bias)

    # print(optim_list[-1])

    # return

    wandb.init(project="transformer_margin", entity="ltecot", config=config)

    for epoch in range(config['epochs']):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # loss = distiller(inputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Spectral clipping
            if config['spectral_norm_frequency'] > 0 and i % config['spectral_norm_frequency'] == config['spectral_norm_frequency']-1:
                net.spectral_clipping(config['spectral_norm_caps'])

            # print statistics
            running_loss += loss.item()
            if i % config['print_interval'] == config['print_interval']-1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / config["print_interval"]:.3f}')
                wandb.log({"training/running_loss": running_loss / config['print_interval']})
                running_loss = 0.0

        print(f'Accuracy of train images: {100 * correct / total} %')
        wandb.log({"training/train_accuracy": 100 * correct / total, "Epoch": epoch})

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of test images: {100 * correct / total} %')
        wandb.log({"testing/test_accuracy": 100 * correct / total, "Epoch": epoch})

        if epoch % config['save_interval'] == config['save_interval']-1:
            now = datetime.now()
            dt_string = now.strftime("%Y:%m:%d:%H:%M:%S")
            PATH = './models/net_' + dt_string + '.pth'
            torch.save(net.state_dict(), PATH)
            wandb.save("model_weights_epoch_"+str(epoch)+".pt")

    print('Finished Training')

if __name__ == "__main__":
    main()