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
import torchvision.models as models
from torchvision.models import resnet50
from vit import ViT
from datasets import mnist_dataset, cifar100_dataset, cifar10_dataset, RandomizedDataset


# rand labels
# target_transform=lambda y: torch.randint(0, 10, (1,)).item(),

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    config = {
        'dataset' : 'MNIST', # 'tiny_imagenet', # 'CIFAR100', # 'CIFAR10' 'MNIST'
        'weight_decay' : 0,  
        'random_labels' : True,
        # 'num_classes' : 10, # 10, 100, 200
        'batch_size' : 128,
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
        config['image_size'] = 28
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

    # print(net.to_patch_embedding[1].weight.shape)
    # return

    if config['random_labels']:
        trainset = RandomizedDataset(trainset)

    loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                         shuffle=True, num_workers=config['num_workers'])

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    wandb.init(project="transformer_margin", entity="ltecot", config=config)

    for epoch in range(config['epochs']):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(loader, 0):
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
            # if config['spectral_norm_frequency'] > 0 and i % config['spectral_norm_frequency'] == config['spectral_norm_frequency']-1:
            #     net.spectral_clipping(config['spectral_norm_caps'])

            # print statistics
            running_loss += loss.item()
            if i % config['print_interval'] == config['print_interval']-1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / config["print_interval"]:.3f}')
                wandb.log({"training/running_loss": running_loss / config['print_interval']})
                running_loss = 0.0

        print(f'Accuracy of train images: {100 * correct / total} %')
        wandb.log({"training/train_accuracy": 100 * correct / total, "Epoch": epoch})

        # net.eval()
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for data in testloader:
        #         images, labels = data[0].to(device), data[1].to(device)
        #         # calculate outputs by running images through the network
        #         outputs = net(images)
        #         # the class with the highest energy is what we choose as prediction
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # print(f'Accuracy of test images: {100 * correct / total} %')
        # wandb.log({"testing/test_accuracy": 100 * correct / total, "Epoch": epoch})

        if epoch % config['save_interval'] == config['save_interval']-1:
            now = datetime.now()
            dt_string = now.strftime("%Y:%m:%d:%H:%M:%S")
            PATH = './models/net_epoch_' + str(epoch+1) + '.pth'
            torch.save(net.state_dict(), PATH)
            # wandb.save("model_weights_epoch_"+str(epoch)+".pt")
            wandb.save(PATH)

    print('Finished Training')

if __name__ == "__main__":
    main()


# # restore the model file "model.h5" from a specific run by user "lavanyashukla"
# # in project "save_and_restore" from run "10pr4joa"
# best_model = wandb.restore('model.h5', run_path="lavanyashukla/save_and_restore/10pr4joa")

# # use the "name" attribute of the returned object if your framework expects a filename, e.g. as in Keras
# model.load_weights(best_model.name)