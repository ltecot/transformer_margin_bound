import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from datetime import datetime
import wandb
# from vit_pytorch import ViT
from vit_pytorch.vit_for_small_dataset import ViT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

config = {
    'batch_size' : 64,
    'epochs' : 500,
    'lr' : 1e-3,
    # 'gamma' : 0.99,
    'weight_decay' : 0.005,
    'print_interval' : 10,
    'save_interval' : 50,
    'image_size' : 32,
    'patch_size' : 2,
    'num_classes' : 10,
    'dim' : 1024,
    'depth' : 3,
    'heads' : 12,
    'mlp_dim' : 1024,
    'dropout' : 0,
    'emb_dropout' : 0,
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

# wandb.config.batch_size = batch_size
# wandb.config.epochs = epochs
# wandb.config.lr = lr
# wandb.config.gamma = gamma
# wandb.config.weight_decay = weight_decay
# # wandb.config.image_size = net.image_size
# wandb.config.patch_size = patch_size
# wandb.config.dim = dim
# wandb.config.depth = depth
# wandb.config.heads = heads
# wandb.config.mlp_dim = mlp_dim
# wandb.config.dropout = dropout
# wandb.config.emb_dropout = emb_dropout

# functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'],
                                         shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
        #    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# net = Net()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
# scheduler
# scheduler = StepLR(optimizer, step_size=1, gamma=config['gamma'])
scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

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
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()
        if i % config['print_interval'] == config['print_interval']-1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / config["print_interval"]:.3f}')
            wandb.log({"training/running_loss": running_loss / config['print_interval']})
            running_loss = 0.0

    print(f'Accuracy of train images: {100 * correct // total} %')
    wandb.log({"training/train_accuracy": 100 * correct // total})

    net.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of test images: {100 * correct // total} %')
    wandb.log({"testing/test_accuracy": 100 * correct // total})

    if epoch % config['save_interval'] == config['save_interval']-1:
        now = datetime.now()
        dt_string = now.strftime("%Y:%m:%d:%H:%M:%S")
        PATH = './models/net_' + dt_string + '.pth'
        torch.save(net.state_dict(), PATH)

print('Finished Training')

# dataiter = iter(testloader)
# images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


# net = Net()
# net.load_state_dict(torch.load(PATH))

# outputs = net(images)

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
#                               for j in range(4)))

# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1


# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')