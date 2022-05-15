import matplotlib.pyplot as plt
import seaborn as sns
import torch

margins = torch.load('margin_files/mnist_margins.pt')
margins = margins.squeeze().cpu().numpy()
sns.distplot(margins, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'MNIST')

margins = torch.load('margin_files/cifar10_margins.pt')
margins = margins.squeeze().cpu().numpy()
sns.distplot(margins, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'CIFAR10')

plt.legend(loc="upper right")
plt.xlabel(None)
plt.ylabel(None)
plt.xticks([])
plt.yticks([])
plt.savefig('plot_images/temp.png')