import matplotlib.pyplot as plt
import seaborn as sns
import torch

# ax = plt.axes()
# ax.set_facecolor("grey")

margins = torch.load('margin_files/mnist_margins_3.pt')
margins = margins.squeeze().cpu().numpy()
sns.distplot(margins, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = 'MNIST')

margins = torch.load('margin_files/cifar10_margins_3.pt')
margins = margins.squeeze().cpu().numpy()
sns.distplot(margins, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = 'CIFAR10')

margins = torch.load('margin_files/cifar100_margins_3.pt')
margins = margins.squeeze().cpu().numpy()
sns.distplot(margins, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = 'CIFAR100')

# margins = torch.load('margin_files/cifar10_1e-3_margins.pt')
# margins = margins.squeeze().cpu().numpy()
# sns.distplot(margins, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = '1e-3')

# margins = torch.load('margin_files/cifar10_1e-4_margins.pt')
# margins = margins.squeeze().cpu().numpy()
# sns.distplot(margins, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = '1e-4')

# margins = torch.load('margin_files/cifar10_1e-5_margins.pt')
# margins = margins.squeeze().cpu().numpy()
# sns.distplot(margins, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = '1e-5')

# margins = torch.load('margin_files/cifar10_1e-6_margins.pt')
# margins = margins.squeeze().cpu().numpy()
# sns.distplot(margins, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = '1e-6')


plt.legend(loc="upper right")
plt.xlabel(None)
plt.ylabel(None)
plt.xticks([])
plt.yticks([])
# plt.xlim(0, 2e-19)
# plt.ylim(0, 4e20)
# plt.style.use('seaborn-paper')
plt.savefig('plot_images/datasets_compare.png')
# plt.savefig('plot_images/cifar_wd.png')