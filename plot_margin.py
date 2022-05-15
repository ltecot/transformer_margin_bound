import matplotlib.pyplot as plt
import seaborn as sns
import torch

margins = torch.load('margin_files/mnist_margins.pt')
margins = margins.squeeze().cpu().numpy()
print(margins.shape)
fig = sns.distplot(margins, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3})
fig.set(xlabel=None)
fig.set(ylabel=None)
plt.savefig('plot_images/temp.png')