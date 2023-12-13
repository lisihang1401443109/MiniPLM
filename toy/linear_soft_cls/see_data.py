from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# data_path = "/home/lidong1/yuxian/sps-toy/processed_data/toy_data/10-30-1"
# data_path = "/home/lidong1/yuxian/sps-toy/processed_data/toy_data/10-10-1"
data_path = "/home/lidong1/yuxian/sps-toy/processed_data/toy_data/10-20-1"


train_x, train_y, dev_x, dev_y, test_x, test_y, theta_init = torch.load(
    os.path.join(data_path, "data.pt"), map_location="cpu")

X = torch.cat([train_x, dev_x, test_x], dim=0).numpy()

X_std = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_std)
# X_pca = np.vstack((X_pca.T, y)).T

print(X_pca.shape)

train_x_pca = torch.tensor(X_pca[:train_x.size(0)])
dev_x_pca = torch.tensor(X_pca[train_x.size(0):train_x.size(0)+dev_x.size(0)])
test_x_pca = torch.tensor(X_pca[train_x.size(0)+dev_x.size(0):])

plt.scatter(train_x_pca[:, 0], train_x_pca[:, 1], label="train", s=4, c=train_y, cmap="coolwarm", marker="x")
plt.scatter(dev_x_pca[:, 0], dev_x_pca[:, 1], label="dev", s=4, c=dev_y, cmap="coolwarm", marker="^")
plt.scatter(test_x_pca[:, 0], test_x_pca[:, 1], label="test", s=4, c=test_y, cmap="coolwarm", marker="o")

plt.savefig(os.path.join(data_path, "toy_data.pdf"), format="pdf")