import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_analyzing import get_data

parameters = [
    ("Unchanged data", False, False),
    ("Normalised", False, True),
    ("Only important features", True, False),
    ("Only important features and normalised", True, True),
]

for title, filter, normalise in parameters:
    X, y = get_data(only_important_features=filter, normalise=normalise)
    X['y'] = y

    clip_size = 2.5
    colors = StandardScaler().fit_transform(np.array(y).reshape(-1, 1))[:, 0].clip(-clip_size, clip_size) + clip_size

    X = PCA(n_components=2).fit_transform(X)

    # Reorder the labels to have colors matching the cluster results
    plt.scatter(X[:, 0], X[:, 1], s=20, c=colors, cmap="RdBu_r", edgecolor='k')

    plt.title(title)
    plt.show()
