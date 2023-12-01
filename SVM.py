import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
import os




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


path_to_npy = '../Untitled/Embeddings/'

x = []
y = []
for file_name in [file for file in os.listdir(path_to_npy) if file.endswith('.npy')]:
    frames = np.load(path_to_npy + file_name)
    temp = 0
    frames = frames.reshape(-1, 17*512)
    frames = np.mean(frames, axis=0)


    if file_name.startswith("Perfect"):
        y.append(1)
    else:
        y.append(0)
    x.append(frames)


np.random.seed(42)
x = np.asarray(x)
y = np.asarray(y)
rand_indices = np.random.choice(x.shape[0], 14, replace=False)
r = np.array([])
for i in range(0,18):
    if i not in rand_indices:

        r = np.append(r,i)

print(rand_indices)
print(r.astype(int))
x_train = x[rand_indices]
y_train = y[rand_indices]

x_test = x[r.astype(int)]
y_test = y[r.astype(int)]


support = svm.LinearSVC(random_state=20)
support.fit(x_train, y_train)
predicted = support.predict(x_train)
score=accuracy_score(y_train,predicted)
print(score)

predicted = support.predict(x_test)
score=accuracy_score(y_test,predicted)
print(score)


from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

# We want to get TSNE embedding with 2 dimensions
# n_components = 3
# tsne = TSNE(n_components,perplexity=5)
# x = np.asarray(x)
#
# tsne_result = tsne.fit_transform(x)
#
# # (1000, 2)
# # Two dimensions for each of our images
#
# # Plot the result of our TSNE with the label color coded
# # A lot of the stuff here is about making the plot look pretty and not TSNE
# tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
# fig, ax = plt.subplots(1)
# sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120)
# lim = (tsne_result.min() - 5, tsne_result.max() + 5)
# ax.set_xlim(lim)
# ax.set_ylim(lim)
# ax.set_aspect('equal')
# ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# plt.show()