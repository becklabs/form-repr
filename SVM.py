import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
import os


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


path_to_npy = './data/embed/Embeddings/'

x = []
y = []
for file_name in [file for file in os.listdir(path_to_npy) if file.endswith('.npy')]:
    frames = np.load(path_to_npy + file_name)
    frames = frames.reshape(-1, 17*512)
    frames = np.mean(frames, axis=0)

    if file_name.startswith("Perfect"):
        y.append("Good_form")
    else:
        y.append("Overstride/Heelstrike")
    x.append(frames)

overstride_frames = np.load('./data/embed/oliver/IMG_5046.npy')
overstride_frames = overstride_frames.reshape(-1, 17*512)
overstride_frames = np.mean(overstride_frames, axis=0)


np.random.seed(69)
x = np.asarray(x)
y = np.asarray(y)
print(x.shape)
kf = KFold(n_splits=4, shuffle=True,random_state=16)
cnt = 1
# split()  method generate indices to split data into training and test set.
for train_index, test_index in kf.split(x, y):
    print(f'Fold:{cnt}, Train set: {train_index}, Test set:{test_index}')
    x_train = []
    y_train = []
    for i in train_index:
        x_train.append(x[i])
        y_train.append(y[i])

    x_test = []
    y_test = []
    for i in test_index:
        x_test.append(x[i])
        y_test.append(y[i])

    support = svm.LinearSVC(random_state=cnt)
    support.fit(x_train, y_train)
    predicted = support.predict(x_train)
    score = accuracy_score(y_train, predicted)
    print(score)
    predicted = support.predict(x_test)
    score = accuracy_score(y_test, predicted)
    print(score)
    print()
    cnt += 1



from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

# We want to get TSNE embedding with 2 dimensions
n_components = 2
tsne = TSNE(n_components,perplexity=5)
x = np.asarray(x)

tsne_result = tsne.fit_transform(x)

# (1000, 2)
# Two dimensions for each of our images

# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})

# figure = plt.figure(figsize=(9,9))
# axes = figure.add_subplot(111,projection = "3d")
# axes.scatter(xs = tsne_result[:,0], ys = tsne_result[:,1], zs = tsne_result[:,2], data=tsne_result, s=120)
# plt.show()

fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120)
lim = (tsne_result.min() - 5, tsne_result.max() + 5)
ax.set_xlim(lim)
ax.set_ylim(lim)
plt.legend()
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
plt.show()