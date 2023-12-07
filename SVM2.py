import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Simulating the data loading as described in the user's code
# Generating synthetic data for demonstration as we don't have access to the actual data
np.random.seed(42)
x = np.random.rand(18, 17 * 512)  # 18 samples, each with 17*512 features
y = np.random.randint(0, 2, 18)   # Random binary labels

# Splitting the data as per user's code
rand_indices = np.random.choice(x.shape[0], 14, replace=False)
r = np.array([i for i in range(18) if i not in rand_indices])

x_train = x[rand_indices]
y_train = y[rand_indices]
x_test = x[r.astype(int)]
y_test = y[r.astype(int)]

# Training the SVM model
support = svm.LinearSVC(random_state=20)
support.fit(x_train, y_train)

# Predicting and calculating accuracy
predicted_train = support.predict(x_train)
score_train = accuracy_score(y_train, predicted_train)

predicted_test = support.predict(x_test)
score_test = accuracy_score(y_test, predicted_test)

# Output the train and test scores
print(f"Training Accuracy: {score_train}")
print(f"Test Accuracy: {score_test}")

# TSNE for dimensionality reduction to 2D
tsne = TSNE(n_components=2, perplexity=5)
tsne_result = tsne.fit_transform(x)

# Creating a DataFrame for plotting
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})

# Plotting the TSNE results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, s=120)
plt.title('t-SNE visualization of SVM Data')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()

# Saving the plot to a file
plot_file_path = '/mnt/data/svm_tsne_visualization.png'
plt.savefig(plot_file_path)
plt.close()

plot_file_path, score_train, score_test
