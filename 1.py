import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris_data = pd.read_csv('Iris.csv')
data_features = iris_data.iloc[:, :-1].values
data_target = iris_data.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoded_target = encoder.fit_transform(data_target)

standardizer = StandardScaler()
std_features = standardizer.fit_transform(data_features)

lda = LinearDiscriminantAnalysis(n_components=2)
lda_features = lda.fit_transform(std_features, encoded_target)

pca = PCA(n_components=2)
pca_features = pca.fit_transform(std_features)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for idx, class_label in zip(range(len(encoder.classes_)), encoder.classes_):
    plt.scatter(lda_features[encoded_target == idx, 0], lda_features[encoded_target == idx, 1], label=class_label)
plt.title('LDA: Iris Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()

plt.subplot(1, 2, 2)
for idx, class_label in zip(range(len(encoder.classes_)), encoder.classes_):
    plt.scatter(pca_features[encoded_target == idx, 0], pca_features[encoded_target == idx, 1], label=class_label)
plt.title('PCA: Iris Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

plt.tight_layout()
plt.show()
