import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

data_wine = pd.read_csv('wine.csv')
data_features = data_wine.drop(columns='Target')
data_labels = data_wine['Target']

X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, test_size=0.3, random_state=42, stratify=data_labels)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=2)
train_lda = lda.fit_transform(train_scaled, y_train)
test_lda = lda.transform(test_scaled)

logistic = LogisticRegression(max_iter=200)
logistic.fit(train_lda, y_train)

pred_lda = lda.predict(test_scaled)
accuracy_lda = accuracy_score(y_test, pred_lda)
precision_lda = precision_score(y_test, pred_lda, average='weighted')
recall_lda = recall_score(y_test, pred_lda, average='weighted')
confusion_lda = confusion_matrix(y_test, pred_lda)

print("LDA Model Evaluation:")
print(f"Accuracy: {accuracy_lda:.2f}")
print(f"Precision: {precision_lda:.2f}")
print(f"Recall: {recall_lda:.2f}")
print("Confusion Matrix:")
print(confusion_lda)

pred_logistic = logistic.predict(test_lda)
accuracy_logistic = accuracy_score(y_test, pred_logistic)
precision_logistic = precision_score(y_test, pred_logistic, average='weighted')
recall_logistic = recall_score(y_test, pred_logistic, average='weighted')
confusion_logistic = confusion_matrix(y_test, pred_logistic)

print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy_logistic:.2f}")
print(f"Precision: {precision_logistic:.2f}")
print(f"Recall: {recall_logistic:.2f}")
print("Confusion Matrix:")
print(confusion_logistic)

xx, yy = np.meshgrid(np.linspace(train_lda[:, 0].min() - 1, train_lda[:, 0].max() + 1, 100),
                     np.linspace(train_lda[:, 1].min() - 1, train_lda[:, 1].max() + 1, 100))

grid = np.c_[xx.ravel(), yy.ravel()]
Z = logistic.predict(grid)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(train_lda[:, 0], train_lda[:, 1], c=y_train, edgecolor='k', marker='o')
plt.title('Logistic Regression Decision Boundaries in LDA 2D Space')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.show()
