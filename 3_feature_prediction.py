import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA  # Add import statement

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Separate features and target
X = data[['Glucose', 'BMI', 'Age']]
y = data['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naïve Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naïve Bayes Classifier Accuracy:", accuracy_nb)

# K-NN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("K-NN Classifier Accuracy:", accuracy_knn)

# Logistic Regression Classifier
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Classifier Accuracy:", accuracy_lr)



# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.show()

# Plot confusion matrix for Naïve Bayes classifier
plot_confusion_matrix(y_test, y_pred_nb, "Naïve Bayes Classifier Confusion Matrix")

# Plot confusion matrix for K-NN classifier
plot_confusion_matrix(y_test, y_pred_knn, "K-NN Classifier Confusion Matrix")

# Plot confusion matrix for Logistic Regression classifier
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression Classifier Confusion Matrix")

# Function to plot decision regions
def plot_decision_boundary(X, y, classifier, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    classifier.fit(X_pca, y)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X_pca, y.values, clf=classifier, legend=2)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.show()

# Plot decision boundary for K-NN classifier
plot_decision_boundary(X_test, y_test, knn_classifier, 'K-NN Classifier Decision Boundary')

# Plot decision boundary for Logistic Regression classifier
plot_decision_boundary(X_test, y_test, lr_classifier, 'Logistic Regression Classifier Decision Boundary')

# Add Logistic Regression Equation
coef = lr_classifier.coef_[0]
intercept = lr_classifier.intercept_[0]

x_values = np.array([X_test.values[:, 0].min(), X_test.values[:, 0].max()])
y_values = (-intercept - coef[0] * x_values) / coef[1]
plt.plot(x_values, y_values, label=f'Logistic Regression: BMI = {-coef[0]/coef[1]:.2f}*Glucose + {-intercept/coef[1]:.2f}', color='red')

plt.legend()
plt.show()
