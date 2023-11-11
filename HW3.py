from sklearn.datasets import load_iris, load_breast_cancer, load_wine
Xbreast, ybreast = load_breast_cancer(return_X_y=True)
Xwine, ywine = load_wine(return_X_y=True)

# clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# dimension reduction
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# scoring
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
#plt.style.use('seaborn-poster')
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

Xiris = StandardScaler().fit_transform(Xiris)
Xbreast = StandardScaler().fit_transform(Xbreast)
Xwine = StandardScaler().fit_transform(Xwine)

clusters = np.arange(1,10)
inertiaListWine = []

for n in clusters:
    km = KMeans(n_clusters=n)
    km.fit(Xiris)
    inertiaListWine.append(km.inertia_)

clusters = np.arange(1,10)
inertiaListBreast = []

for n in clusters:
    km = KMeans(n_clusters=n)
    km.fit(Xbreast)
    inertiaListBreast.append(km.inertia_)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(clusters, inertiaListWine)
plt.ylabel('Inertia')
plt.xlabel("Clusters")
plt.title('Red Wine - Inertia v Clusters')

plt.subplot(1,2,2)
plt.plot(clusters, inertiaListBreast)
plt.ylabel('Inertia')
plt.xlabel("Clusters")
plt.title('Breast Cancer - Inertia v Clusters')

plt.tight_layout()

from sklearn.metrics import silhouette_score

cluster_range = range(2, 11)

silhouette_avg_scores = []

# Calculate silhouette scores for different number of clusters
for n_clusters in cluster_range:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(Xbreast)
    silhouette_avg = silhouette_score(Xbreast, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg}")

# Plot the results
plt.plot(cluster_range, silhouette_avg_scores, marker='o')
plt.title("Silhouette Coefficient for each k - Breast Cancer")
plt.xlabel("clusters (k)")
plt.ylabel("Silhouette Coefficient")

from sklearn.mixture import GaussianMixture

bic_scores = []
aic_scores = []
n_components_range = range(1, 11)

# Applying GMM for each number of components in the range
for n_components in n_components_range:
    # Create a Gaussian Mixture model with the current number of components
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    # Fit the model to the data
    gmm.fit(Xwine)
    # Append BIC and AIC scores for the current model to the lists
    bic_scores.append(gmm.bic(Xwine))
    aic_scores.append(gmm.aic(Xwine))

# Plotting the BIC & AIC scores
plt.figure(figsize=(10, 5))
plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
plt.plot(n_components_range, aic_scores, label='AIC', marker='o')
plt.xlabel('components (k)')
plt.ylabel('Scores')
plt.legend()
plt.title('BIC & AIC Scores for Wine Dataset')

from collections import defaultdict

clusters = np.arange(1,10)
covTypeList = ['full', 'tied', 'diag', 'spherical']

for X in [Xiris, Xbreast]:
    aicDict = defaultdict(list)
    bicDict = defaultdict(list)
    for covType in covTypeList:
        for n in clusters:
            gm = GaussianMixture(n_components=n, covariance_type=covType, max_iter=500, n_init=5, reg_covar=1e-2)
            gm.fit(X)
            aic = gm.aic(X)
            bic = gm.bic(X)
            aicDict[covType].append(aic)
            bicDict[covType].append(bic)

    plt.figure(figsize=(12,4))


    plt.subplot(1,2,1)
    for key, value in aicDict.items():
        plt.plot(clusters, value, label=key)
    plt.ylabel('AIC score')
    plt.xlabel("number of clusters")
    plt.yscale('log')
    plt.legend()
    plt.title('AIC')

    plt.subplot(1,2,2)
    for key, value in bicDict.items():
        plt.plot(clusters, value, label=key)
    plt.ylabel('BIC score')
    plt.xlabel("number of clusters")
    plt.yscale('log')
    plt.legend()
    plt.title('BIC')

    plt.tight_layout()

# Load the breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define dictionaries to hold the accuracy and time scores for each method
accuracy_scores = {}
time_scores = {}

# Function to train and evaluate the model, returning accuracy and time taken
def train_evaluate(mlp, X_train, y_train, X_test, y_test):
    start_time = time.time()
    mlp.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    y_test_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    return accuracy, elapsed_time

# Initialize the Multi-layer Perceptron classifier with parameters to help prevent overfitting
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, alpha=0.01,
                    solver='adam', random_state=21, tol=0.0001, early_stopping=True, validation_fraction=0.1)

# No Reduction
accuracy_scores['No Reduction'], time_scores['No Reduction'] = train_evaluate(mlp, X_train_scaled, y_train, X_test_scaled, y_test)

# PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
accuracy_scores['PCA'], time_scores['PCA'] = train_evaluate(mlp, X_train_pca, y_train, X_test_pca, y_test)

# ICA
ica = FastICA(n_components=10, random_state=42)
X_train_ica = ica.fit_transform(X_train_scaled)
X_test_ica = ica.transform(X_test_scaled)
accuracy_scores['ICA'], time_scores['ICA'] = train_evaluate(mlp, X_train_ica, y_train, X_test_ica, y_test)

# Random Projection
grp = GaussianRandomProjection(n_components=10, random_state=42)
X_train_grp = grp.fit_transform(X_train_scaled)
X_test_grp = grp.transform(X_test_scaled)
accuracy_scores['RP'], time_scores['RP'] = train_evaluate(mlp, X_train_grp, y_train, X_test_grp, y_test)

# Information Gain
info_gain = SelectKBest(mutual_info_classif, k=10)
X_train_info_gain = info_gain.fit_transform(X_train_scaled, y_train)
X_test_info_gain = info_gain.transform(X_test_scaled)
accuracy_scores['IG'], time_scores['IG'] = train_evaluate(mlp, X_train_info_gain, y_train, X_test_info_gain, y_test)

# Plotting the accuracies and times
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
methods = list(accuracy_scores.keys())
accuracy_values = list(accuracy_scores.values())
axes[0].bar(methods, accuracy_values, color=['orange', 'blue', 'green', 'red', 'purple'])
axes[0].set_xlabel('Dimensionality Reduction Method')
axes[0].set_ylabel('Accuracy Score')
axes[0].set_title('Accuracy Scores by Method')
axes[0].set_ylim([0.7, 1])

# Time plot
time_values = list(time_scores.values())
axes[1].bar(methods, time_values, color=['orange', 'blue', 'green', 'red', 'purple'])
axes[1].set_xlabel('Dimensionality Reduction Method')
axes[1].set_ylabel('Time Taken (seconds)')
axes[1].set_title('Computation Time by Method')

plt.tight_layout()
plt.show()



cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

methods = ['No Reduction', 'PCA', 'ICA', 'RP', 'IG']
accuracy_scores_em = {}

em_no_reduction = GaussianMixture(n_components=2, random_state=42)
em_no_reduction.fit(X_train_scaled)
accuracy_scores_em['No Reduction'] = accuracy_score(y_test, em_no_reduction.predict(X_test_scaled))

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
em_pca = GaussianMixture(n_components=2, random_state=42)
em_pca.fit(X_train_pca)
accuracy_scores_em['PCA'] = accuracy_score(y_test, em_pca.predict(X_test_pca))

ica = FastICA(n_components=10, random_state=42)
X_train_ica = ica.fit_transform(X_train_scaled)
X_test_ica = ica.transform(X_test_scaled)
em_ica = GaussianMixture(n_components=2, random_state=42)
em_ica.fit(X_train_ica)
accuracy_scores_em['ICA'] = accuracy_score(y_test, em_ica.predict(X_test_ica))

grp = GaussianRandomProjection(n_components=10, random_state=42)
X_train_grp = grp.fit_transform(X_train_scaled)
X_test_grp = grp.transform(X_test_scaled)
em_grp = GaussianMixture(n_components=2, random_state=42)
em_grp.fit(X_train_grp)
accuracy_scores_em['RP'] = accuracy_score(y_test, em_grp.predict(X_test_grp))

info_gain = SelectKBest(mutual_info_classif, k=10)
X_train_info_gain = info_gain.fit_transform(X_train_scaled, y_train)
X_test_info_gain = info_gain.transform(X_test_scaled)
em_info_gain = GaussianMixture(n_components=2, random_state=42)
em_info_gain.fit(X_train_info_gain)
accuracy_scores_em['IG'] = accuracy_score(y_test, em_info_gain.predict(X_test_info_gain))

scores_em = [accuracy_scores_em[method] for method in methods]

plt.figure(figsize=(10, 6))
plt.bar(methods, scores_em, color=['yellow', 'blue', 'orange', 'green', 'red'])
plt.xlabel('Dimensionality Reduction Method')
plt.ylabel('Accuracy Score with EM')
plt.title('EM Clustering Accuracy Across Different Dimensionality Reduction Methods')
plt.show()
