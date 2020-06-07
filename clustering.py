# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:21:08 2020

@author: Gaurav
"""
import seaborn as sns
from sklearn.datasets import make_gaussian_quantiles, make_blobs
import numpy as np
import matplotlib.pyplot as plt

x, _ = make_blobs(n_samples=2000, centers=[[1, 2],
                                           [3, 4],
                                           [5, 6],
                                           [-1, 10]],
                  cluster_std=[0.2, 0.3, 0.5, 0.1],
                  n_features=2,random_state=0)
x = np.dot(x, [[1, -3],
               [-2, 1]]) #covariance

sns.scatterplot(x[:,0], x[:,1],color='red')
#%% kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def silhouette_plot(scores, classes, ax):
    mean_score = scores.mean()
    unique_clusters = list(np.unique(classes))
    scores_list = []
    num_plotted_points = 0
    for i in unique_clusters:
        scores_list.append(np.sort(scores[classes == i]))
        start_index = num_plotted_points
        end_index = num_plotted_points + scores_list[-1].shape[0]
        ax.fill_between(range(start_index, end_index), scores_list[-1])
        num_plotted_points = num_plotted_points + scores_list[-1].shape[0]
    ax.plot(range(scores.shape[0]) , np.tile(mean_score, scores.shape[0]), color='k')
    return scores_list


fig = plt.figure(figsize=(20,9), constrained_layout=False)
fig2 = plt.figure(figsize=(20,9), constrained_layout=False)
inertia = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters)
    #kmeans = MiniBatchKMeans(n_clusters=n_clusters) #faster
    #y_dist = kmeans.fit_transform(x) #predict the distances
    y = kmeans.fit_predict(x) #predict the classes
    params = kmeans.get_params() #get the parameters set during training
    score = kmeans.score(x) #negative of intertia
    inertia.append(kmeans.inertia_)
    print(kmeans.n_iter_)
    ax = fig.add_subplot(3, 4, n_clusters)
    ax.set(title='n_clusters = ' + str(n_clusters) + ',score= ' + str(round(score, 2)))
    sns.scatterplot(x[:,0], x[:,1], hue=y,
                    palette=sns.color_palette('bright', np.unique(y).shape[0]), ax=ax)
    sns.scatterplot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='k', ax=ax)
    ax2 = fig2.add_subplot(3, 4, n_clusters)
    scores = silhouette_samples(x, kmeans.predict(x))
    silhouette_plot(scores, kmeans.predict(x), ax2)
    
sns.lineplot(range(1,5), inertia)
#%% image compression
from sklearn.cluster import DBSCAN
import tensorflow as tf

img = tf.keras.preprocessing.image.load_img('original.jpg')
img = tf.keras.preprocessing.image.img_to_array(img).astype(int)
x = img.reshape(-1, 3)
kmeans = KMeans(n_clusters=8)
kmeans.fit(x[np.random.randint(0, x.shape[0], (1000,))])
x_ = kmeans.cluster_centers_[kmeans.predict(x)]
x_ = x_.reshape(img.shape).astype(int)
plt.imshow(x_)

#%% clustering as preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
x, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
del x, y

clf = LogisticRegression().fit(x_train, y_train)
clf.score(x_test, y_test)

clf = Pipeline([('kmeans', KMeans(n_clusters=50)),
                ('log_clf', LogisticRegression())]).fit(x_train, y_train)
clf.score(x_test, y_test)

#%% clusttering as semi-supervised learning [no y_label available]
del y_train
kmeans = KMeans(n_clusters=50).fit(x_train)
x_transform = kmeans.transform(x_train)
id_x_transform = np.argmin(x_transform, axis=0)

x_train_ = x_train[id_x_transform].reshape(-1,8,8)

fig = plt.figure()
for i in range(50):
    ax = fig.add_subplot(7, 8, i+1)
    plt.imshow(x_train_[i])

#manually add label
y_train_ = np.array([6, 8, 3, 7, 4, 8, 5, 0,
            2, 1, 9, 1, 4, 5, 2, 6,
            6, 0, 7, 4, 3, 8, 1, 1,
            4, 7, 9, 6, 8, 5, 9, 8,
            1, 2, 9, 7, 3, 0, 7, 2,
            4, 1, 1, 3, 0, 9, 2, 8,
            6, 4])

clf = LogisticRegression()
clf.fit(x_train_.reshape(-1,64), y_train_)
clf.score(x_test, y_test)

#propagate the labels
propagate_length = int(0.3 * x_train.shape[0]/50)
x_train_prop = np.zeros((propagate_length*50, 64))
y_train_prop = np.zeros(propagate_length*50)
for i in range(50):
    ids = np.argsort(x_transform[:,i])[:propagate_length]
    x_train_prop[i*propagate_length:(i+1)*propagate_length,:] = x_train[ids,:]
    y_train_prop[i*propagate_length:(i+1)*propagate_length] = y_train_[i]

clf.fit(x_train_prop, y_train_prop)
clf.score(x_test, y_test)


#%%DBSCAN
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

x, _ = make_moons(1000, noise=0.05, random_state=42)
scale = StandardScaler().fit(x)
x_train = scale.transform(x)

#sns.scatterplot(x[:,0],x[:,1])
#plt.figure()
#sns.scatterplot(x_train[:,0],x_train[:,1])
dbscan = DBSCAN(eps=0.12).fit(x_train)
sns.scatterplot(x[:,0], x[:,1], hue=dbscan.labels_,
                palette=sns.color_palette('bright',len(np.unique(dbscan.labels_))))

idx_core = dbscan.core_sample_indices_
plt.figure()
sns.scatterplot(x[idx_core,:][:,0], x[idx_core,:][:,1])

clf = KNeighborsClassifier(n_neighbors=50).fit(dbscan.components_, dbscan.labels_[idx_core])
x_test = np.meshgrid(np.linspace(-1,2, 50), np.linspace(-0.75,1.5, 50))
x_test_0 = x_test[0].flatten()[:, None]
x_test_1 = x_test[1].flatten()[:, None]
x_test = np.concatenate([x_test_0, x_test_1], axis=-1)
del x_test_0, x_test_1

y_pred = clf.predict(scale.transform(x_test))
sns.scatterplot(x_test[:,0], x_test[:,1], hue=y_pred)

#%% Gaussian Mixture
x1, _ = make_blobs(n_samples=2000, centers=[[1, 2],
                                           [3, 4],
                                           [5, 6],
                                           [-1, 10]],
                  cluster_std=[0.2, 0.3, 0.5, 0.1],
                  n_features=2,random_state=0)

x1 = np.dot(x1, [[1, -3],
               [-2, 1]]) #covariance

x2, _ = make_blobs(n_samples=2000, centers=[[-15, 0]],
                  cluster_std=0.15,
                  n_features=2,random_state=0)
x = np.concatenate([x1, x2], axis=0)
del x1, x2
plt.figure()
sns.scatterplot(x[:,0], x[:,1],color='red', s=5)
sns.jointplot(x[:,0], x[:,1],color='red', s=5, kind='scatter')

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
gm = GaussianMixture(n_components=5, n_init=10).fit(x)
gm.converged_
#gm = BayesianGaussianMixture(n_components=10).fit(x) helpful when no of cluster is diff to obtain

sns.scatterplot(gm.means_[:,0], gm.means_[:,1])
gm.weights_
gm.covariances_
gm.n_iter_
gm.score_samples(x)

plt.figure()
x_new, y_new = gm.sample(3000)
palette = sns.color_palette('bright',len(np.unique(y_new)))
sns.scatterplot(x_new[:,0], x_new[:,1], hue=y_new, palette=palette)

#picking right no of clusters
aic = []
bic = []
for i in range(1,10):
    gm = GaussianMixture(n_components=i, n_init=10).fit(x)
    aic.append(gm.aic(x))
    bic.append(gm.bic(x))
    
sns.lineplot(range(1,10), aic)
sns.lineplot(range(1,10), bic)

x1 = np.linspace(-23, 5, 100)
x2 = np.linspace(-15, 20, 100)
x1 , x2 = np.meshgrid(x1, x2)
x_mesh = np.concatenate([x1.flatten()[:,None], x2.flatten()[:,None]], axis=-1)
y_mesh = gm.score_samples(x_mesh).reshape(100,100)
plt.figure()
plt.contour(x1, x2, y_mesh, levels=200)
