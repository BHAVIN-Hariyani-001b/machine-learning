import random
import numpy as np

class KMeans:
    def __init__(self,n_clusters=2,max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self,X):
        random_index = random.sample(range(X.shape[0]),self.n_clusters)
        s = self.centroids = X[random_index]

        for i in range(self.max_iter):
            ## assing clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids

            ## move centroids
            self.centroids = self.move_centroids(X,cluster_group)

            ## check finish
            if np.allclose(old_centroids, self.centroids):
                break
        return cluster_group 

    def assign_clusters(self,X):
        cluster_group = []

        distances = [] # append distance for 2 cluster

        for row in X:
            for centroid in self.centroids:
                distances.append(np.linalg.norm(row - centroid))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()
        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []
    
        for i in range(self.n_clusters):
            points = X[cluster_group == i]
    
            if len(points) == 0:
                # Keep the old centroid if no points are assigned
                new_centroids.append(self.centroids[i])
            else:
                new_centroids.append(points.mean(axis=0))
    
        return np.array(new_centroids)

    def calculate_wcss(self, X, cluster_group):

        wcss = 0

        for i in range(self.n_clusters):

            points = X[cluster_group == i]

            for point in points:
                wcss += np.sum((point - self.centroids[i]) ** 2)

        return wcss