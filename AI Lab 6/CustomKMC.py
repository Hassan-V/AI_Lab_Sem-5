import numpy as np

class CustomKMC:
    def __init__(self, K, max_iter=100, tol=1e-4):
        """
        Initialize the K-Means Clustering algorithm.

        Parameters:
        - K: Number of clusters
        - max_iter: Maximum number of iterations for convergence
        - tol: Tolerance for convergence (when centroids do not change significantly)
        """
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        """ Randomly initialize centroids from the data points """
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(X.shape[0], size=self.K, replace=False)
        return X[indices]

    def compute_distance(self, X, centroids):
        """ Calculate Euclidean distance from each point to each centroid """
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return distances

    def assign_clusters(self, X, centroids):
        """ Assign each point to the nearest centroid """
        distances = self.compute_distance(X, centroids)
        return np.argmin(distances, axis=1)

    def compute_centroids(self, X, labels):
        """ Compute new centroids as the mean of all points assigned to each cluster """
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.K)])
        return new_centroids

    def fit(self, X):
        """
        Fit the K-Means model to the data.

        Parameters:
        - X: The dataset (2D numpy array or DataFrame converted to numpy array).

        This method runs the K-Means algorithm.
        """
        # Step 1: Initialize centroids
        self.centroids = self.initialize_centroids(X)
        prev_centroids = self.centroids.copy()

        for i in range(self.max_iter):
            # Step 2: Assign points to clusters
            self.labels = self.assign_clusters(X, self.centroids)

            # Step 3: Compute new centroids
            self.centroids = self.compute_centroids(X, self.labels)

            # Step 4: Check for convergence (if centroids do not change)
            if np.all(np.abs(self.centroids - prev_centroids) < self.tol):
                print(f"Convergence reached after {i + 1} iterations.")
                break

            prev_centroids = self.centroids.copy()

    def predict(self, X):
        """ Predict the cluster labels for the given data points """
        return self.assign_clusters(X, self.centroids)

    def get_centroids(self):
        """ Return the final centroids """
        return self.centroids

    def get_labels(self):
        """ Return the cluster labels for the dataset """
        return self.labels