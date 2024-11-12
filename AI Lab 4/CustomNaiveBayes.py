import numpy as np


class CustomNaiveBayes:

    def __init__(self):
        self.class_prob = {}
        self.class_feature_prob = {}
        self.classes = []
        self.features = []

    def fit(self, X, y, alpha=1):
        self.classes = np.unique(y)
        self.features = X.columns

        for c in self.classes:
            self.class_prob[c] = np.sum(y == c) / len(y)
            self.class_feature_prob[c] = {}
            for f in self.features:
                self.class_feature_prob[c][f] = {}

                feature_values, counts = np.unique(X[f], return_counts=True)
                total_count = np.sum(y == c) + alpha * len(feature_values)

                for v in feature_values:
                    count = np.sum((X[f] == v) & (y == c)) + alpha  # Laplace smoothing
                    self.class_feature_prob[c][f][v] = count / total_count

                unseen_prob = alpha / total_count
                self.class_feature_prob[c][f]["unseen"] = unseen_prob

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            max_log_prob = -np.inf
            max_class = None
            for c in self.classes:
                log_prob = np.log(self.class_prob[c])
                for f in self.features:
                    feature_value = X[f].iloc[i]
                    if feature_value in self.class_feature_prob[c][f]:
                        log_prob += np.log(self.class_feature_prob[c][f][feature_value])
                    else:
                        log_prob += np.log(self.class_feature_prob[c][f]["unseen"])
                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    max_class = c
            y_pred.append(max_class)
        return y_pred
