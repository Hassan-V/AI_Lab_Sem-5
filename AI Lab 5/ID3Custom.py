import numpy as np

class CustomID3:
    def __init__(self):
        self.tree = None
        self.most_frequent_label = None

    def entropy(self, data, target_index):
        """Calculate the entropy of a dataset."""
        labels, counts = np.unique(data[:, target_index], return_counts=True)
        probabilities = counts / counts.sum()
        return -sum(p * np.log2(p) for p in probabilities)

    def information_gain(self, data, split_index, target_index):
        """Calculate the information gain of a dataset split on a feature."""
        total_entropy = self.entropy(data, target_index)
        values, counts = np.unique(data[:, split_index], return_counts=True)

        weighted_entropy = 0
        for value, count in zip(values, counts):
            subset = data[data[:, split_index] == value]
            weighted_entropy += (count / len(data)) * self.entropy(subset, target_index)

        return total_entropy - weighted_entropy

    def find_best_split(self, data, feature_indices, target_index):
        """Find the best feature to split on."""
        gains = [
            (index, self.information_gain(data, index, target_index))
            for index in feature_indices
        ]
        return max(gains, key=lambda x: x[1])

    def build_tree(self, data, feature_indices, target_index):
        """Recursively build the decision tree."""
        if len(np.unique(data[:, target_index])) == 1:
            return data[0, target_index]

        if not feature_indices:
            target_values = data[:, target_index]
            return np.bincount(target_values).argmax()

        best_feature, _ = self.find_best_split(data, feature_indices, target_index)

        tree = {best_feature: {}}
        values = np.unique(data[:, best_feature])

        for value in values:
            subset = data[data[:, best_feature] == value]
            remaining_features = [i for i in feature_indices if i != best_feature]

            subtree = self.build_tree(subset, remaining_features, target_index)
            tree[best_feature][value] = subtree

        return tree

    def fit(self, data, target_index):
        """Fit the decision tree."""
        feature_indices = [i for i in range(data.shape[1]) if i != target_index]
        self.tree = self.build_tree(data, feature_indices, target_index)

        target_values = data[:, target_index].astype(int)
        self.most_frequent_label = np.bincount(target_values).argmax()

    def predict_one(self, tree, instance):
        """Predict the label for a single instance."""
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        value = instance[feature]

        if value not in tree[feature]:
            return self.most_frequent_label

        return self.predict_one(tree[feature][value], instance)

    def predict(self, data):
        """Predict labels for a dataset."""
        return np.array([self.predict_one(self.tree, instance) for instance in data])
