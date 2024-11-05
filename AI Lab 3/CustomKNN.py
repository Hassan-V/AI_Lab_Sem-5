import pandas as pd
import numpy as np
from collections import Counter


class CustomKNN:

    def __init__(self, dataset: pd.DataFrame, target_attr: pd.Series, k: int = 3) -> None:
        """
        Initialize the CustomKNN with a dataset, target attribute, and value of k (number of neighbors).

        :param dataset: DataFrame containing feature data.
        :param target_attr: Series containing the target attribute (labels).
        :param k: Number of nearest neighbors to consider for classification.
        """
        self.dataset: pd.DataFrame = dataset
        self.target_attr: pd.Series = target_attr
        self.k = k

    def _euclidean_distance(self, row1: pd.Series, row2: pd.Series) -> float:
        """
        Calculate the Euclidean distance between two data points.

        :param row1: First data point (Series).
        :param row2: Second data point (Series).
        :return: Euclidean distance as a float.
        """
        return np.sqrt(np.sum((row1 - row2) ** 2))

    def predict(self, test_data: pd.DataFrame) -> pd.Series:
        predictions = []

        for _, test_sample in test_data.iterrows():
            distances = []

            for i, train_sample in self.dataset.iterrows():
                distance = self._euclidean_distance(train_sample, test_sample)
                distances.append((distance, self.target_attr.iloc[i]))

            distances.sort(key=lambda x: x[0])
            k_nearest_labels = [label for _, label in distances[:self.k]]

            print("Test sample:", test_sample.values)  # Debug: test sample values
            print("Distances:", distances[:self.k])  # Debug: top k distances
            print("Nearest labels:", k_nearest_labels)  # Debug: nearest labels

            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        return pd.Series(predictions, index=test_data.index)

    def score(self, test_data: pd.DataFrame, test_labels: pd.Series) -> float:
        """
        Calculate the accuracy of the classifier on the test dataset.

        :param test_data: DataFrame containing test samples (features only).
        :param test_labels: Series containing true labels for test samples.
        :return: Accuracy as a float.
        """
        predictions = self.predict(test_data)
        correct = (predictions == test_labels).sum()
        return correct / len(test_labels)
