import numpy as np
import pandas as pd


class CustomNaiveBayes:
    """
    A custom implementation of the Naive Bayes classifier for categorical features.
    This classifier applies Laplace smoothing for handling unseen feature values.

    Attributes:
        class_prob (dict): Dictionary storing the prior probabilities of each class.
        class_feature_prob (dict): Dictionary storing conditional probabilities for each feature value given a class.
        classes (np.ndarray): Array of unique class labels.
        features (pd.Index): List of feature names.
    """

    def __init__(self) -> None:
        """
        Initializes the Naive Bayes classifier with empty dictionaries
        to store class probabilities and feature conditional probabilities.
        """
        self.class_prob = {}
        self.class_feature_prob = {}
        self.classes = []
        self.features = []

    def fit(self, X: pd.DataFrame, y: pd.Series, alpha=1) -> None:
        """
        Fits the Naive Bayes model to the given dataset.

        Parameters:
            X (pd.DataFrame): The feature data where each column is a feature, and each row is an observation.
            y (pd.Series): The target labels corresponding to each observation in X.
            alpha (int, optional): The smoothing parameter for Laplace smoothing. Defaults to 1.

        Sets:
            self.class_prob: The prior probabilities for each class.
            self.class_feature_prob: The conditional probabilities for each feature given each class,
            including a probability for unseen feature values based on Laplace smoothing.

        Example:
            model = CustomNaiveBayes()
            model.fit(X_train, y_train, alpha=1)
        """
        self.classes = np.unique(y)
        self.features = X.columns

        for c in self.classes:
            self.class_prob[c] = np.sum(y == c) / len(y)  # Prior probability of class c
            self.class_feature_prob[c] = {}

            for f in self.features:
                self.class_feature_prob[c][f] = {}

                feature_values, counts = np.unique(X[f], return_counts=True)
                total_count = np.sum(y == c) + alpha * len(feature_values)  # Total count with smoothing

                for v in feature_values:
                    count = np.sum((X[f] == v) & (y == c)) + alpha  # Smoothed count
                    self.class_feature_prob[c][f][v] = count / total_count

                unseen_prob = alpha / total_count
                self.class_feature_prob[c][f]["unseen"] = unseen_prob

    def predict(self, X: pd.DataFrame) -> list:
        """
        Predicts the class labels for each instance in the given feature data.

        Parameters:
            X (pd.DataFrame): The feature data for which predictions are to be made.

        Returns:
            list: Predicted class labels for each instance in X.

        Example:
            y_pred = model.predict(X_test)
        """
        y_pred = []
        for i in range(len(X)):
            max_log_prob = -np.inf  # Initialize the max log probability
            max_class = None  # Initialize the best class

            for c in self.classes:
                log_prob = np.log(self.class_prob[c])  # Start with the log prior probability

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
