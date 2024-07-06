# from __future__ import annotations
# from typing import Tuple, NoReturn
# from base_estimator import BaseEstimator
# import numpy as np
# from itertools import product
# from loss_functions import misclassification_error
#
#
# class DecisionStump(BaseEstimator):
#     """
#     A decision stump classifier for {-1,1} labels according to the CART algorithm
#
#     Attributes
#     ----------
#     self.threshold_ : float
#         The threshold by which the data is split
#
#     self.j_ : int
#         The index of the feature by which to split the data
#
#     self.sign_: int
#         The label to predict for samples where the value of the j'th feature is about the threshold
#     """
#     def __init__(self) -> DecisionStump:
#         """
#         Instantiate a Decision stump classifier
#         """
#         super().__init__()
#         self.threshold_, self.j_, self.sign_ = None, None, None
#
#     def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
#         """
#         Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Input data to fit an estimator for
#
#         y : ndarray of shape (n_samples, )
#             Responses of input data to fit to
#         """
#         num_samples, num_features = X.shape
#
#         # Initialize the best threshold, feature and sign
#         best_thr, best_j, best_sign = None, None, None
#         best_err = np.inf
#
#         # Iterate over all features
#         for j in range(num_features):
#             for sign in [-1, 1]:
#                 thr, err = self._find_threshold(X[:, j], y, sign)
#                 if err < best_err:
#                     best_thr, best_j, best_sign = thr, j, sign
#                     best_err = err
#
#         self.threshold_, self.j_, self.sign_ = best_thr, best_j, best_sign
#
#
#
#     def _predict(self, X: np.ndarray) -> np.ndarray:
#         """
#         Predict sign responses for given samples using fitted estimator
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Input data to predict responses for
#
#         y : ndarray of shape (n_samples, )
#             Responses of input data to fit to
#
#         Returns
#         -------
#         responses : ndarray of shape (n_samples, )
#             Predicted responses of given samples
#
#         Notes
#         -----
#         Feature values strictly below threshold are predicted as `-sign` whereas values which equal
#         to or above the threshold are predicted as `sign`
#         """
#         samples_num = X.shape[0]
#         y_pred = np.full(samples_num, self.sign_)
#         below_threshold = X[:, self.j_] < self.threshold_
#         y_pred[below_threshold] = -self.sign_
#         return y_pred
#
#
#     def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
#         """
#         Given a feature vector and labels, find a threshold by which to perform a split
#         The threshold is found according to the value minimizing the misclassification
#         error along this feature
#
#         Parameters
#         ----------
#         values: ndarray of shape (n_samples,)
#             A feature vector to find a splitting threshold for
#
#         labels: ndarray of shape (n_samples,)
#             The labels to compare against
#
#         sign: int
#             Predicted label assigned to values equal to or above threshold
#
#         Returns
#         -------
#         thr: float
#             Threshold by which to perform split
#
#         thr_err: float between 0 and 1
#             Misclassificaiton error of returned threshold
#
#         Notes
#         -----
#         For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
#         which equal to or above the threshold are predicted as `sign`
#         """
#
#         # Sort values
#         sorted_indices = np.argsort(values)
#         values = values[sorted_indices]
#         labels = labels[sorted_indices]
#
#         # Calculate the initial loss for classifying all as `sign`
#         initial_loss = np.sum(np.abs(labels)[np.sign(labels) == sign])
#
#         # Compute losses for each threshold
#         losses = np.append(initial_loss, initial_loss - np.cumsum(labels * sign))
#
#         # Find the index of the minimum loss
#         best_index = np.argmin(losses)
#
#         # Return the best threshold and its corresponding loss
#         thresholds = np.concatenate([[-np.inf], values[1:], [np.inf]])
#         return thresholds[best_index], losses[best_index]
#
#
#     def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
#         """
#         Evaluate performance under misclassification loss function
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Test samples
#
#         y : ndarray of shape (n_samples, )
#             True labels of test samples
#
#         Returns
#         -------
#         loss : float
#             Performance under missclassification loss function
#         """
#         y_pred = self.predict(X)
#         return misclassification_error(y, y_pred)
#
#
#
from __future__ import annotations
from typing import Tuple, NoReturn
from base_estimator import BaseEstimator
import numpy as np
from itertools import product
from loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None
        self.min_err = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Initialize minimum error and threshold
        min_err = 1
        # Iterate through all possible features and signs
        for j, sign in product(range(X.shape[1]), [-1, 1]):
            # Find threshold for current feature
            t, err = self._find_threshold(X[:, j], y, sign)
            # Update minimum error and threshold
            if err < min_err:
                min_err, self.threshold_, self.j_, self.sign_ = err, t, j, sign
        self.min_err = min_err

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold (scaled by weights of labels)

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        # Sort values and labels according to the sorted values in ascending order
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # initialize error to be predicting all as sign
        current_error = np.sum(np.abs(sorted_labels) * (np.sign(sorted_labels) != sign))
        best_error = current_error
        best_thresh = sorted_values[0] - 1
        for i in range(0, len(sorted_values) - 1):
            # we now predict the ith value as -sign as such if the ith value is sign we need to add it to the error, if it is -sign we need to subtract it
            if np.sign(sorted_labels[i]) == sign:
                current_error += np.abs(sorted_labels[i])
            else:
                current_error -= np.abs(sorted_labels[i])
            if current_error < best_error:
                best_error = current_error
                best_thresh = (sorted_values[i] + sorted_values[i + 1]) / 2
        # check if putting above last value is better
        current_error = np.sum(np.abs(sorted_labels) * (np.sign(sorted_labels) != -sign))
        if current_error < best_error:
            best_error = current_error
            best_thresh = sorted_values[-1] + 1

        return best_thresh, best_error

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))



