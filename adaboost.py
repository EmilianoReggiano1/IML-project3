import numpy as np
from typing import Callable, NoReturn
from base_estimator import BaseEstimator
from loss_functions import misclassification_error

class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        samples_num, features_num = X.shape

        # Initialize uniform distribution
        D = np.ones(samples_num)/samples_num
        self.models_ = []
        self.weights_ = []
        # Iterate through number of learners
        for _ in range(self.iterations_):
            # Create and fit weak learner
            learner = self.wl_()

            # sample train examples according to D
            train_indices = np.random.choice(samples_num, size=samples_num, p=D, replace=True)
            X_train_for_learner = X[train_indices]
            y_train_for_learner = y[train_indices]
            learner.fit(X_train_for_learner, y_train_for_learner)

            y_pred = learner.predict(X)

            # Compute weighted error
            misclassified = (y_pred != y)
            weighted_error = np.sum(D * misclassified)

            # Compute learner weight
            w = 0.5 * np.log((1-weighted_error) / weighted_error)

            # Update sample weights and normalize
            D *= np.exp(-y * w * y_pred)
            D /= np.sum(D)

            # Save model and weight
            self.models_.append(learner), self.weights_.append(w)

        # save distribution
        self.D_ = D

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

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
        self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        prediction = np.zeros(X.shape[0])

        for t in range(T):
            model = self.models_[t]
            weight = self.weights_[t]
            prediction += model.predict(X) * weight
        final_prediction = np.sign(prediction)
        return final_prediction

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.partial_predict(X, T)
        return misclassification_error(y, y_pred)