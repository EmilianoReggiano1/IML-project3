import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def plot_question_1(train_error, test_error):
    """ Plot the training- and test errors as a function of the number of fitted learners in the same plot """
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=np.arange(1, len(train_error) + 1), y=train_error, mode='lines', name='Training error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(1, len(test_error) + 1), y=test_error, mode='lines', name='Test error'), row=1, col=1)
    fig.update_layout(title="<b>Training- and test errors as a function of the number of fitted learners</b>", xaxis_title="Number of learners", yaxis_title="Error")
    fig.show()


def plot_question_2(adaboost, T, lims, X_test, y_test):
    """ Plot the decision surfaces of the AdaBoost model at the specified iterations """
    for t in T:
        fig = go.Figure()
        fig.add_trace(decision_surface(lambda X: adaboost.partial_predict(X, t), lims[0], lims[1]))
        fig.add_trace(go.Scatter(x=X_test[y_test == 1, 0], y=X_test[y_test == 1, 1], mode='markers',  marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=X_test[y_test == -1, 0], y=X_test[y_test == -1, 1], mode='markers',  marker=dict(color='red')))
        fig.update_layout(title=f"<b>Decision surface of AdaBoost with {t} learners</b>", height=800, width=800)
        fig.show()


def plot_question_3(adaboost, best_learner, lims, X_test, y_test, accuracy):
    """ Plot the decision surface of the best performing ensemble """
    fig = go.Figure()
    fig.add_trace(decision_surface(lambda X: adaboost.partial_predict(X, best_learner), lims[0], lims[1]))
    fig.add_trace(go.Scatter(x=X_test[y_test == 1, 0], y=X_test[y_test == 1, 1], mode='markers', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=X_test[y_test == -1, 0], y=X_test[y_test == -1, 1], mode='markers', marker=dict(color='red')))
    fig.update_layout(title=f"<b>Decision surface of optimal AdaBoost with {best_learner} learners, accuracy: {accuracy}</b>", height=800, width=800)
    fig.show()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    X_train, y_train = generate_data(train_size, noise)
    X_test, y_test = generate_data(test_size, noise)

    # Fit model
    adaboost.fit(X_train, y_train)

    # Calculate training- and test errors per learner number
    train_error_per_learners = []
    test_error_per_learners = []

    for t in range(1, n_learners + 1):
        train_error_per_learners.append(adaboost.partial_loss(X_train, y_train, t))
        test_error_per_learners.append(adaboost.partial_loss(X_test, y_test, t))


    # Plot, in a single figure, the training- and test errors as a function of the number of fitted learners (i.e. use the partial_loss function for t = 1,...,250).
    plot_question_1(train_error_per_learners, test_error_per_learners)


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[X_train, X_test].min(axis=0), np.r_[X_train, X_test].max(axis=0)]).T + np.array([-.1, .1])
    # plot_question_2(adaboost, T, lims, X_test, y_test)


    # Question 3: Decision surface of best performing ensemble
    best_learner = np.argmin(test_error_per_learners) + 1
    accuracy = 1 - test_error_per_learners[best_learner]
    plot_question_3(adaboost, best_learner, lims, X_test, y_test, accuracy)

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)