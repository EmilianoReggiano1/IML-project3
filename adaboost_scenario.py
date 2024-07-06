# import numpy as np
# from typing import Tuple
# from utils import *
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
#
# from adaboost import AdaBoost
# from decision_stump import DecisionStump
#
# def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Generate a dataset in R^2 of specified size
#
#     Parameters
#     ----------
#     n: int
#         Number of samples to generate
#
#     noise_ratio: float
#         Ratio of labels to invert
#
#     Returns
#     -------
#     X: np.ndarray of shape (n_samples,2)
#         Design matrix of samples
#
#     y: np.ndarray of shape (n_samples,)
#         Labels of samples
#     """
#     '''
#     generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
#     num_samples: the number of samples to generate
#     noise_ratio: invert the label for this ratio of the samples
#     '''
#     X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
#     y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
#     y[np.random.choice(n, int(noise_ratio * n))] *= -1
#     return X, y
#
#
# def plot_question_1(adaboost, train_error, test_error, noise):
#     """ Plot the training- and test errors as a function of the number of fitted learners in the same plot """
#     fig = make_subplots(rows=1, cols=1)
#     fig.add_trace(go.Scatter(x=np.arange(1, adaboost.iterations_ + 1), y=train_error, mode='lines', name='Training error'), row=1, col=1)
#     fig.add_trace(go.Scatter(x=np.arange(1, adaboost.iterations_ + 1), y=test_error, mode='lines', name='Test error'), row=1, col=1)
#     fig.update_layout(title=f"<b>Training and test errors vs. number of learners, noise: {noise}</b>", xaxis_title="Number of learners", yaxis_title="Mislassification Error")
#     fig.write_image(f'train_test_errors_noise_{noise}.png')
#
#
# def plot_question_2(adaboost, T, lims, X_test, y_test):
#     """ Plot the decision surfaces of the AdaBoost model at the specified iterations """
#     for t in T:
#         fig = go.Figure()
#         fig.add_trace(decision_surface(lambda X: adaboost.partial_predict(X, t), lims[0], lims[1]))
#         fig.add_trace(go.Scatter(x=X_test[y_test == 1, 0], y=X_test[y_test == 1, 1], mode='markers',  marker=dict(color='blue')))
#         fig.add_trace(go.Scatter(x=X_test[y_test == -1, 0], y=X_test[y_test == -1, 1], mode='markers',  marker=dict(color='red')))
#         fig.update_layout(title=f"<b>Decision surface of AdaBoost with {t} learners</b>", height=800, width=800)
#         fig.show()
#
#
# def plot_question_3(adaboost, best_learner, lims, X_test, y_test, accuracy):
#     """ Plot the decision surface of the best performing ensemble """
#     fig = go.Figure()
#     fig.add_trace(decision_surface(lambda X: adaboost.partial_predict(X, best_learner), lims[0], lims[1]))
#     fig.add_trace(go.Scatter(x=X_test[y_test == 1, 0], y=X_test[y_test == 1, 1], mode='markers', marker=dict(color='blue')))
#     fig.add_trace(go.Scatter(x=X_test[y_test == -1, 0], y=X_test[y_test == -1, 1], mode='markers', marker=dict(color='red')))
#     fig.update_layout(title=f"<b>Decision surface of optimal AdaBoost with {best_learner} learners, accuracy: {accuracy}</b>", height=800, width=800)
#     fig.show()
#
#
# def plot_question_4(adaboost, X_train, y_train, D, lims):
#     """ Plot the decision surface of the AdaBoost model with weighted samples """
#     fig = go.Figure()
#     fig.add_trace(decision_surface(lambda X: adaboost.predict(X), lims[0], lims[1]))
#     fig.add_trace(go.Scatter(x=X_train[:, 0], y=X_train[:, 1], mode="markers", marker=dict(color=y_train, size=5*D / np.max(D))))
#     fig.update_layout(title="<b>final AdaBoost Decision surface with training set points </b>", height=800, width=800)
#     fig.show()
#
#
#
# def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
#     (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
#
#     # Question 1: Train- and test errors of AdaBoost in noiseless case
#     adaboost = AdaBoost(DecisionStump, n_learners)
#
#     # Fit model
#     adaboost.fit(train_X, train_y)
#
#     # Calculate training- and test errors per learner number
#     train_error_per_learners = []
#     test_error_per_learners = []
#
#     for t in range(1, n_learners + 1):
#         train_error_per_learners.append(adaboost.partial_loss(train_X, train_y, t))
#         test_error_per_learners.append(adaboost.partial_loss(test_X, test_y, t))
#
#
#     # Plot, in a single figure, the training- and test errors as a function of the number of fitted learners (i.e. use the partial_loss function for t = 1,...,250).
#     plot_question_1(adaboost, train_error_per_learners, test_error_per_learners, noise)
#
#
#     # Question 2: Plotting decision surfaces
#     T = [5, 50, 100, 250]
#     lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
#     # plot_question_2(adaboost, T, lims, X_test, y_test)
#
#
#     # Question 3: Decision surface of best performing ensemble
#     best_learner = np.argmin(test_error_per_learners)
#     accuracy = 1 - test_error_per_learners[best_learner]
#     # plot_question_3(adaboost, best_learner, lims, X_test, y_test, accuracy)
#
#     # Question 4: Decision surface with weighted samples
#     D = adaboost.D_
#     # plot_question_4(adaboost, X_train, y_train, D, lims)
#
# if __name__ == '__main__':
#     np.random.seed(0)
#     fit_and_evaluate_adaboost(0.4)

import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump
from loss_functions import misclassification_error

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

def test_adaboost(fitted_model, X_train, y_train, X_test, y_test, noise):
    """
    Test AdaBoost on training and testing data for different number of learners and plot the results
    noise is used only for plot title
    """
    test_errs = []
    train_errs = []
    for i in range(1, fitted_model.iterations_ + 1):
        test_errs.append(fitted_model.partial_loss(X_test, y_test, i))
        train_errs.append(fitted_model.partial_loss(X_train, y_train, i))
    # plot the training and testing errors in a single plot
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=np.arange(1, fitted_model.iterations_ + 1), y=train_errs, mode='lines', name='Train Error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(1, fitted_model.iterations_ + 1), y=test_errs, mode='lines', name='Test Error'), row=1, col=1)
    fig.update_layout(title=f'Train and Test Errors for noise = {noise}', xaxis_title='Number of Learners', yaxis_title='Misclassification Error')
    #fig.write_html(f'train_test_errors_noise_{noise}.html')
    # save as png
    fig.write_image(f'train_test_errors_noise_{noise}.png')
    return test_errs, train_errs


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    test_errs, train_errs = test_adaboost(model, train_X, train_y, test_X, test_y, noise)


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2)
    for i, t in enumerate(T):
        predict_t = lambda x: model.partial_predict(x, t)
        fig.add_trace(decision_surface(predict_t, lims[0], lims[1], showscale=False), row=(i // 2) + 1, col=(i % 2) + 1)
        # add title for T
        fig.update_xaxes(title_text=f'T={t}', row=(i // 2) + 1, col=(i % 2) + 1)
        # add test set with coloring for labels in each subplot
        fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', marker=dict(color=test_y, colorscale=class_colors(2), symbol=class_symbols[1]), showlegend=False), row=(i // 2) + 1, col=(i % 2) + 1)
    fig.update_layout(title=f'Decision Surfaces of AdaBoost for different number of learners', showlegend=False, height=800, width=800)
    #fig.write_html(f'decision_surfaces_noise_{noise}.html')
    # save as png
    fig.write_image(f'decision_surfaces_noise_{noise}.png')


    # Question 3: Decision surface of best performing ensemble
    best_num_learners = np.argmin(test_errs) + 1
    best_predict = lambda x: model.partial_predict(x, best_num_learners)
    best_acc = 1 - model.partial_loss(test_X, test_y, best_num_learners)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(decision_surface(best_predict, lims[0], lims[1], showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', marker=dict(color=test_y, colorscale=class_colors(2), symbol=class_symbols[1]), showlegend=False), row=1, col=1)
    fig.update_layout(title=f'Decision Surface of best AdaBoost with T={best_num_learners} and Test Acc={best_acc:.6f}', showlegend=False, height=800, width=800)
    #fig.write_html(f'best_decision_surface_noise_{noise}.html')
    # save as png
    fig.write_image(f'best_decision_surface_noise_{noise}.png')

    # Question 4: Decision surface with weighted samples
    D = model.D_
    D = 15*D / np.max(D) # changed to scale by 20 because 5 gave very small points
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(decision_surface(model.predict, lims[0], lims[1], showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers', marker=dict(color=train_y,
                colorscale=class_colors(2), symbol=class_symbols[1], opacity=1, line=dict(width=0), size=D),
                                                                            showlegend=False), row=1, col=1)
    fig.update_layout(title=f'Decision Surface of AdaBoost with weighted samples for noise = {noise}', showlegend=False, height=800, width=800)
    #fig.write_html(f'weighted_samples_noise_{noise}.html')
    # save as png
    fig.write_image(f'weighted_samples_noise_{noise}.png')


if __name__ == '__main__':
    np.random.seed(0)
    # fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)




