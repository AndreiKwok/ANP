# Standard imports
import pickle
import time

# Third party imports
import sklearn.preprocessing as Preprocessing
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# sklearn imports
from sklearn.base import BaseEstimator as Estimator
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline

# Plotting imports
from matplotlib.axes import Axes
from jupyterthemes import jtplot

# Local application imports
from . import evaluation

jtplot.style(context="notebook", ticks=True, grid=False)



def add_metrics_to_legend(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          ax: Axes,
                          **kwargs) -> None:
    """
    Add regression metrics to the axes legend.

    Parameters
    ----------
        y_true : array-like of shape (n_samples,)
            The target vector.

        y_pred : array-like of shape (n_samples,)
        Predicted target vector.

        ax : ``matplotlib.axes.Axes`` or array of axes.
        Axes for the subplot.

        **kwargs from ``matplotlib.pyplot.legend``
    """
    # Metrics to evaluate regression models
    metrics = evaluation.regression_metrics(y_true=y_true, y_pred=y_pred)
    # Add scores to legend
    scores = (
        r"$R^2={:.2f}$"
        + "\n"
        # + r"$MSE={:.2f}$"
        # + "\n"
        + r"$RMSE={:.2f}$"
        + "\n"
        + r"$MAE={:.2f}$"
        + "\n"
        + r"$MAPE={:.2f}$"
    ).format(
        metrics['R²'],
        # metrics['MSE'],
        metrics['RMSE'],
        metrics['MAE'],
        metrics['MAPE (%)']
    )
    # Rectangle patch to place legend
    extra = plt.Rectangle(
                    (0, 0), 0, 0,
                    fc="w",
                    fill=False,
                    edgecolor="none",
                    linewidth=0
                )
    ax.legend([extra], [scores], **kwargs)


def lineplot(X: np.ndarray,
             y_true: np.ndarray,
             y_pred: np.ndarray,
             ax: Axes,
             **kwargs) -> None:
    """
    Lineplot of the predicted x observed values.

    Parameters
    ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        y_true : array-like of shape (n_samples,)
            The target vector.

        y_pred : array-like of shape (n_samples,)
            Predicted target vector.

        ax : ``matplotlib.axes.Axes`` or array of axes.
            Axes for the subplot.

         **kwargs from ``matplotlib.pyplot.legend``
    """
    # Observed values
    ax.plot(X.index, y_true, label='Observed')
    # Predicted values
    ax.plot(
        X.index,
        y_pred,
        label='Predicted',
        color='orange',
        ls='--'
    )
    # Place a legend on the Axes
    add_metrics_to_legend(y_true=y_true, y_pred=y_pred, ax=ax, **kwargs)


def scatterplot(y_true: np.ndarray,
                y_pred: np.ndarray,
                ax: Axes,
                **kwargs) -> None:
    """
    Scatter plot of the predicted x target vectors.

    Parameters
    ----------
        y_true : array-like of shape (n_samples,)
            The target vector.

        y_pred : array-like of shape (n_samples,)
            Predicted target vector.

        ax : ``matplotlib.axes.Axes`` or array of axes.
            Axes for the subplot.

         **kwargs from ``matplotlib.pyplot.legend``
    """
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()], linewidth=2, color='b')

    ax.scatter(y_true, y_pred, alpha=0.5, color='orange')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])

    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")

    # Place a legend on the Axes
    add_metrics_to_legend(y_true=y_true, y_pred=y_pred, ax=ax, **kwargs)


def regression_plot(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    ax: Axes) -> None:
    """
    Calculate and plot Linear least-squares for y_pred vs y_true.

    Parameters
    ----------
        y_true : array-like of shape (n_samples,)
            The target vector.

        y_pred : array-like of shape (n_samples,)
            Predicted target vector.

        ax : ``matplotlib.axes.Axes`` or array of axes.
            Axes for the subplot.
    """
    # outputs from Scipy's linear regression
    slope, intercept, r, p, stderr = scipy.stats.linregress(x=y_pred, y=y_true)
    # regression line equation
    line = f'Regression line: y = {intercept:.2f} + {slope:.2f}x'

    # Plot linear least-squares regression line
    ax.plot(
        y_pred,
        intercept + slope * y_pred,
        label=line,
        c="b",
        ls='--',
        lw=1
    )
    # Scatter plot of observed x predicted values
    ax.scatter(x=y_pred, y=y_true, c='orange', s=100)

    # x and y-axis layout
    ax.set_xlabel('Predicted', labelpad=20, style='oblique')
    ax.set_ylabel('Observed', labelpad=15, style='oblique')

    # plotting Pearson Coefficient
    ax.annotate(
        f'Pearson Coefficient : {round(r, 2)}',
        xy=(0.08, 0.92),
        xytext=(0, -30),
        xycoords='axes fraction',
        textcoords='offset points',
        style='oblique',
        color='w',
        weight='bold',
        size=15,
        bbox=dict(
                boxstyle="round",
                fc="#125D98",
                ec="#f7f3e9"
            ),
        arrowprops=dict(
                    arrowstyle="wedge,tail_width=1.",
                    fc="#125D98",
                    ec="#f7f3e9",
                    patchA=None,
                    relpos=(0.2, 0.5),
                    connectionstyle="arc3,rad=-0.1"
                )
    )
    # customized legend icon
    reg_label = mlines.Line2D([], [], color="b", ls='--', lw=2, label=line)
    # customized legend
    ax.legend(
        handles=[reg_label],
        loc='upper left',
        ncol=1,
        prop={'style':'normal', 'size':13}
    )


def cross_val_scores_boxplot(data: np.ndarray,
                             labels: list = None) -> None:
    """
    Draw a box and whisker plot for the cross validation scores.

    Parameters
    ----------
        data : array-like
            The input data.

        labels : list
            Labels for each dataset (one per dataset).
    """
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(20, 10))

    # A dictionary mapping each component of the boxplot
    # to a list of the .Line2D instances created
    flier = dict(markerfacecolor='orange', marker='D', markersize=12)
    median = dict(linewidth=5, color='orange')
    whisker = dict(linewidth=2.5, color='#F9E4C8')
    mean = dict(markerfacecolor='green', marker='D', markersize=8)

    # Draw a box and whisker plot for the cross validation scores
    bplots = ax.boxplot(x=data,
                        labels=labels,
                        flierprops=flier,
                        patch_artist=True,
                        medianprops=median,
                        whiskerprops=whisker,
                        capprops=whisker,
                        showmeans=True,
                        meanprops=mean)

    # customized patches
    colors = ['#F9E4C8' for i in range(0, len(data))]
    for patch, color in zip(bplots['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor('black')

    # horizontal grid lines
    ax.yaxis.grid(True)

    # add a title to the Axes
    plt.title('Algorithm Comparison')
    plt.show()


def plot_cross_val_scores(models: list,
                          X: np.ndarray,
                          y: np.ndarray,
                          scaler: Preprocessing = None,
                          cv = None,
                          scoring = None) -> None:
    """
    Draw a box and whisker plot to evaluate scores by cross-validation.

    Parameters
    ----------
        models : list
            Models to evaluate.

        X : array-like of shape (n_samples, n_features)
            The data matrix.

        y : array-like of shape (n_samples,)
            The target vector.

        scaler : ``sklearn.preprocessing``, optional, Defaults to None
            Utility functions or transformer classes from sklearn.preprocessing.

        cv : int, cross-validation generator or an iterable, optional, Defaults to None.
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - ``CV splitter``
        - An iterable yielding (train, test) splits as arrays of indices.

        scoring : str, callable, optional, Defaults to None
            A string (see https://scikit-learn.org/stable/modules/model_evaluation.html) or
            a scorer callable object / function with signature scorer(estimator, X, y).
    """
    results, names = [], []

    for name, model in models:
        print("name", name)
        pipeline = make_pipeline(scaler, model)

        cv_results = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    cross_val_scores_boxplot(data=results, labels=names)


def plot_stacking_models(X: np.ndarray,
                         y: np.ndarray,
                         estimator: StackingRegressor,
                         shift: int,
                         kind: str = 'scatter',
                         save_pickle: bool = False) -> None:
    """
    Scatter plot of the predicted vs true targets for each model in 'estimators'.

    Also plot the scatter for the stack of estimators with the final regressor.

    Parameters
    ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        y : array-like of shape (n_samples,)
            The target vector.

        estimator : ``StackingRegressor``
            Stack of estimators with a final regressor.

        shift : int
            Number of periods to shift.
            This parameter sets the range of the predicted month.

        kind : str in {'scatter', 'lineplot'}, optional, Defaults to 'scatter'
            Set the kind of the plot.

        save_pickle : bool, optional, Defaults to False
            If True, write fitted ``estimator`` to a pickle (.sav).
            Note.: All the base estimators and the stacked regressor
            will be written to a pickle.
    """
    # Set the number of subplots according to the number of stacked models
    nrows = int((len(estimator.estimators)+1)/2)
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 15))

    for ax, (name, est) in zip(
                            np.ravel(axs),
                            estimator.estimators + [("Stacking", estimator)]
                        ):
        # calculate elapsed time to fit the model
        start_time = time.time()
        # fit model
        est.fit(X, y)
        # elapsed time
        elapsed_time = time.time() - start_time
        # predictions from estimator
        y_pred, y_true = est.predict(X), y
        # evaluate regression model
        metrics = evaluation.regression_metrics(
                                    y_true=y_true,
                                    y_pred=y_pred
                                )
        print(name, metrics)

        if kind == 'lineplot':
            # Lineplot of the correct x estimated target values
            lineplot(X, y_true, y_pred, ax, loc='upper left')
        elif kind == 'scatter':
            # Scatter plot of the predicted x target vectors.
            scatterplot(y_true, y_pred, ax, loc='upper left')

        # Add title to subplot
        title = name + f"\n Evaluation in {round(elapsed_time, 2)} s"
        ax.set_title(title)
        # Save pickle
        if save_pickle:
            # Folder to which save the pickle files
            PICKLES_DIR = r'..\data\pickles'
            pickle.dump(
                est,
                open(f'{PICKLES_DIR}/{name}_m{shift}.sav', 'wb')
            )
    fig.suptitle(f"{shift} month(s)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)


def plot_error_distribution(X: np.ndarray,
                            y: np.ndarray,
                            estimator: StackingRegressor) -> None:
    """
    Plot the distribution of the errors averaged from the ``estimator``.

    Note.: The errors must follow a normal distribution shape. It indicates
    that the ``estimator`` is performing well.

    Parameters
    ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        y : array-like of shape (n_samples,)
            The target vector.

        estimator : ``StackingRegressor``
            Stack of estimators with a final regressor.
    """
    # Set the number of subplots according to the number of stacked models
    nrows = int((len(estimator.estimators)+1)/2)
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 15))

    for ax, (name, est) in zip(
                            np.ravel(axs),
                            estimator.estimators + [("Stacking", estimator)]
                        ):
        # fit model
        est.fit(X, y)
        # predictions from estimator
        y_pred, y_true = est.predict(X), y
        # calculate error
        error = y_true - y_pred
        # plot the distribution of error
        ax.hist(error)
        # Add title to subplot
        ax.set_title(name)

    fig.suptitle(f"Error distribution")
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)


def plot_model_performance(X: np.ndarray,
                           y: np.ndarray,
                           estimator: Estimator,
                           title: str,
                           add_residuals: bool = False) -> None:
    """
    Plot model performance on train and test set.

    Parameters
    ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        y : array-like of shape (n_samples,)
            The target vector.

        estimator : ``Estimator`` instance
            Supervised learning estimator with a fit method that provides
            information about feature importance.
    """
    # Predicted values
    y_pred = estimator.predict(X)
    # Calculate residuals on predictions
    if add_residuals:
        residuals = (y - y_pred)
        y_pred += residuals.mean()
    # Evaluate model
    print(evaluation.regression_metrics(y_true=y, y_pred=y_pred))

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 15))
    fig.suptitle(title, style='oblique')
    fig.tight_layout()

    # Lineplot of the predicted x observed values
    lineplot(X=X, y_true=y, y_pred=y_pred, ax=ax1, loc='upper left')

    # linear least-squares regression plot
    regression_plot(y_true=y, y_pred=y_pred, ax=ax2)


def plot_benchmark_performance(X: np.ndarray,
                               y: np.ndarray,
                               title: str,
                               benchmark_col: str) -> None:
    """
    Plot model performance on train and test set.

    Parameters
    ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        y : array-like of shape (n_samples,)
            The target vector.
    """
    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 15))
    fig.suptitle(title, style='oblique')
    fig.tight_layout()

    # Evaluate model
    print(evaluation.regression_metrics(y_true=y, y_pred=X[benchmark_col]))

    # Lineplot of the predicted x observed values
    lineplot(X=X, y_true=y, y_pred=X[benchmark_col], ax=ax1, loc='upper left')

    # linear least-squares regression plot
    regression_plot(y_true=y, y_pred=X[benchmark_col], ax=ax2)
