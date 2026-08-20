# Third party imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pylab

# sklearn imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator as Estimator
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import *

# statsmodels imports
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Other local applications imports
from pandas import DataFrame, Series
from matplotlib.patches import Patch
from typing import Tuple
from scipy import stats

cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm


def split(data: DataFrame, train_size: int) -> Tuple:
    """
    Split arrays into train and test subsets.

    Parameters
    ----------
        data : ``pandas.DataFrame``
            The input data and the target values.

        train_size : int
            Size of the training sample.

    Returns
    -------
        x_train : ``pandas.DataFrame``
            The data matrix on train set.

        x_test : ``pandas.DataFrame``
            The data matrix on test set.

        y_train : ``pandas.Series``
            The target vector on train set.

        y_test : ``pandas.Series``
            The target vector on test set.
    """
    # the input (x) data and target values (y)
    x_features = data[data.columns[0:-1]]
    y_target = data[data.columns[-1]]

    # training and target vectors
    x_train = x_features[0:train_size]
    y_train = y_target[0:train_size].astype("float")

    # testing and target vectors
    x_test = x_features[train_size::]
    y_test = y_target[train_size::].astype("float")

    return x_train, x_test, y_train, y_test


def feature_selection(
    X: np.ndarray, y: np.ndarray, percentile: int = 10, method: str = "linear"
) -> None:
    """
    Feature selection/dimensionality reduction on sample sets.

    Select features according to a percentile of the
    highest scores using ``SelectPercentile``.

    References :

    - https://scikit-learn.org/stable/modules/feature_selection.html
    - ``class sklearn.feature_selection.SelectPercentile``

    Parameters
    ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        y : array-like of shape (n_samples,)
            The target vector.

        percentile : int, optional, Defaults to 10
            Percent of features to keep.

        method : str, optional, Defaults to 'linear'
            Feature selection methods from ``sklearn.feature_selection``.
            The methods based on F-test estimate
            the degree of linear dependency.
            Mutual information methods can capture
            any kind of statistical dependency.

        - 'linear' stands for ``f_regression``
        - 'non-linear' stands for ``mutual_info_regression``
    """
    if method == "linear":
        scoring = f_regression(X, y)
        scoring = pd.Series(scoring[0])
        selected_top_columns = SelectPercentile(f_regression, percentile=percentile)
    else:
        scoring = mutual_info_regression(X, y)
        scoring = pd.Series(scoring)
        selected_top_columns = SelectPercentile(
            mutual_info_regression, percentile=percentile
        )

    scoring.index = X.columns
    scoring.sort_values(ascending=False).plot.bar(figsize=(15, 5))
    selected_top_columns.fit(X, y)
    print(
        f"Features selected ({percentile} %) :\n\n \
        {X.columns[selected_top_columns.get_support()]}"
    )


def recursive_selection(X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """
    Recursive feature elimination with cross-validation to
    select the number of features using ``sklearn.feature_selection.RFECV``.

    References :

    - https://scikit-learn.org/stable/modules/feature_selection.html#rfe
    - ``class sklearn.feature_selection.RFECV``

    Parameters
    ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The data matrix.

        y : array-like of shape (n_samples,)
            The target vector.

        estimator : ``Estimator`` instance
            Supervised learning estimator with a fit method that provides
            information about feature importance.

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
    # Recursive feature elimination with cross-validation
    rfecv = RFECV(**kwargs).fit(X, y)
    # select the number of features
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    # plt.plot(
    #     range(1, len(rfecv.cv_results_["split(k)_test_score"]) + 1),
    #     rfecv.cv_results_["split(k)_test_score"],
    # )
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.errorbar(
        range(1, n_scores + 1),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
    )
    plt.show()

    # The feature ranking
    # Selected features are assigned rank 1
    ranking = pd.DataFrame(rfecv.ranking_, index=X.columns, columns=["Rank"])
    print(ranking.sort_values(by="Rank", ascending=True))


def plot_cv_indices(
    X: np.ndarray,
    y: np.ndarray,
    cv: TimeSeriesSplit,
    n_splits: int = 5,
    date_col: Series = None,
) -> None:
    """
    Create a sample plot for indices of a ``cross-validation`` object.

    Parameters
    ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The data matrix.

        y : array-like of shape (n_samples,)
            The target vector.

        cv : ``sklearn.model_selection.TimeSeriesSplit``
            Time Series cross-validator.

        n_splits : int, optional, Defaults to 5
            Number of splits. Must be at least 2.

        date_col : ``pandas.Series``
            Sample to which set as x-axis.
    """
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    fig.tight_layout()

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=10,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    if date_col is not None:
        tick_locations = ax.get_xticks()
        ticks_list = list(tick_locations[1:-1])

        tick_dates = [" "] + date_col.iloc[ticks_list].astype(str).tolist() + [" "]

        tick_locations_str = [str(int(i)) for i in tick_locations]

        new_labels = ["\n\n".join(x) for x in zip(list(tick_locations_str), tick_dates)]

        ax.set_xticks(tick_locations)
        ax.set_xticklabels(new_labels)

    # set y and x axis
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 0.2, -0.2],
    )

    # place a legend on the Axes
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )

    # add a title to the Axes
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)


def normality(data: DataFrame, feature: str) -> None:
    """
    Function to return plots for the feature.

    Check if data is normally distributed by a QQPlot.

    Parameters
    ----------
        data : ``pandas.DataFrame``
            The input data and the target values.

        feature : str
            Feature to which analyze data distribution.
    """
    # Create a new figure
    plt.figure(figsize=(20, 5))
    # Add a title to the figure
    plt.title(feature)
    # Add an Axes to the current figure
    plt.subplot(1, 2, 1)
    # Plot univariate or bivariate distributions
    # using kernel density estimation
    sns.kdeplot(data[feature])
    # Add a title to the figure
    plt.subplot(1, 2, 2)
    # Calculate quantiles for a probability plot
    stats.probplot(data[feature], plot=pylab)
    plt.show()


def autocorrelation(data: np.ndarray, lags: int = 12) -> None:
    """
    Plot both autocorrelation and partial autocorrelation functions.

    Autocorrelation is the correlation between a
    time series' current value with past values.

    References:

    - https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

    - https://neptune.ai/blog/select-model-for-time-series-prediction-task

    - https://www.statsmodels.org/devel/generated/statsmodels.graphics.tsaplots.plot_acf.html

    Parameters
    ----------
        data : array-like
            Array of time-series values.

        lags : int, optional, Defaults to 12
            An int or array of lag values, used on horizontal axis.
    """
    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
    fig.tight_layout()

    # Plot the autocorrelation function
    plot_acf(data, lags=lags, ax=ax1)
    # Adding plot title
    ax1.set_title("Autocorrelation Plot")

    # Plot the partial autocorrelation function
    plot_pacf(data, lags=lags, ax=ax2)
    # Adding plot title
    ax2.set_title("Partial Autocorrelation Plot")

    # Providing x-axis name
    ax1.set_xlabel("Lags")
    ax2.set_xlabel("Lags")
    # set the current tick locations
    ax1.set_xticks(np.arange(0, lags + 1, 1))
    ax2.set_xticks(np.arange(0, lags + 1, 1))

    # Display figure
    plt.show()


def seasonal_decomposition(
    data: np.ndarray, plot: bool = True, **kwargs
) -> DecomposeResult:
    """
    Seasonal decomposition using moving averages.

    Parameters
    ----------
        data : array-like
            Array of time-series values.

        plot : bool, optional, Defaults to True
            Plot the decomposition (observed, trend, seasonal, resids).

        **kwargs from ``statsmodels.tsa.seasonal.seasonal_decompose``

    Returns
    -------
        decomposition : ``statsmodels.tsa.seasonal.DecomposeResult``
            A object with seasonal, trend, and resid attributes.
    """
    # A object with seasonal, trend, and resid attributes
    decomposition = seasonal_decompose(
        x=data, model="additive", extrapolate_trend="freq", **kwargs
    )
    if plot is True:
        # Create a figure and a set of subplots
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(24, 12))
        fig.tight_layout()
        # Plotting seasonal, trend, and resid attributes
        decomposition.observed.plot(ax=ax0, ylabel="observed")
        decomposition.trend.plot(ax=ax1, ylabel="trend")
        decomposition.seasonal.plot(ax=ax2, ylabel="seasonal")
        decomposition.resid.plot(ax=ax3, ylabel="resid")
        plt.show()
    else:
        pass

    return decomposition


def adf_test(data: np.ndarray) -> None:
    """
    Check if Time Series Data is Stationary by
    applying the ``Augmented Dickey-Fuller test``.

    The Augmented Dickey-Fuller test can be used to test for a
    unit root in a univariate process in the presence of serial correlation.

    The intuition behind a unit root test is that it determines
    how strongly a time series is defined by a trend.

    Reference :

    https://machinelearningmastery.com/time-series-data-stationary-python/

    Parameters
    ----------
        data : array-like
            Array of time-series values.
    """
    # Augmented Dickey-Fuller test
    dftest = adfuller(data.dropna(), autolag="AIC")
    print(" > Is the raw data stationary ?")

    # The test statistic
    print("Test statistic = {:.3f}".format(dftest[0]))
    # MacKinnon's approximate p-value based on MacKinnon (1994, 2010)
    print("P-value = {:.3f}".format(dftest[1]))

    # Critical values for the test statistic at the 1 %, 5 %, and 10 % levels.
    # Based on MacKinnon (2010)
    print("Critical values :")
    for k, v in dftest[4].items():
        print(
            "\t{}: {} - The data is {} stationary with {}% confidence".format(
                k, v, "not" if v < dftest[0] else "", 100 - int(k[:-1])
            )
        )


def features_correlation(
    data: DataFrame, transform: bool = True, method: str = "pearson"
) -> None:
    """
    Compute pairwise correlation of columns.

    Correlation is a bivariate analysis that measures the strength of
    association between two variables and the direction of the relationship.

    Parameters
    ----------
        data : ``pandas.DataFrame``
            The input data and the target values.

        transform : bool, optional, Defaults to True.
            If True, apply a ``Power Transformer`` featurewise to
            make data more ``Gaussian-like``.
            For ``Pearson`` correlations it is easy to
            calculate and interpret when both
            variables have a well understood Gaussian distribution.

        method : str, optional, {'pearson', 'kendall', 'spearman'}
            Method of correlation:

        - ``pearson`` : standard correlation coefficient

                Measure the degree of the relationship between
                linearly related variables

        - ``kendall`` : Kendall Tau correlation coefficient

                Non-parametric test that measures the strength of
                dependence between two variables.

        - ``spearman`` : Spearman rank correlation

                Non-parametric test that is used to measure the degree of
                association between two variables.
    """
    if transform is True:
        # apply a power transform featurewise to make data more Gaussian-like
        trans = PowerTransformer()
        transformed = trans.fit_transform(data)
        # convert the fitted array back to a dataframe
        dataset = pd.DataFrame(transformed)
        dataset.columns = data.columns
    else:
        dataset = data

    # calculate correlations
    corr = dataset.corr(method=method)
    # Create a figure and a set of subplots
    f, ax = plt.subplots(figsize=(20, 10))
    # Plot heatmap (color-encoded matrix) to compute correlations
    sns.heatmap(
        corr,
        cmap="coolwarm_r",
        annot=True,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        fmt=".2f",
    )
