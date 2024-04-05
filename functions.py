import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.patches import Patch
from nltk.util import ngrams
from typing import List, Optional
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from scipy.stats import anderson, kstest
from typing import Optional, Union, Dict, List
from numpy import ndarray


def get_cols_with_missing_values(df: pd.DataFrame, threshold_prc: float) -> None:
    """
    Plots a bar chart showing columns with missing values exceeding a specified threshold.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        threshold_prc (float): The percentage threshold for missing values (e.g., 10.0 for 10%).

    Returns:
        None
    """
    nan_col = (df.isnull().sum() / len(df)) * 100
    nan_col = nan_col[nan_col > threshold_prc].sort_values()

    plt.figure(figsize=(20, 4))
    nan_col.plot(kind="bar")

    plt.title(
        f"List of Columns & NA percentages where NA values are more than {threshold_prc}%"
    )
    plt.xlabel("Features")
    plt.ylabel("Percentage of Missing Values")
    plt.show()

    
def format_labels(value: float) -> str:
    """
    Format a numerical value into a human-readable label.

    Args:
        value (float): The numerical value to format.

    Returns:
        str: The formatted label.
    """
    if value >= 1e6:
        return f"{value/1e6:.1f}M"
    elif value >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return str(value)

    
def plot_categorical_distribution_with_order(
    df: pd.DataFrame,
    category_column: str,
    title: str,
    x_label: str,
    y_label: str,
    category_order: Optional[List[str]] = None,
    rotation: int = 0,
    log_scale: bool = False
    ) -> None:
    """
    Plot the distribution of a categorical variable in a DataFrame with a specified order for categories.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the categorical variable.
        category_column (str): The column name of the categorical variable.
        title (str): The title for the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        category_order (List[str], optional): Order of categories for x-axis (default is None).
        rotation (int, optional): Rotation angle for x-axis labels (default is 0).
        log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis (default is False).
    """
    if category_order is not None:
        category_counts = (
            df[category_column].value_counts().reindex(category_order).reset_index()
        )
    else:
        category_counts = df[category_column].value_counts().reset_index()

    category_counts.columns = [category_column, "Count"]

    plt.figure(figsize=(10, 6))
    plt.bar(category_counts[category_column], category_counts["Count"])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=rotation)

    if log_scale:
        plt.yscale("log")

    for i, count in enumerate(category_counts["Count"]):
        plt.text(i, count, format_labels(count), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def plot_pie_distribution(var: pd.Series) -> None:
    """
    Plot a pie chart to visualize the distribution of labels in a variable using Seaborn.

    Args:
        var (pd.Series): The Pandas Series containing the labels to be visualized.

    Returns:
        None
    """
    label_counts = var.value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(label_counts, labels=label_counts.index, autopct="%.2f%%", startangle=140)
    plt.title("Distribution of Labels")
    plt.axis("equal")
    plt.show()


def train_test_sets(X: pd.DataFrame, label_col: str) -> tuple:
    """
    Split a DataFrame into training and testing sets for a binary classification problem.

    Parameters:
        X (pd.DataFrame): The input DataFrame containing features and labels.
        label_col: The column name of predictor column

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test.
    """
    labels = X[label_col]
    predictors = X.drop(label_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        predictors, labels, test_size=0.1, random_state=42
    )
    return X_train, X_test, y_train, y_test


def drop_rows_if(df: pd.DataFrame, perc: float) -> pd.DataFrame:
    """
    Drop rows from a DataFrame if the percentage of missing values in a row is greater than a threshold.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        perc (float): The threshold percentage for missing values in a row.

    Returns:
        pd.DataFrame: The DataFrame with rows dropped if they exceed the threshold.
    """
    df_na_counts = df.isna().sum(axis=1)
    df = df.drop(df_na_counts[df_na_counts > df.shape[1] * perc].index, axis=0)
    return df


def print_shape(df: pd.DataFrame) -> None:
    """
    Print the number of entries (rows) and features (columns) in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None
    """
    print(f"There are {df.shape[0]} entries and {df.shape[1]} features")


def print_n_duplicates_missing(df: pd.DataFrame) -> None:
    """
    Print the number of duplicate rows and the total count of missing (null) values in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None
    """
    duplicate_count = df.duplicated().sum()
    missing_count = df.isna().sum().sum()
    print(f"There are {duplicate_count} duplicates.\nNull values: {missing_count}")


def get_significance(p: float) -> str:
    """
    Returns a string indicating if H0 is rejected or not, comparing a given p-value to alpha (0.05).

    Args:
        p (float): The p-value from a statistical test.

    Returns:
        str: A string indicating whether to reject or fail to reject the null hypothesis.
    """
    if p <= 0.05:
        return "P value is below alpha 0.05 --> Reject H0."
    elif p > 0.05:
        return "P value is above alpha 0.05 --> Fail to reject H0"


def remove_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame based on a specified column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column to remove outliers from.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    df.reset_index(inplace=True, drop=True)
    start_shape = df.shape
    print("Old Shape: ", start_shape)

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    upper_array = np.where(df[col] >= upper)[0]
    lower_array = np.where(df[col] <= lower)[0]

    df.drop(index=upper_array, inplace=True)
    df.drop(index=lower_array, inplace=True)

    end_shape = df.shape
    print("New Shape: ", end_shape)

    print(f"Values dropped: {start_shape[0]-end_shape[0]}")

    return df


def plot_precision_recall_and_confusion_matrix(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> None:
    """
    Plot the Precision-Recall Curve and Confusion Matrix for a given classification model.

    Args:
        model (XGBClassifier): The classification model.
        X (pd.DataFrame): The input features for evaluation.
        y (pd.Series): The true target labels.

    Returns:
        None
    """
    precision, recall, thresholds = precision_recall_curve(y, model.predict_proba(X)[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=axes[0])
    axes[0].set_title("Precision-Recall Curve")

    cm = ConfusionMatrixDisplay.from_estimator(
        model, X, y, cmap=plt.cm.Blues, normalize="true", ax=axes[1]
    )
    cm.ax_.set_title("Confusion Matrix")
    
    
def plot_displot(df_column: pd.Series) -> None:
    """
    Create a distribution plot (displot) for a DataFrame column.

    Parameters:
        df_column (pd.Series): The column from a DataFrame for which to create the distribution plot.

    Returns:
        None
    """
    plt.figure()
    skewness = df_column.skew()
    sns.histplot(df_column, kde=False, bins=30, color="cornflowerblue")
    plt.ylabel("Count")
    plt.title(f"{df_column.name} distribution")
    plt.text(
        df_column.min(),
        plt.gca().get_ylim()[1] - 5,
        "Skew: {:.2f}".format(skewness),
        fontsize=12,
        horizontalalignment="left",
        verticalalignment="top",
    )
    
    custom_xticks = np.linspace(df_column.min(), df_column.max(), num=10)
    custom_xticks = np.round(custom_xticks).astype(int)
    plt.xticks(ticks=custom_xticks, labels=custom_xticks)
    plt.show()
    
    
def top_words(df: pd.DataFrame,
              text_col_name: str,
              ngram_no: int,
              n_top: int,
              exclude_list: Optional[List[str]] = None) -> Dict:
    """
    Returns a dictionary of the top n_top words.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data for analysis.
        text_col_name (str): The name of the column containing the text for analysis.
        ngram_no (int): The number of top words from each string to return.
        n_top (int): The number of top word groups to return.
        exclude_list (List[str], optional): Words that should be excluded from the analysis. Default is None.

    Returns:
        dict: A dictionary containing the top n_top words.

    Example:

    top_words(df=dataframe, text_col_name='text', ngram_no=3, n_top=10, exclude_list=['not', 'these', 'words'])
    """
    if exclude_list is None:
        exclude_list = []

    df = df.loc[df[text_col_name].apply(lambda x: isinstance(x, str))]

    top_words = defaultdict(int)

    for sentence in df[text_col_name]:
        words = sentence.split()
        words = [w for w in words if w not in exclude_list]
        for ngram in ngrams(words, ngram_no):
            top_words[ngram] += 1

    return dict(sorted(top_words.items(), key=lambda x: x[1], reverse=True)[:n_top])


def check_normality(data: np.ndarray) -> None:
    """
    Check the normality of a given dataset using the Anderson-Darling and Kolmogorov-Smirnov tests.

    Parameters:
        data (np.ndarray): The input dataset to be checked for normality.

    Returns:
        None
    """
    print(f"Skewness coef.: {stats.skew(data):.2f}")
    anderson_test = stats.anderson(data)
    if anderson_test.statistic > anderson_test.critical_values[2]:
        print("Anderson-Darling Test: Not normally distributed")
    else:
        print("Anderson-Darling Test: Normally distributed")

    stat, p = stats.kstest(data, "norm")
    if p < 0.05:
        print("Kolmogorov-Smirnov Test: Not normally distributed \n")
    else:
        print("Kolmogorov-Smirnov Test: Normally distributed \n")
        
        
def plot_dunn_results(data: Union[ndarray, pd.DataFrame], alpha: float) -> None:
    """
    Plot the results of Dunn's test as a heatmap.

    Parameters:
        data (Union[ndarray, pd.DataFrame]): A 2D array or DataFrame containing p-values from Dunn's test.
        alpha (float): The significance level for Dunn's test.

    Returns:
        None
    """
    up_triang = np.triu(np.ones_like(data, dtype=bool))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True,
        mask=(data > alpha) | up_triang,
        cmap=sns.color_palette(["lightgreen"]),
        cbar=False,
        linewidths=0.5,
    )

    sns.heatmap(
        data,
        annot=False,
        cmap=sns.color_palette(["white"]),
        cbar=False,
        mask=~up_triang,
    )
    legend_elements = [
        Patch(
            facecolor="lightgreen",
            edgecolor="black",
            label="Statistically Significant Difference",
        )
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.title("Dunn's test results (p-values)")
    plt.show()
    
    