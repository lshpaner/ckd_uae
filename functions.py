import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc

################################################################################
################## Custom Functions for CKD_UAE Project ########################
################################################################################


# Function to move a column to be immediately to the left of another column
def move_column_before(df, target_column, before_column):
    """
    Moves a specified column in a pandas DataFrame to be immediately to the left
    of another specified column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to be rearranged.
    - target_column (str): The name of the column to move. This column will be
                           repositioned in the DataFrame.
    - before_column (str): The name of the column before which the target
                           column will be placed. The target column will be 
                           moved to the immediate left of this column.

    Returns:
    - pd.DataFrame: A DataFrame with the columns rearranged according to the
                    specified order. If either the target_column or before_column 
                    does not exist in the DataFrame, the function will print an 
                    error message and return the original DataFrame unchanged.

    Raises:
    - ValueError: If `target_column` or `before_column` are not found in the
                  DataFrame's columns.
    """

    # Ensure both columns exist in the DataFrame
    if target_column not in df.columns or before_column not in df.columns:
        print(
            f"One or both specified columns ('{target_column}', "
            f"'{before_column}') are not in the DataFrame."
        )
        return df

    # Create a list of columns without the target column
    cols = list(df.columns)
    cols.remove(target_column)

    # Find index of the before_column
    before_column_index = cols.index(before_column)

    # Insert the target column back into the list at the new position
    cols.insert(before_column_index, target_column)

    # Reindex the DataFrame with the new column order
    return df[cols]


################################################################################
################################ Cross-Tab Plot ################################
################################################################################


def crosstab_plot(
    df,
    sub1,
    sub2,
    x,
    y,
    list_name,
    col1,
    bbox_to_anchor,
    w_pad,
    h_pad,
    item1=None,
    item2=None,
    label1=None,
    label2=None,
    crosstab_option=True,
):
    """
    Generates a series of crosstab plots to visualize the relationship between
    an outcome variable and several categorical variables within a dataset. Each
    subplot represents the distribution of outcomes for a specific categorical
    variable, allowing for comparisons across categories.

    The subplot grid, plot size, legend placement, and subplot padding are
    customizable. The function can create standard or normalized crosstab plots
    based on the 'crosstab_option' flag.

    Parameters:
    - df: The DataFrame to pass in.
    - sub1, sub2 (int): The number of rows and columns in the subplot grid.
    - x, y (int): Width & height of ea. subplot, affecting the overall fig. size.
    - list_name (list[str]): A list of strings rep. the column names to be plotted.
    - label1, label2 (str): Labels for the x-axis categories, corresponding to
                            the unique values in the 'outcome' variable of the
                            dataframe.
    - col1 (str): The column name in the dataframe for which custom legend
                  labels are desired.
    - item1, item2 (str): Custom legend labels for the plot corresponding to 'col1'.
    - bbox_to_anchor (tuple): A tuple of (x, y) coordinates to anchor the legend to a
                              specific point within the axes.
    - w_pad, h_pad (float): The amount of width and height padding (space)
                            between subplots.
    - crosstab_option (bool, optional): If True, generates standard crosstab
                                        plots. If False, generates normalized
                                        crosstab plots, which are useful for
                                        comparing distributions across groups
                                        with different sizes.

    The function creates a figure with the specified number of subplots laid out
    in a grid, plots the crosstabulation data as bar plots within each subplot,
    and then adjusts the legend and labels accordingly. It uses a tight layout
    with specified padding to ensure that subplots are neatly arranged without
    overlapping elements.
    """

    fig, axes = plt.subplots(sub1, sub2, figsize=(x, y))
    for item, ax in zip(list_name, axes.flatten()):
        if crosstab_option:
            crosstab_data = pd.crosstab(df["outcome"], df[item])
            crosstab_data.plot(
                kind="bar",
                stacked=True,
                rot=0,
                ax=ax,
                color=["#00BFC4", "#F8766D"],
            )
        else:
            # Computing normalized crosstabulation
            crosstab_data = pd.crosstab(df["outcome"], df[item], normalize="index")
            crosstab_data.plot(
                kind="bar",
                stacked=True,
                rot=0,
                ax=ax,
                color=["#00BFC4", "#F8766D"],
            )

        new_labels = [label1, label2]
        ax.set_xticklabels(new_labels)
        # new_legend = ["Not Obese", "Obese"]
        # ax.legend(new_legend)
        ax.set_title(f"Outcome vs. {item}")
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Frequency")
        # Dynamically setting legend labels
        # Check if the current column is 'Sex' for custom legend labels
        if item == col1:
            legend_labels = [item1, item2]
        else:
            # Dynamically setting legend labels for other columns
            legend_labels = ["No {}".format(item), "{}".format(item)]

        # Updating legend with custom labels
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,
        )

    plt.tight_layout(w_pad=w_pad, h_pad=h_pad)
    plt.show()


################################################################################
############################## Stacked Bar Graph ###############################
################################################################################


def stacked_plot(
    x,
    y,
    p,
    df,
    col,
    truth,
    condition,
    kind,
    width,
    rot,
    ascending=True,
    string=None,
    custom_order=None,
    legend_labels=False,
):
    """
    Generates a pair of stacked bar plots for a specified column against a ground
    truth column, with the first plot showing absolute distributions and the
    second plot showing normalized distributions.

    Parameters:
    - x (int): The width of the figure.
    - y (int): The height of the figure.
    - p (int): The padding between the subplots.
    - df (DataFrame): The pandas DataFrame containing the data.
    - col (str): The name of the column in the DataFrame to be analyzed.
    - truth (str): The name of the ground truth column in the DataFrame.
    - condition: Unused parameter, included for future use.
    - kind (str): The kind of plot to generate (e.g., 'bar', 'barh').
    - width (float): The width of the bars in the bar plot.
    - rot (int): The rotation angle of the x-axis labels.
    - ascending (bool, optional): Determines the sorting order of the DataFrame
                                  based on the 'col'. Defaults to True.
    - string (str, optional): Descriptive string to inc. in title of the plots.
    - custom_order (list, optional): A list specifying a custom order for the
                                     categories in the 'col'. If provided, the
                                     DataFrame is sorted according to this order.
    - legend_labels (bool or list, optional): Specifies whether to display
                                              legend labels and what those
                                              labels should be. If False, no
                                              legend is displayed. If a list,
                                              the list values are used as legend
                                              labels.

    Returns:
    - None: The function creates & displays the plots but doesn't return value.

    Note:
    - The function assumes the matplotlib and pandas libraries have been
      imported as plt and pd respectively.
    - The function automatically handles the layout and spacing of the subplots
      to prevent overlap.
    """
    # Setting custom order if provided
    if custom_order:
        df[col] = pd.Categorical(df[col], categories=custom_order, ordered=True)
        df.sort_values(
            by=col, inplace=True
        )  # Ensure the DataFrame respects the custom ordering

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(x, y))
    fig.tight_layout(w_pad=5, pad=p, h_pad=5)
    fig.suptitle(
        "Absolute Distributions vs. Normalized Distributions",
        fontsize=12,
    )

    # Crosstabulation of column of interest and ground truth
    crosstabdest = pd.crosstab(df[col], df[truth])

    # Normalized crosstabulation
    crosstabdestnorm = crosstabdest.div(crosstabdest.sum(1), axis=0)

    title1 = f"{string} by {truth.capitalize()}"
    title2 = f"{string} by {truth.capitalize()} (Normalized)"
    xlabel1 = xlabel2 = f"{col}"
    ylabel1 = "Count"
    ylabel2 = "Frequency"

    # Plotting the first stacked bar graph
    crosstabdest.plot(
        kind=kind,
        stacked=True,
        title=title1,
        ax=axes[0],
        color=["#00BFC4", "#F8766D"],
        width=width,
        rot=rot,
        fontsize=12,
    )
    axes[0].set_title(title1, fontsize=12)
    axes[0].set_xlabel(xlabel1, fontsize=12)
    axes[0].set_ylabel(ylabel1, fontsize=12)
    axes[0].legend(legend_labels, fontsize=12)

    # Plotting the second, normalized stacked bar graph
    crosstabdestnorm.plot(
        kind=kind,
        stacked=True,
        title=title2,
        ylabel="Frequency",
        ax=axes[1],
        color=["#00BFC4", "#F8766D"],
        width=width,
        rot=rot,
        fontsize=12,
    )
    axes[1].set_title(label=title2, fontsize=12)
    axes[1].set_xlabel(xlabel2, fontsize=12)
    axes[1].set_ylabel(ylabel2, fontsize=12)
    axes[1].legend(legend_labels, fontsize=12)

    fig.align_ylabels()
    plt.show()


################################################################################
############################# ROC Curves by Category ###########################
################################################################################

def plot_roc_curves_by_category(
    X_test,
    y_test,
    predictions,
    feature,
    category_labels,
    outcome,
    title,
):
    """
    Plots ROC curves for each category in a specified feature, using custom
    category labels.

    Parameters:
    - X_test: DataFrame containing the test features, including the categorical
              feature to stratify by.
    - y_test: Series or array containing the true labels.
    - predictions: Array containing the predicted probabilities.
    - feature: Str, the name of the categorical feature in X_test to stratify by.
    - category_labels: Dict, mapping of category codes to descriptive labels.
    - title: String, the title of the plot.
    """
    plt.title(title)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    categories = X_test[feature].unique()
    for category in categories:
        if category in category_labels:
            # Filter y_test and predictions by the current category
            category_filter = (X_test[feature] == category).values
            y_test_filtered = y_test[category_filter]
            predictions_filtered = predictions[category_filter]

            # Compute ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_test_filtered, predictions_filtered)
            roc_auc = auc(fpr, tpr)

            # Plot the ROC curve using the custom label
            plt.plot(
                fpr,
                tpr,
                label=(
                    f"{category_labels[category]}, "
                    f"count = {len(y_test_filtered)}, "
                    f"$H_0$ = {y_test_filtered[outcome].value_counts()[0]}, "
                    f"$H_1$ = {y_test_filtered[outcome].value_counts()[1]}, "
                    f"(AUC = {roc_auc:.2f})"
                ),
            )

    plt.legend(loc="lower right")
    plt.show()
