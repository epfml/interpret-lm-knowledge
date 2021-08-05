#!/usr/bin/env python3

# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi


def make_spider(df, row, title, color):
    """
    Define a function that do a plot for one line of the dataset
    """
    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(1, 3, row + 1, polar=True, )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='darkgray', size=8)

    # Draw y-labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.50, 0.75], ["0.25", "0.50", "0.75"], color="darkgray", size=10)
    plt.ylim(0, 1)

    # Ind1
    values = df.loc[row].drop('model').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=11, color='black', y=1.1)


def create_global_plot(df, color):
    """
    Apply the function to all individual plots
    """
    # Initialize the figure
    my_dpi = 96
    plt.figure(figsize=(1000 / my_dpi, 450 / my_dpi), dpi=my_dpi)

    # Loop to plot
    for data_row in range(0, len(df.index)):
        make_spider(df=df, row=data_row, title=df['model'][data_row], color=color)

    return plt


def transform_data(df):
    """
    Transforms data to accuracy measurement.
    """
    df[df.columns[1:]] = (100 - abs(df[df.columns[1:]])) / 100
    return df


if __name__ == '__main__':
    # Read data
    data = pd.read_csv('./results/pos_results_for_spider_graphs.csv')
    data_1 = data.copy().iloc[:3].reset_index(drop=True)  # pre-trained BERT-like models
    data_2 = data.copy().iloc[3:6].reset_index(drop=True)  # Roberta different epochs

    data_1 = transform_data(data_1)
    my_palette = plt.cm.get_cmap("Set2", 1)(1)
    plt_1 = create_global_plot(data_1, my_palette)

    plt_1.savefig('./results/BERT_like_models_pos_performance.png')
    plt_1.show()

    data_2 = transform_data(data_2)
    my_palette = plt.cm.get_cmap("Set2", 8)(2)
    plt_2 = create_global_plot(data_2, my_palette)

    plt_2.savefig('./results/Robert_pos_performance_per_epoch.png')
    plt_2.show()
