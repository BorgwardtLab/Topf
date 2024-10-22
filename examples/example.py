#!/usr/bin/env python3
#
# Demonstrates how to use `topf` in order to filter peaks in a simple
# data set.


import topf

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


if __name__ == '__main__':
    data = np.genfromtxt('example.txt')          # load data
    transformer = topf.PersistenceTransformer()  # prepare transformer
    peaks = transformer.fit_transform(data)      # transform data into peaks

    # First, let's plot the original data. We can see that there is
    # quite a number of relatively small peaks.
    plt.subplot(3, 1, 1)
    sns.lineplot(x=data[:, 0], y=data[:, 1])

    # Second, let's show the transformed data. Here, every non-zero
    # point depicts the *prominence* of a peak.
    plt.subplot(3, 1, 2)
    sns.lineplot(x=peaks[:, 0], y=peaks[:, 1])

    # Third, let's show an example of filtering. This is the manual way
    # of filtering. It is particularly useful if you know the height of
    # an interesting peak.
    filtered_data = data[peaks[:, 1] > 4]  # only keep high peaks

    plt.subplot(3, 1, 3)
    sns.lineplot(x=data[:, 0], y=data[:, 1], alpha=0.5)
    sns.scatterplot(
        x=filtered_data[:, 0],
        y=filtered_data[:, 1],
    )

    plt.tight_layout()
    plt.show()
