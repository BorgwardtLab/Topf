#!/usr/bin/env python3
#
# example_filtering.py: demonstrates how to use `topf` with automated
# peak filtering.

import topf

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


if __name__ == '__main__':
    data = np.genfromtxt('example.txt')

    # This will automatically instruct the transformer to filter peaks
    # until only the 3 highest ones are kept.
    transformer = topf.PersistenceTransformer(
        n_peaks=3
    )
    peaks = transformer.fit_transform(data)

    # First, let's plot the original data. We can see that there is
    # quite a number of relatively small peaks.
    plt.subplot(3, 1, 1)
    sns.lineplot(x=data[:, 0], y=data[:, 1])

    # Second, let's show the transformed data. Here, every non-zero
    # point depicts the *prominence* of a peak.
    plt.subplot(3, 1, 2)
    sns.lineplot(x=peaks[:, 0], y=peaks[:, 1])

    plt.subplot(3, 1, 3)
    sns.lineplot(x=data[:, 0], y=data[:, 1], alpha=0.5)
    sns.scatterplot(
        x=data[peaks[:, 1] > 0][:, 0],
        y=data[peaks[:, 1] > 0][:, 1],
    )

    plt.tight_layout()
    plt.show()
