 <img src="topf.svg" height="128" />

# Topf &mdash; Topological peak filtering

`topf` is a small library for Python 3 that permits the detection and
subsequent filtering of peaks in one-dimensional functions. The method
is based on a topological notion of *prominence* or *persistence* of a
peak with respect to all other peaks.

# Dependencies

- Python 3.7
- `numpy`

# Usage

Install the library using `pip3 install topf`. You can then access the
main class, `PersistenceTransformer` by issuing `import topf`. As
a simple example, we load the file `example.txt`, depict its peaks,
and filter the smallest ones:

```python
import topf

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

# Third, let's show an example of filtering. At present, there is
# not automated way of doing so.
filtered_data = data[peaks[:, 1] > 4]  # only keep high peaks

plt.subplot(3, 1, 3)
sns.lineplot(x=data[:, 0], y=data[:, 1], alpha=0.5)
sns.scatterplot(
    x=filtered_data[:, 0],
    y=filtered_data[:, 1],
)

plt.tight_layout()
plt.show()
```

This file is also available as [`example.py`](example.py) in this
repository&nbsp;(with some minor modifications to simplify usage).

# Licence notice

The icon of this project was created by <a href="http://www.freepik.com"
title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/"
title="Flaticon">www.flaticon.com</a> and is licensed by <a
href="http://creativecommons.org/licenses/by/3.0/" title="Creative
Commons BY 3.0" target="_blank">CC 3.0 BY</a>.
