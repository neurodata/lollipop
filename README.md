# Linear Optimal Low Rank Projection (lolP)


[![arXiv shield](https://img.shields.io/badge/arXiv-1709.01233-red.svg?style=flat)](https://arxiv.org/abs/1709.01233)
[![Build Status](https://travis-ci.org/j1c/lol.svg?branch=master)](https://travis-ci.org/j1c/lol)
[![codecov](https://codecov.io/gh/j1c/lol/branch/master/graph/badge.svg)](https://codecov.io/gh/j1c/lol)

# Overview

Supervised learning techniques designed for the situation when the dimensionality exceeds the sample size have a tendency to overfit as the dimensionality of the data increases. To remedy this High dimensionality; low sample size (HDLSS) situation, we attempt to learn a lower-dimensional representation of the data before learning a classifier. That is, we project the data to a situation where the dimensionality is more manageable, and then are able to better apply standard classification or clustering techniques since we will have fewer dimensions to overfit. A number of previous works have focused on how to strategically reduce dimensionality in the unsupervised case, yet in the supervised HDLSS regime, few works have attempted to devise dimensionality reduction techniques that leverage the labels associated with the data. In this package, we provide several methods for feature extraction, some utilizing labels and some not, along with easily extensible utilities to simplify cross-validative efforts to identify the best feature extraction method. Additionally, we include a series of adaptable benchmark simulations to serve as a standard for future investigative efforts into supervised HDLSS. Finally, we produce a comprehensive comparison of the included algorithms across a range of benchmark simulations and real data applications. (Credit: Eric Bridgeford)

For R implmentation, please [here](https://github.com/neurodata/lol).

# System Requirements

## Hardware Requirements
- **lolP** package requires only a standard computer with enough RAM to support the in-memory operations.
- Requires no non-standard hardware to run.

## Software Requirements
- **lolP** was developed in Python 3.6. Currently, there is no plan to support Python 2.
- Was developed and tested primarily on Mac OS (Sierra 10.12.6).
- **lolP** package should be compatible with Windows, Mac, and Linux operating systems.
- **lolP** is robust to Python package versions as it only requires the following packages:
```
numpy
scikit-learn
scipy
```

# Installation Guide

## Stable Release
`lolP` is available on PyPi:
```
pip install lolP
```

# Demo
The **lolP** package offers identical API to that of scikit-learn. Thus, if you have used scikit-learn,
you will find the usage very familiar. Below is a very simple demo on the usage of **lolP**.

```
from lol import LOL
import numpy as np

# Generate two random datasets

# 100 samples, 10 dimensions
X = np.random.rand(100, 10)
X2 = np.random.rand(100, 10)

# Two classes with equal proportions
y = np.random.binomial(1, 0.5, size=100)

lmao = LOL(n_components=4, svd_solver='full')
lmao.fit(X, y)
X2_transformed = lmao.transform(X2)
```