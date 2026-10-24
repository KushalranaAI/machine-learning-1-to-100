"""mlscratch - every classical machine-learning algorithm implemented from the math, in NumPy only.

The goal is understanding, not speed: each file reads like the derivation in its lesson
(see curriculum/). scikit-learn appears only in tests/, to check our answers.
"""

from . import (anomaly, association, cluster, datasets, decomposition, ensemble, feature_selection, linear_model,
               metrics, model_selection, naive_bayes, neighbors, neural_network, optim, preprocessing, rl, svm, text, tree)

__all__ = ["anomaly", "association", "cluster", "datasets", "decomposition", "ensemble", "feature_selection",
           "linear_model", "metrics", "model_selection", "naive_bayes", "neighbors", "neural_network", "optim",
           "preprocessing", "rl", "svm", "text", "tree"]
__version__ = "1.0.0"
