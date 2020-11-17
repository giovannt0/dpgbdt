# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Estimator wrapper around the implementation."""

from typing import Dict, Any, Optional, List
# pylint: disable=import-error
from sklearn.base import BaseEstimator
# pylint: enable=import-error

import numpy as np
from model import GradientBoostingEnsemble


class DPGBDT(BaseEstimator):  # type: ignore
  """Scikit wrapper around the model."""
  # pylint: disable=too-many-arguments, invalid-name

  def __init__(self,
               privacy_budget: float,
               nb_trees: int,
               nb_trees_per_ensemble: int,
               max_depth: int,
               learning_rate: float,
               n_classes: Optional[int] = None,
               max_leaves: Optional[int] = None,
               min_samples_split: int = 2,
               balance_partition: bool = True,
               use_bfs: bool = False,
               use_3_trees: bool = False,
               binary_classification: bool = False,
               cat_idx: Optional[List[int]] = None,
               num_idx: Optional[List[int]] = None) -> None:
    """Initialize the wrapper.

    Args:
      privacy_budget (float): The privacy budget to use.
      nb_trees (int): The number of trees in the model.
      nb_trees_per_ensemble (int): The number of trees per ensemble.
      max_depth (int): The max depth for the trees.
      learning_rate (float): The learning rate.
      n_classes (int): Number of classes. Triggers regression (None) vs classification.
      max_leaves (int): Optional. The max number of leaf nodes for the trees.
          Tree will grow in a best-leaf first fashion until it contains
          max_leaves or until it reaches maximum depth, whichever comes first.
      min_samples_split (int): Optional. The minimum number of samples required
          to split an internal node. Default is 2.
      balance_partition (bool): Optional. Balance data repartition for training
          the trees. The default is True, meaning all trees within an ensemble
          will receive an equal amount of training samples. If set to False,
          each tree will receive <x> samples where <x> is given in line 8 of
          the algorithm in the author's paper.
      use_bfs (bool): Optional. If max_leaves is specified, then this is
          automatically True. This will build the tree in a BFS fashion instead
          of DFS. Default is False.
      use_3_trees (bool): Optional. If True, only build trees that have 3
          nodes, and then assemble nb_trees based on these sub-trees, at random.
          Default is False.
      binary_classification (bool): Optional. If true, maps back the
          predictions to labels.
      cat_idx (List): Optional. List of indices for categorical features.
      num_idx (List): Optional. List of indices for numerical features.
    """
    self.model = None
    self.privacy_budget = privacy_budget
    self.nb_trees = nb_trees
    self.nb_trees_per_ensemble = nb_trees_per_ensemble
    self.max_depth = max_depth
    self.max_leaves = max_leaves
    self.min_samples_split = min_samples_split
    self.learning_rate = learning_rate
    self.balance_partition = balance_partition
    self.use_bfs = use_bfs
    self.use_3_trees = use_3_trees
    self.binary_classification = binary_classification
    self.cat_idx = cat_idx
    self.num_idx = num_idx
    self.model = GradientBoostingEnsemble(
        self.nb_trees,
        self.nb_trees_per_ensemble,
        n_classes=n_classes,
        max_depth=self.max_depth,
        privacy_budget=self.privacy_budget,
        learning_rate=self.learning_rate,
        max_leaves=self.max_leaves,
        min_samples_split=self.min_samples_split,
        balance_partition=self.balance_partition,
        use_bfs=self.use_bfs,
        use_3_trees=self.use_3_trees,
        cat_idx=self.cat_idx,
        num_idx=self.num_idx)

  def fit(self, X: np.array, y: np.array) -> 'GradientBoostingEnsemble':
    """Fit the model to the dataset.

    Args:
      X (np.array): The features.
      y (np.array): The label.

    Returns:
      GradientBoostingEnsemble: A GradientBoostingEnsemble object.
    """
    assert self.model
    return self.model.Train(X, y)

  def predict(self, X: np.array) -> np.array:
    """Predict the label for a given dataset.

    Args:
      X (np.array): The dataset for which to predict values.

    Returns:
      np.array: The predictions.
    """
    assert self.model
    # try (multi-class) classification output first,
    # ow fallback to the raw regression values
    try:
      return self.model.PredictLabels(X)
    except ValueError:
      reg_preds = self.model.Predict(X).squeeze()  # shape: (n_samples,)
      # binary classification is here conducted by regression
      # and not by the deviance loss (could be improved later)
      if not self.binary_classification:
        return reg_preds
      else:
        return np.where(reg_preds < 0, -1, 1)

  def get_params(
      self,
      deep: bool = True) -> Dict[str, Any]:  # pylint: disable=unused-argument
    """Stub for sklearn cross validation"""
    return {
        'privacy_budget': self.privacy_budget,
        'nb_trees': self.nb_trees,
        'nb_trees_per_ensemble': self.nb_trees_per_ensemble,
        'max_depth': self.max_depth,
        'learning_rate': self.learning_rate,
        'max_leaves': self.max_leaves,
        'min_samples_split': self.min_samples_split,
        'balance_partition': self.balance_partition,
        'use_bfs': self.use_bfs,
        'use_3_trees': self.use_3_trees,
        'binary_classification': self.binary_classification,
        'cat_idx': self.cat_idx,
        'num_idx': self.num_idx
    }

  def set_params(self,
                 **parameters: Dict[str, Any]) -> 'DPGBDT':
    """Stub for sklearn cross validation"""
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self
