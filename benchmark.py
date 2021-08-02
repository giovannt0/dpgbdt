# -*- coding: utf-8 -*-
# gtheo@ethz.ch
# ypo@informatik.uni-kiel.de

"""Example test file."""

import csv
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn import model_selection

import estimator



def get_abalone(n_rows: Optional[int] = None) -> Any:
  """Parse the abalone dataset.

  Args:
    n_rows (int): Numbers of rows to read.

  Returns:
    Any: X, y, cat_idx, num_idx
  """
  # pylint: disable=redefined-outer-name,invalid-name
  # Re-encode gender information
  data = pd.read_csv(
      './abalone.data',
      names=['sex', 'length', 'diameter', 'height', 'whole weight',
             'shucked weight', 'viscera weight', 'shell weight', 'rings'])
  data['sex'] = pd.get_dummies(data['sex'])
  if n_rows:
    data = data.head(n_rows)
  y = data.rings.values.astype(np.float)
  del data['rings']
  X = data.values.astype(np.float)
  cat_idx = [0]  # Sex
  num_idx = list(range(1, X.shape[1]))  # Other attributes
  return X, y, cat_idx, num_idx

def cross_validate(parameters, X, y, filename):
  # dummy_estimator = estimator.DPGBDT(
  #   privacy_budget = 0.1,
  #   clipping_bound = 1.0,
  #   nb_trees = 50,
  #   nb_trees_per_ensemble = 50,
  #   max_depth = 60,
  #   learning_rate= 0.1
  # )
  dummy_estimator = estimator.DPGBDT(
    clipping_bound = None,
    nb_trees = 0,
    nb_trees_per_ensemble = 0,
    max_depth = 0,
    learning_rate= 0.0
  )
  best_model = model_selection.GridSearchCV(
    estimator = dummy_estimator,
    param_grid = parameters,
    scoring = "neg_root_mean_squared_error",
    n_jobs = 60,
    cv = model_selection.RepeatedKFold(n_splits = 5, n_repeats = 10),
    verbose = 2
  )
  best_model.fit(X, y)
  df = pd.DataFrame(best_model.cv_results_)
  df.to_csv(filename)

def cv_clipping():
  X, y, cat_idx, num_idx = get_abalone()
  privacy_budget = np.append(
    np.linspace(0.1, 0.9, num = 9),
    np.linspace(1.0, 5.0, num = 9)
  )
  common_params = dict(
    privacy_budget = privacy_budget,
    clipping_bound = np.logspace(-2, 1, 20),
    nb_trees = [50],
    nb_trees_per_ensemble = [50],
    max_depth = [6],
    max_leaves=[24],
    learning_rate = [0.1],
    cat_idx = [cat_idx],
    num_idx = [num_idx]
  )

  dfs_parameters = common_params
  cross_validate(dfs_parameters, X, y, "dfs_5-fold-RMSE.csv")

  bfs_parameters = dict (
      use_bfs = [True],
      **common_params
  )
  cross_validate(bfs_parameters, X, y, "use_bfs_5-fold-RMSE.csv")

  three_trees_parameters = dict (
      use_3_trees = [True],
      **common_params
  )
  cross_validate(three_trees_parameters, X, y, "use_3_trees_5-fold-RMSE.csv")

def cv_no_dp():
  X, y, cat_idx, num_idx = get_abalone()
  common_params = dict(
    privacy_budget = [None],
    clipping_bound = [1000000000000.0],
    nb_trees = [50],
    nb_trees_per_ensemble = [50],
    max_depth = [6],
    learning_rate = [0.1],
    cat_idx = [cat_idx],
    num_idx = [num_idx]
  )
  cross_validate(common_params, X, y, "no_dp.csv")

if __name__ == '__main__':
  cv_clipping()