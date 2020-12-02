# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Implement ensemble of differentially private gradient boosted trees.

From: https://arxiv.org/pdf/1911.04209.pdf
"""

import math
import logging
import operator
from queue import Queue
from typing import List, Any, Optional, Dict, Tuple

import numpy as np
# pylint: disable=import-error
from sklearn.model_selection import train_test_split
# pylint: enable=import-error
from sklearn.ensemble._gb_losses import LeastSquaresError, MultinomialDeviance, LossFunction

logger = logging.getLogger(__name__)


class GradientBoostingEnsemble:
  """Implement gradient boosting ensemble of trees.

  Attributes:
    nb_trees (int): The total number of trees in the model.
    nb_trees_per_ensemble (int): The number of trees in each ensemble.
    max_depth (int): The depth for the trees.
    privacy_budget (float): The privacy budget available for the model.
    learning_rate (float): The learning rate.
    l2_threshold (int): Threshold for the loss function. For the square loss
        function (default), this is 1.
    l2_lambda (float): Regularization parameter for l2 loss function.
        For the square loss function (default), this is 0.1.
    trees (List[List[DifferentiallyPrivateTree]]): A list of k-classes DP trees.
  """
  # pylint: disable=invalid-name, too-many-arguments, unused-variable

  def __init__(self,
               nb_trees: int,
               nb_trees_per_ensemble: int,
               n_classes: Optional[int] = None,
               max_depth: int = 6,
               privacy_budget: float = 1.0,
               learning_rate: float = 0.1,
               max_leaves: Optional[int] = None,
               min_samples_split: int = 2,
               balance_partition: bool = True,
               use_bfs: bool = False,
               use_3_trees: bool = False,
               cat_idx: Optional[List[int]] = None,
               num_idx: Optional[List[int]] = None) -> None:
    """Initialize the GradientBoostingEnsemble class.

    Args:
      nb_trees (int): The total number of trees in the model.
      nb_trees_per_ensemble (int): The number of trees in each ensemble.
      n_classes (int): Number of classes. Triggers regression (None) vs classification.
      max_depth (int): Optional. The depth for the trees. Default is 6.
      privacy_budget (float): Optional. The privacy budget available for the
          model. Default is 1.0.
      learning_rate (float): Optional. The learning rate. Default is 0.1.
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
      cat_idx (List): Optional. List of indices for categorical features.
      num_idx (List): Optional. List of indices for numerical features.
      """
    self.nb_trees = nb_trees
    self.nb_trees_per_ensemble = nb_trees_per_ensemble
    self.max_depth = max_depth
    self.privacy_budget = privacy_budget
    self.learning_rate = learning_rate
    self.max_leaves = max_leaves
    self.min_samples_split = min_samples_split
    self.balance_partition = balance_partition
    self.use_bfs = use_bfs
    self.use_3_trees = use_3_trees
    self.cat_idx = cat_idx
    self.num_idx = num_idx
    self.trees = []  # type: List[List[DifferentiallyPrivateTree]]
    # classification vs regression
    self.loss_ = MultinomialDeviance(n_classes) if n_classes else LeastSquaresError(1)  # type: LossFunction
    self.init_ = self.loss_.init_estimator()

    # Loss parameters
    self.l2_threshold = 1.0
    self.l2_lambda = 0.1

    # Initial score
    self.init_score = None

    if self.use_3_trees and self.use_bfs:
      # Since we're building 3-node trees it's the same anyways.
      self.use_bfs = False

  def Train(self,
            X: np.array,
            y: np.array) -> 'GradientBoostingEnsemble':
    """Train the ensembles of gradient boosted trees.

    Args:
      X (np.array): The features.
      y (np.array): The label.

    Returns:
      GradientBoostingEnsemble: A GradientBoostingEnsemble object.
    """

    # Init gradients
    self.init_.fit(X, y)
    self.init_score = self.loss_.get_init_raw_predictions(X, self.init_)  # (n_samples, K)
    update_gradients = True

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X, y = X_train, y_train

    # Number of ensembles in the model
    nb_ensembles = int(np.ceil(self.nb_trees / self.nb_trees_per_ensemble))

    # Privacy budget allocated to all trees in each ensemble
    tree_privacy_budget = np.divide(self.privacy_budget, nb_ensembles)
    # In multi-class classification the budget for each tree
    # is the same as for the whole K trees but halved
    # As each datapoint is only assigned to one class,
    # it only matters if it is assigned to the considered class or not but not to which other
    # Thus it always remains 2 - independently of how many total classes exists
    privacy_budget_per_tree = 2 if self.loss_.is_multi_class else 1
    tree_privacy_budget = np.divide(tree_privacy_budget, privacy_budget_per_tree)

    prev_score = np.inf

    # Train all trees
    for tree_index in range(self.nb_trees):
      # Compute sensitivity
      delta_g = 3 * np.square(self.l2_threshold)
      delta_v = min(self.l2_threshold / (1 + self.l2_lambda),
                    2 * self.l2_threshold * math.pow(
                      (1 - self.learning_rate), tree_index))

      current_tree_for_ensemble = tree_index % self.nb_trees_per_ensemble
      if current_tree_for_ensemble == 0:
        # Initialize the dataset and the gradients
        X_ensemble = np.copy(X)
        y_ensemble = np.copy(y)
        prev_score = np.inf
        update_gradients = True
        # gradient initialization will happen later in the per-class-loop

      # Compute the number of rows that the current tree will use for training
      if self.balance_partition:
        # All trees will receive same amount of samples
        if self.nb_trees % self.nb_trees_per_ensemble == 0:
          # Perfect split
          number_of_rows = int(len(X) / self.nb_trees_per_ensemble)
        else:
          # Partitioning data across ensembles
          if np.ceil(tree_index / self.nb_trees_per_ensemble) == np.ceil(
              self.nb_trees / self.nb_trees_per_ensemble):
            number_of_rows = int(len(X) / (
                self.nb_trees % self.nb_trees_per_ensemble))
          else:
            number_of_rows = int(len(X) / self.nb_trees_per_ensemble) + int(
                len(X) / (self.nb_trees % self.nb_trees_per_ensemble))
      else:
        # Line 8 of Algorithm 2 from the paper
        number_of_rows = int((len(X) * self.learning_rate * math.pow(
          (1 - self.learning_rate), current_tree_for_ensemble)) / (
              1 - math.pow((
                  1 - self.learning_rate), self.nb_trees_per_ensemble)))

      # If using the formula from the algorithm, some trees may not get
      # samples. In that case we skip the tree and issue a warning. This
      # should hint the user to change its parameters (likely the ensembles
      # are too unbalanced)
      if number_of_rows == 0:
        logger.warning('The choice of trees per ensemble vs. the total number '
                       'of trees is not balanced properly; some trees will '
                       'not get any training samples. Try using '
                       'balance_partition=True or change your parameters.')
        continue

      # Select <number_of_rows> rows at random from the ensemble dataset
      rows = np.random.randint(len(X_ensemble), size=number_of_rows)
      X_tree = X_ensemble[rows, :]
      y_tree = y_ensemble[rows]

      # train for each class a seperate tree on the same rows.
      # In regression or binary classification, K has been set to one.
      k_trees = []  # type: List[DifferentiallyPrivateTree]
      for kth_tree in range(self.loss_.K):
        if tree_index == 0:
            # First tree, start with initial scores (mean of labels)
            gradients = self.ComputeGradientForLossFunction(
                y, self.init_score[:len(y)], kth_tree)
        else:
            # Update gradients of all training instances on loss l
            if update_gradients:
                gradients = self.ComputeGradientForLossFunction(
                    y_ensemble, self.Predict(X_ensemble), kth_tree)  # type: ignore

        assert gradients is not None
        gradients_tree = gradients[rows]

        # Gradient based data filtering
        norm_1_gradient = np.abs(gradients_tree)
        rows_gbf = norm_1_gradient <= self.l2_threshold
        X_tree = X_tree[rows_gbf, :]
        y_tree = y_tree[rows_gbf]
        gradients_tree = gradients_tree[rows_gbf]

        # Get back the original row index from the first filtering
        selected_rows = rows[rows_gbf] if tree_index > 0 else rows

        # Fit a differentially private decision tree
        tree = DifferentiallyPrivateTree(
            tree_index,
            self.learning_rate,
            self.l2_threshold,
            self.l2_lambda,
            tree_privacy_budget,
            delta_g,
            delta_v,
            self.loss_,
            max_depth=self.max_depth,
            max_leaves=self.max_leaves,
            min_samples_split=self.min_samples_split,
            use_bfs=self.use_bfs,
            use_3_trees=self.use_3_trees,
            cat_idx=self.cat_idx,
            num_idx=self.num_idx)
        # in multi-class classification, the target has to be binary
        # as each tree is a per-class regressor
        y_target = ((y_tree == kth_tree).astype(np.float64)
                    if self.loss_.is_multi_class
                    else y_tree)
        tree.Fit(X_tree, y_target, gradients_tree)

        # Add the tree to its corresponding ensemble
        k_trees.append(tree)
      self.trees.append(k_trees)

      score = self.loss_(y_test, self.Predict(X_test))  # i.e. mse or deviance
      if score >= prev_score:
        # This tree doesn't improve overall prediction quality, removing from
        # model
        update_gradients = self.loss_.is_multi_class  # not reusing gradients in multi-class as they are class-dependent
        self.trees.pop()
      else:
        print(tree_index, score)
        update_gradients = True
        prev_score = score
        # Remove the selected rows from the ensemble's dataset
        # The instances that were filtered out by GBF can still be used for the
        # training of the next trees
        X_ensemble = np.delete(X_ensemble, selected_rows, axis=0)
        y_ensemble = np.delete(y_ensemble, selected_rows)
    if self.use_3_trees:
      self.Combine_3_trees(self.trees)
    return self

  def Combine_3_trees(self,
                      k_trees: List[List['DifferentiallyPrivateTree']]) -> None:
    """Combine 3-trees together to construct bigger decision trees.

    Args:
      k_trees (List[List[DifferentiallyPrivateTree]]): A k-list of 3-trees.
    """

    self.trees = []  # Re-init final predictions trees
    for index, k_three_tree in enumerate(k_trees):  # iterate through the ensemble
      k_trees_ = []  # type: List[DifferentiallyPrivateTree]
      for k, three_tree in enumerate(k_three_tree):  # iterate through the classes
        # select a whole ensemble per class; continue as if there were only one class
        copy = list(np.copy([i[k] for i in k_trees]))
        copy.pop(index)
        if len(copy) == 0:
          continue
        queue_children = Queue()  # type: Queue['DecisionNode']
        queue_children.put(three_tree.root_node.left_child)  # type: ignore
        queue_children.put(three_tree.root_node.right_child)  # type: ignore
        depth = 1
        privacy_budget_for_node = np.around(
            np.divide(three_tree.privacy_budget / 2, three_tree.max_depth + 1),
            decimals=7)
        while not queue_children.empty():
          if depth == self.max_depth or len(copy) == 0:
            break
          left_child = queue_children.get()
          right_child = queue_children.get()
          for child in [left_child, right_child]:
            if len(copy) == 0 or not child or not child.X.any():  # type: ignore
              continue
            # Apply exponential mechanism to find sub 3-node tree
            probabilities = []
            max_gain = -np.inf
            for candidate_index, candidate in enumerate(copy):
              if not candidate.root_node.X.any():
                continue
              # Compute distance between the two nodes. Lower is better.
              gain = np.linalg.norm(np.matmul(np.transpose(
                  child.X), child.X) - np.matmul(np.transpose(
                      candidate.root_node.X), candidate.root_node.X))
              exp_gain = (privacy_budget_for_node * gain) / (
                  2. * three_tree.delta_g)
              if exp_gain > max_gain:
                max_gain = exp_gain
              prob = {
                'candidate_index': candidate_index,
                'index': candidate.root_node.index,
                'value': candidate.root_node.value,
                'gain': exp_gain
              }
              probabilities.append(prob)
            candidate = ExponentialMechanism(
                probabilities, max_gain, reverse=True)
            if not candidate or not candidate['index'] or not candidate['value']:
              continue
            copy.pop(candidate['candidate_index'])
            split_index = candidate['index']
            split_value = candidate['value']
            left_, right_ = self.SplitNode(child,
                                           split_index,
                                           split_value,
                                           three_tree.privacy_budget,
                                           index,
                                           three_tree.delta_v)
            queue_children.put(left_)
            queue_children.put(right_)
          depth += 1
        k_trees_.append(three_tree)
      self.trees.append(k_trees_)
    if not self.trees or self.trees == [[]]:
      self.trees = k_trees

  def SplitNode(self,
                node: 'DecisionNode',
                index: int,
                value: float,
                tree_privacy_budget: float,
                tree_index: int,
                delta_v: float) -> Tuple['DecisionNode', 'DecisionNode']:
    """Split children of a 3-nodes tree based on the (index, value) pair.

    Args:
      node (DecisionNode): The node to split.
      index (int): The feature's index on which to split the node.
      value (float): The feature's value on which to split the node.
      tree_privacy_budget (float): The privacy budget for the current tree.
      tree_index (int): The index of the tree.
      delta_v (float): The loss function's sensitivity for the tree.

    Returns:
      Tuple: Children created after the split.
    """

    assert node.X is not None
    assert node.y is not None
    assert node.gradients is not None

    # Split indices of instances from the node's dataset
    lhs_op, rhs_op = self.GetOperators(index)
    lhs = np.where(lhs_op(node.X[:, index], value))[0]
    rhs = np.where(rhs_op(node.X[:, index], value))[0]

    # Compute the associated predictions
    lhs_prediction = ComputePredictions(node.gradients[lhs], node.y[lhs], self.loss_, self.l2_lambda)
    rhs_prediction = ComputePredictions(node.gradients[rhs], node.y[rhs], self.loss_, self.l2_lambda)

    # Mark current node as split node and not leaf node
    node.prediction = None
    node.index = index
    node.value = value

    # Add children to node
    node.left_child = DecisionNode(
        X=node.X[lhs],
        y=node.y[lhs],
        prediction=lhs_prediction,
        gradients=node.gradients[lhs])
    node.right_child = DecisionNode(
        X=node.X[rhs],
        y=node.y[rhs],
        prediction=rhs_prediction,
        gradients=node.gradients[rhs])

    # Apply Geometry leaf clipping
    ClipLeaves([node.left_child, node.right_child],
               self.l2_threshold,
               self.learning_rate,
               tree_index)

    # Add noise to the leaf predictions
    laplace_scale = delta_v / tree_privacy_budget / 2
    AddLaplacianNoise([node.left_child, node.right_child],
                      laplace_scale)

    # Shrink by learning rate
    Shrink([node.left_child, node.right_child], self.learning_rate)

    return node.left_child, node.right_child

  def Predict(self, X: np.array) -> np.array:
    """Predict values from the ensemble of gradient boosted trees.

    See https://github.com/microsoft/LightGBM/issues/1778.

    Args:
      X (np.array): The dataset for which to predict values.

    Returns:
      np.array of shape (n_samples, K): The predictions.
    """
    # sum across the ensemble per class
    predictions = np.sum([[tree.Predict(X) for tree in k_trees] for k_trees in self.trees], axis=0).T
    assert self.init_score is not None
    init_score = self.init_score[:len(predictions)]
    return np.add(init_score, predictions)

  def PredictLabels(self, X: np.ndarray) -> np.ndarray:
    """Predict labels out of the raw prediction values of `Predict`.
    Only defined for classification tasks.

    Args:
      X (np.ndarray): The dataset for which to predict labels.

    Returns:
      np.ndarray: The label predictions.
    """
    if type(self.loss_) is not MultinomialDeviance:
      raise ValueError("Labels are not defined for regression tasks.")

    raw_predictions = self.Predict(X)
    encoded_labels = self.loss_._raw_prediction_to_decision(raw_predictions)
    return encoded_labels

  def ComputeGradientForLossFunction(self, y: np.array, y_pred: np.array, k: int) -> np.array:
    """Compute the gradient of the loss function.

    Args:
      y (np.array): The true values.
      y_pred (np.array): The predictions.
      k (int): the class.

    Returns:
      (np.array): The gradient of the loss function.
    """
    if self.loss_.is_multi_class:
      y = (y == k).astype(np.float64)
    # sklearn's impl is using the negative gradient (i.e. y - F).
    # Here the positive gradient is used though
    return -self.loss_.negative_gradient(y, y_pred, k=k)

  def GetOperators(self, index: int) -> Tuple[Any, Any]:
    """Return operators to use to split a node's dataset.

    Args:
      index (int): The index for the feature to split the data on.

    Returns:
      Tuple[Any, Any]: The operators to use.
    """
    if self.cat_idx and index in self.cat_idx:
      # Categorical feature
      return operator.eq, operator.ne
    # Numerical feature
    return operator.lt, operator.ge


class DecisionNode:
  """Implement a decision node.

  Attributes:
    X (np.array): The dataset.
    y (np.ndarray): The dataset labels.
    gradients (np.array): The gradients for the dataset instances.
    index (int): An index for the feature on which the node splits.
    value (Any): The corresponding value for that index.
    depth (int): The depth of the node.
    left_child (DecisionNode): The left child of the node, if any.
    right_child (DecisionNode): The right child of the node, if any.
    prediction (float): For a leaf node, holds the predicted value.
    processed (bool): If a node has been processed during BFS tree construction.
  """

  def __init__(self,
               X: Optional[np.array] = None,
               y: Optional[np.array] = None,
               gradients: Optional[np.array] = None,
               index: Optional[int] = None,
               value: Optional[Any] = None,
               depth: Optional[int] = None,
               left_child: Optional['DecisionNode'] = None,
               right_child: Optional['DecisionNode'] = None,
               prediction: Optional[float] = None) -> None:
    """Initialize a decision node.

    Args:
      X (np.array): Optional. The dataset associated to the node. Only for
          BFS tree and 3-tree construction.
      y (np.ndarray): Optional. The dataset labels associated to the node and
          used for the leaf predictions. Only for BFS tree and 3-tree construction.
      gradients (np.array): The gradients for the dataset instances.
      index (int): Optional. An index for the feature on which the node splits.
          Default is None.
      value (Any): Optional. The corresponding value for that index. Default
          is None.
      depth (int): Optional. The depth for the node. Only for BFS tree
          construction.
      left_child (DecisionNode): Optional. The left child of the node, if any.
          Default is None.
      right_child (DecisionNode): Optional. The right child of the node, if any.
          Default is None.
      prediction (float): Optional. For a leaf node, holds the predicted value.
          Default is None.
    """
    # pylint: disable=invalid-name

    self.X = X
    self.y = y
    self.gradients = gradients
    self.index = index
    self.value = value
    self.depth = depth
    self.left_child = left_child
    self.right_child = right_child
    self.prediction = prediction
    self.processed = False


class DifferentiallyPrivateTree:
  """Implement a differentially private decision tree.

  Attributes:
    root_node (DecisionNode): The root node of the decision tree.
    nodes_bfs (List[DecisionNode]): All nodes in the tree.
    tree_index (int): The index of the tree being trained.
    learning_rate (float): The learning rate.
    l2_threshold (float): Threshold for leaf clipping.
    l2_lambda (float): Regularization parameter for l2 loss function.
    privacy_budget (float): The tree's privacy budget.
    delta_g (float): The utility function's sensitivity.
    delta_v (float): The sensitivity for leaf clipping.
    loss (LossFunction): An sklearn loss wrapper
        suitable for regression and classification.
    max_depth (int): Max. depth for the tree.
  """
  # pylint: disable=invalid-name,too-many-arguments

  def __init__(self,
               tree_index: int,
               learning_rate: float,
               l2_threshold: float,
               l2_lambda: float,
               privacy_budget: float,
               delta_g: float,
               delta_v: float,
               loss: LossFunction,
               max_depth: int = 6,
               max_leaves: Optional[int] = None,
               min_samples_split: int = 2,
               use_bfs: bool = False,
               use_3_trees: bool = False,
               cat_idx: Optional[List[int]] = None,
               num_idx: Optional[List[int]] = None) -> None:
    """Initialize the decision tree.

    Args:
      tree_index (int): The index of the tree being trained.
      learning_rate (float): The learning rate.
      l2_threshold (float): Threshold for leaf clipping.
      l2_lambda (float): Regularization parameter for l2 loss function.
      privacy_budget (float): The tree's privacy budget.
      delta_g (float): The utility function's sensitivity.
      delta_v (float): The sensitivity for leaf clipping.
      loss (LossFunction): An sklearn loss wrapper
          suitable for regression and classification.
          Valid options are: `LeastSquaresError` or `MultinomialDeviance`.
      max_depth (int): Optional. Max. depth for the tree. Default is 6.
      max_leaves (int): Optional. The max number of leaf nodes for the trees.
          Tree will grow in a best-leaf first fashion until it contains
          max_leaves or until it reaches maximum depth, whichever comes first.
      min_samples_split (int): Optional. The minimum number of samples required
          to split an internal node. Default is 2.
      use_bfs (bool): Optional. If max_leaves is specified, then this is
          automatically True. This will build the tree in a BFS fashion instead
          of DFS. Default is False.
      use_3_trees (bool): Optional. If True, only build trees that have 3
          nodes, and then assemble nb_trees based on these sub-trees, at random.
          Default is False.
      cat_idx (List): Optional. List of indices for categorical features.
      num_idx (List): Optional. List of indices for numerical features.
    """
    assert type(loss) in [LeastSquaresError, MultinomialDeviance]

    self.root_node = None  # type: Optional[DecisionNode]
    self.nodes_bfs = Queue()  # type: Queue[DecisionNode]
    self.nodes = []  # type: List[DecisionNode]
    self.tree_index = tree_index
    self.learning_rate = learning_rate
    self.l2_threshold = l2_threshold
    self.l2_lambda = l2_lambda
    self.privacy_budget = privacy_budget
    self.delta_g = delta_g
    self.delta_v = delta_v
    self.loss = loss
    self.max_depth = max_depth
    self.max_leaves = max_leaves
    self.min_samples_split = min_samples_split
    self.use_bfs = use_bfs
    self.use_3_trees = use_3_trees
    self.cat_idx = cat_idx
    self.num_idx = num_idx

    if self.max_leaves and not use_bfs:
      # If max_leaves is specified, we grow the tree in a best-leaf first
      # approach
      self.use_bfs = True

    # To keep track of total number of leaves in the tree
    self.current_number_of_leaves = 0
    self.max_leaves_reached = False

  def Fit(self, X: np.array, y: np.ndarray, gradients: np.array) -> None:
    """Fit the tree to the data.

    Args:
      X (np.array): The dataset.
      y (np.ndarray): The dataset labels.
      gradients (np.array): The gradients for the dataset instances.
    """

    # Construct the tree recursively
    if self.use_bfs:
      self.root_node = self.MakeTreeBFS(X, y, gradients)
    else:
      depth = 1 if self.use_3_trees else self.max_depth
      self.root_node = self.MakeTreeDFS(X, y, gradients, depth)

    leaves = [node for node in self.nodes if node.prediction]

    # Clip the leaf nodes
    ClipLeaves(leaves, self.l2_threshold, self.learning_rate, self.tree_index)

    # Add noise to the predictions
    privacy_budget_for_leaf_node = self.privacy_budget / 2
    laplace_scale = self.delta_v / privacy_budget_for_leaf_node
    AddLaplacianNoise(leaves, laplace_scale)

    # Shrink by learning rate
    Shrink(leaves, self.learning_rate)

  def MakeTreeDFS(self,
                  X: np.array,
                  y: np.ndarray,
                  gradients: np.array,
                  depth: int) -> DecisionNode:
    """Build a tree recursively, in DFS fashion.

    Args:
      X (np.array): The dataset.
      y (np.ndarray): The dataset labels.
      gradients (np.array): The gradients for the dataset instances.
      depth (int): Current depth for the tree (reversed).

    Returns:
      DecisionNode: A decision node.
    """

    if depth == 0 or len(X) < self.min_samples_split:
      # Max depth reached or not enough samples to split node, node is a leaf
      # node
      if self.use_3_trees:
        node = DecisionNode(X=X,
                            y=y,
                            gradients=gradients,
                            prediction=self.GetLeafPrediction(gradients, y))
      else:
        node = DecisionNode(prediction=self.GetLeafPrediction(gradients, y))
      self.nodes.append(node)
      return node

    best_split = self.FindBestSplit(X, gradients)
    if best_split:
      lhs_op, rhs_op = self.GetOperators(best_split['index'])
      lhs = np.where(lhs_op(X[:, best_split['index']], best_split['value']))[0]
      rhs = np.where(rhs_op(X[:, best_split['index']], best_split['value']))[0]
      left_child = self.MakeTreeDFS(X[lhs], y[lhs], gradients[lhs], depth - 1)
      right_child = self.MakeTreeDFS(X[rhs], y[rhs], gradients[rhs], depth - 1)
      if self.use_3_trees:
        node = DecisionNode(X=X,
                            y=y,
                            gradients=gradients,
                            index=best_split['index'],
                            value=best_split['value'],
                            left_child=left_child,
                            right_child=right_child)
      else:
        node = DecisionNode(index=best_split['index'],
                            value=best_split['value'],
                            left_child=left_child,
                            right_child=right_child)
      self.nodes.append(node)
      return node

    if self.use_3_trees:
      node = DecisionNode(X=X,
                          y=y,
                          gradients=gradients,
                          prediction=self.GetLeafPrediction(gradients, y))
    else:
      node = DecisionNode(prediction=self.GetLeafPrediction(gradients, y))
    self.nodes.append(node)
    return node

  def MakeTreeBFS(self,
                  X: np.array,
                  y: np.ndarray,
                  gradients: np.array) -> DecisionNode:
    """Build a tree in a best-leaf first fashion.

    Args:
      X (np.array): The dataset.
      y (np.ndarray): The dataset labels.
      gradients (np.array): The gradients for the dataset instances.

    Returns:
      DecisionNode: A decision node.
    """

    best_split = self.FindBestSplit(X, gradients)
    if not best_split:
      node = DecisionNode(prediction=self.GetLeafPrediction(gradients, y))
      self.nodes.append(node)
      return node

    # Root node
    node = DecisionNode(X=X,
                        y=y,
                        gradients=gradients,
                        index=best_split['index'],
                        value=best_split['value'],
                        depth=0)
    self.nodes.append(node)
    self.nodes_bfs.put(node)
    self._ExpandTreeBFS()
    for node in self.nodes:
      # Assigning predictions to remaining leaf nodes if we had to stop
      # constructing the tree early because we reached max number of leaf nodes
      if not node.prediction and not node.left_child and not node.right_child:
        node.prediction = self.GetLeafPrediction(node.gradients, node.y)
    return node

  def _ExpandTreeBFS(self) -> None:
    """Expand a tree in a best-leaf first fashion.

    Implement https://researchcommons.waikato.ac.nz/bitstream/handle/10289/2317
    /thesis.pdf?sequence=1&isAllowed=y
    """

    # Node queue is empty or too many leaves, stopping
    if self.nodes_bfs.empty() or self.max_leaves_reached:
      return None

    current_node = self.nodes_bfs.get()

    # If there are not enough samples to split in that node, make it a leaf
    # node and process next node
    assert current_node.gradients is not None
    if len(current_node.gradients) < self.min_samples_split:
      self._MakeLeaf(current_node)
      if not self._IsMaxLeafReached():
        return self._ExpandTreeBFS()
      return None

    # If we reached max depth
    if current_node.depth == self.max_depth:
      self._MakeLeaf(current_node)
      if not self._IsMaxLeafReached():
        if self.max_leaves:
          return self._ExpandTreeBFS()
        while not self.nodes_bfs.empty():
          node = self.nodes_bfs.get()
          self._MakeLeaf(node)
      return None

    # Do the split
    assert current_node.X is not None
    assert current_node.y is not None
    assert current_node.gradients is not None
    lhs_op, rhs_op = self.GetOperators(current_node.index)  # type: ignore
    lhs = np.where(
        lhs_op(current_node.X[:, current_node.index], current_node.value))[0]
    rhs = np.where(
        rhs_op(current_node.X[:, current_node.index], current_node.value))[0]
    lhs_X, rhs_X = current_node.X[lhs], current_node.X[rhs]
    lhs_grad, rhs_grad = current_node.gradients[lhs], current_node.gradients[
        rhs]
    lhs_y, rhs_y = current_node.y[lhs], current_node.y[rhs]
    lhs_best_split = self.FindBestSplit(lhs_X, lhs_grad)
    rhs_best_split = self.FindBestSplit(rhs_X, rhs_grad)

    # Can't split the node, so this becomes a leaf node.
    if not lhs_best_split or not rhs_best_split:
      self._MakeLeaf(current_node)
      if not self._IsMaxLeafReached():
        return self._ExpandTreeBFS()
      return None

    # Splitting the node is possible, creating the children
    assert current_node.depth is not None
    left_child = DecisionNode(X=lhs_X,
                              y=lhs_y,
                              gradients=lhs_grad,
                              index=lhs_best_split['index'],
                              value=lhs_best_split['value'],
                              depth=current_node.depth + 1)
    right_child = DecisionNode(X=rhs_X,
                               y=rhs_y,
                               gradients=rhs_grad,
                               index=rhs_best_split['index'],
                               value=rhs_best_split['value'],
                               depth=current_node.depth + 1)

    current_node.left_child = left_child
    current_node.right_child = right_child
    self.nodes.append(current_node)

    # Adding them to the list of nodes for further expansion in best-gain order
    if lhs_best_split['gain'] >= rhs_best_split['gain']:
      self.nodes_bfs.put(left_child)
      self.nodes_bfs.put(right_child)
    else:
      self.nodes_bfs.put(right_child)
      self.nodes_bfs.put(left_child)
    return self._ExpandTreeBFS()

  def _MakeLeaf(self, node: DecisionNode) -> None:
    """Make a node a leaf node.

    Args:
      node (DecisionNode): The node to make a leaf from.
    """
    node.prediction = self.GetLeafPrediction(node.gradients, node.y)
    self.current_number_of_leaves += 1
    self.nodes.append(node)

  def _IsMaxLeafReached(self) -> bool:
    """Check if we reached maximum number of leaf nodes.

    Returns:
      bool: True if we reached the maximum number of leaf nodes,
          False otherwise.
    """
    leaf_candidates = 0
    for node in list(self.nodes_bfs.queue):
      if not node.left_child and not node.right_child:
        leaf_candidates += 1
    if self.max_leaves:
      if self.current_number_of_leaves + leaf_candidates >= self.max_leaves:
        self.max_leaves_reached = True
    return self.max_leaves_reached

  def FindBestSplit(self,
                    X: np.array,
                    gradients: np.array) -> Optional[Dict[str, Any]]:
    """Find best split of data using the exponential mechanism.

    Args:
      X (np.array): The dataset.
      gradients (np.array): The gradients for the dataset instances.

    Returns:
      Optional[Dict[str, Any]]: A dictionary containing the split
          information, or none if no split could be done.
    """

    # Depth + 1 because root node is at depth 0
    privacy_budget_for_node = np.around(np.divide(self.privacy_budget/2,
                                        self.max_depth + 1), decimals=7)
    probabilities = []
    max_gain = -np.inf
    # Iterate over features
    for feature_index in range(X.shape[1]):
      # Iterate over unique value for this feature
      for value in np.unique(X[:, feature_index]):
        # Find gain for that split
        gain = self.ComputeGain(feature_index, value, X, gradients)
        # Compute probability for exponential mechanism
        exp_gain = (privacy_budget_for_node * gain) / (2. * self.delta_g)
        if exp_gain > max_gain:
          max_gain = exp_gain
        prob = {
            'index': feature_index,
            'value': value,
            'gain': exp_gain
        }
        probabilities.append(prob)
    return ExponentialMechanism(probabilities, max_gain)

  def GetLeafPrediction(self, gradients: np.array, y: np.ndarray) -> float:
    """Compute the leaf prediction.

    Args:
      gradients (np.array): The gradients for the dataset instances.
      y (np.ndarray): The dataset labels.

    Returns:
      float: The prediction for the leaf node
    """
    return ComputePredictions(gradients, y, self.loss, self.l2_lambda)

  def Predict(self, X: np.array) -> np.array:
    """Return predictions for a list of input data.

    Args:
      X: The input data used for prediction.

    Returns:
      np.array: An array with the predictions.
    """
    predictions = []
    for row in X:
      predictions.append(self._Predict(row, self.root_node))  # type: ignore
    return np.asarray(predictions)

  def _Predict(self, row: np.array, node: DecisionNode) -> float:
    """Walk through the decision tree to output a prediction for the row.

    Args:
      row (np.array): The row to classify.
      node (DecisionNode): The current decision node.

    Returns:
      float: A prediction for the row.
    """
    if node.prediction is not None:
      return node.prediction
    value = row[node.index]
    if value >= node.value:
      child_node = node.right_child
    else:
      child_node = node.left_child
    return self._Predict(row, child_node)  # type: ignore

  def ComputeGain(self,
                  index: int,
                  value: Any,
                  X: np.array,
                  gradients: np.array) -> float:
    """Compute the gain for a given split.

    See https://dl.acm.org/doi/pdf/10.1145/2939672.2939785

    Args:
      index (int): The index for the feature to split on.
      value (Any): The feature's value to split on.
      X (np.array): The dataset.
      gradients (np.array): The gradients for the dataset instances.

    Returns:
      float: The gain for the split.
    """
    lhs_op, rhs_op = self.GetOperators(index)
    lhs = np.where(lhs_op(X[:, index], value))[0]
    rhs = np.where(rhs_op(X[:, index], value))[0]
    lhs_grad, rhs_grad = gradients[lhs], gradients[rhs]
    lhs_gain = np.square(np.sum(lhs_grad)) / (
        len(lhs) + self.l2_lambda)  # type: float
    rhs_gain = np.square(np.sum(rhs_grad)) / (
        len(rhs) + self.l2_lambda)  # type: float
    # TODO: friedman_mse ???
    # Total gain can be omitted since it doesn't depend on splitting value
    return lhs_gain + rhs_gain

  def GetOperators(self, index: int) -> Tuple[Any, Any]:
    """Return operators to use to split a node's dataset.

    Args:
      index (int): The index for the feature to split the data on.

    Returns:
      Tuple[Any, Any]: The operators to use.
    """
    if self.cat_idx and index in self.cat_idx:
      # Categorical feature
      return operator.eq, operator.ne
    # Numerical feature
    return operator.lt, operator.ge


def ClipLeaves(leaves: List[DecisionNode],
               l2_threshold: float,
               learning_rate: float,
               tree_index: int) -> None:
  """Clip leaf nodes.

  If the prediction is higher than the threshold, set the prediction to
  that threshold.

  Args:
    leaves (List[DecisionNode]): The leaf nodes.
    l2_threshold (float): Threshold of the l2 loss function.
    learning_rate (float): The learning rate.
    tree_index (int): The index for the current tree.
  """
  threshold = l2_threshold * math.pow((1 - learning_rate), tree_index)
  for leaf in leaves:
    assert leaf.prediction is not None
    if np.abs(leaf.prediction) > threshold:
      if leaf.prediction > 0:
        leaf.prediction = threshold
      else:
        leaf.prediction = -1 * threshold


def AddLaplacianNoise(leaves: List[DecisionNode],
                      scale: float) -> None:
  """Add laplacian noise to the leaf nodes.

  Args:
    leaves (List[DecisionNode]): The list of leaves.
    scale (float): The scale to use for the laplacian distribution.
  """
  # Comment the line below when using the model. This is for stability for
  # cross-validation tests only.
  np.random.seed(0)
  for leaf in leaves:
    noise = np.random.laplace(0, scale)
    leaf.prediction += noise


def Shrink(leaves: List[DecisionNode],
           learning_rate: float) -> None:
  """Shrink leaves by learning_rate

  Args:
    leaves (List[DecisionNode]): List of leaf nodes.
    learning_rate (float): The learning rate by which to shrink.
  """
  for leaf in leaves:
    assert leaf.prediction is not None
    leaf.prediction *= learning_rate


def ComputePredictions(gradients: np.ndarray,
                       y: np.ndarray,
                       loss: LossFunction,
                       l2_lambda: float) -> float:
  """Computes the predictions of a leaf.

  Used in the `DifferentiallyPrivateTree` as well as in `SplitNode`
  for the 3-tree version.

  Ref:
    Friedman 01. "Greedy function approximation: A gradient boosting machine."
      (https://projecteuclid.org/euclid.aos/1013203451)

  Args:
    gradients (np.ndarray): The positive gradients y˜ for the dataset instances.
    y (np.ndarray): The dataset labels y.
    loss (LossFunction): An sklearn loss wrapper
        suitable for regression and classification.
    l2_lambda (float): Regularization parameter for l2 loss function.

  Returns:
    Prediction γ of a leaf
  """
  if len(gradients) == 0:
    prediction = 0.
  elif loss.is_multi_class:
    # sum of neg. gradients divided by sum of 2nd derivatives
    # aka one Newton-Raphson step
    # for details ref. (eq 33+34) in Friedman 01.
    prediction = -1 * np.sum(gradients) * (loss.K - 1) / loss.K  # type: float
    denom = np.sum((y + gradients) * (1 - y - gradients))
    prediction = 0 if abs(denom) < 1e-150 else prediction / (denom + l2_lambda)  # just to make sure
    # TODO: Verify on whether this l2-regularization is correct as is
  else:
    prediction = (-1 * np.sum(gradients) / (len(
      gradients) + l2_lambda))  # type: float
  return prediction


def ExponentialMechanism(
    probabilities: List[Dict[str, Any]],
    max_gain: float,
    reverse: bool = False) -> Optional[Dict[str, Any]]:
  """Apply the exponential mechanism.

  Args:
    probabilities (List[Dict]): List of probabilities to choose from.
    max_gain (float): The maximum gain amongst all probabilities in the list.
    reverse (bool): Optional. If True, sort probabilities in reverse order (
        i.e. lower gains are better).

  Returns:
    Dict: a candidate (i.e. probability) from the list.
  """
  with np.errstate(all='raise'):
    try:
      sum_probabilities = np.sum(
          np.exp(prob['gain']) for prob in probabilities)
      for prob in probabilities:
        # e^0 is 1, so checking for that
        if prob['gain'] == 0.:
          prob['probability'] = 0.
        else:
          prob['probability'] = np.exp(prob['gain']) / sum_probabilities
    # Happens when np.sum() overflows because of a gain that's too high
    except FloatingPointError:
      for prob in probabilities:
        gain = prob['gain']
        if gain != 0.:
          # Check if the gain of each candidate is too small compared to
          # the max gain seen up until now. If so, set the probability for
          # this split to 0.
          try:
            _ = np.exp(max_gain - gain)
          except FloatingPointError:
            prob['probability'] = 0.
          # If it's not too small, we need to compute a new sum that
          # doesn't overflow. For that we only take into account 'large'
          # gains with respect to the current candidate. If again the
          # difference is so small that it would still overflow, we set the
          # probability for this split to 0.
          sum_prob = 0.
          for prob_ in probabilities:
            gain_ = prob_['gain']
            if gain_ != 0.:
              try:
                sum_prob += np.exp(gain_ - gain)
              except FloatingPointError:
                prob['probability'] = 0.
                break
          # Other candidates compare similarly, so we can compute a
          # probability. If it underflows, set it to 0 as well.
          if sum_prob != 0.:
            try:
              prob['probability'] = 1.0 / sum_prob
            except FloatingPointError:
              prob['probability'] = 0.
        else:
          prob['probability'] = 0.

  if (np.asarray([prob['gain'] for prob in probabilities]) <= 0.0).all():
    # No split offers a positive gain, node should be a leaf node
    return None

  # Apply the exponential mechanism
  previous_prob = 0.
  random_prob = np.random.uniform()
  # Sort probabilities by ascending order of gain so that higher gains
  # split will get higher probability
  for prob in sorted(
      probabilities, key=lambda d: d['probability'], reverse=reverse):
    prob['probability'] += previous_prob
    previous_prob = prob['probability']
    op = operator.ge if not reverse else operator.le
    if op(prob['probability'], random_prob):
      return prob
  return None
