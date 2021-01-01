import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from estimator import DPGBDT

if __name__ == '__main__':
  N = 500
  X = np.random.normal(
      loc=[(-1, 1), (2, 5), (4, -4)],
      scale=[(1, 1), (0.5, 0.5), (1.5, 1.5)],
      size=(N, 3, 2),
  ).reshape((-1, 2))
  y = np.arange(3)[np.newaxis, :].repeat(N, axis=0).reshape(-1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

  model = DPGBDT(
      privacy_budget=0.1,
      n_classes=len(set(y_train)),
      nb_trees=50,
      nb_trees_per_ensemble=50,
      max_depth=3,
      use_3_trees=False,
      learning_rate=0.1,
  )
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print(classification_report(y_test, y_pred))
  print("Score %.4f" % np.mean(y_test == y_pred))

  y_pred = model.predict(X_train)
  print("Score %.4f (train)" % np.mean(y_train == y_pred))
