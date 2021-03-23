# Copyright 2021 Google LLC.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Utilities for building outcome metrics and losses."""
from typing import Dict, List, Optional

import ml_collections
import tensorflow as tf


class PearsonCorrelationOnline(tf.keras.metrics.Metric):
  """An online and batched implementation of Pearson correlation.

  This is a variation of Welford's algorithm. This metric calculates running
  mean, variance, and covariance in an online, vectorized manner. On `result()`,
  this class uses these running values to compute the Pearson correlation.

  For some variable `x`, with batches `x_1, x_2, ..., x_n`, the batched running
  mean and variance can be formulated using the following recurrence relations
  for `1 < k <= n`:

  ```
  total_1 = size(x_1)
  total_k = size(x_k) + total_k-1

  mean_x_1 = sum(x_1) / total_1
  mean_x_k = mean_x_k-1 + ((x_k - mean_x_k-1) / total_k)

  var_sum_x_1 = 0
  var_sum_x_k = var_sum_x_k-1 + ((x_k - mean_x_k-1) * (x_k - mean_x_k))
  var_x_k = var_sum_x_k / (total_k - 1)
  ```

  Assuming we have two variables, `x` and `y`, and are maintaing a running mean
  for each, denoted `mean_y_k` and `mean_x_k`, we can calculate a running
  covariance in a similar manner:

  ```
  cov_sum_xy_1 = 0
  cov_sum_xy_k = cov_sum_xy_k-1 + ((y_k - mean_y_k-1) * (x_k - mean_x_k))
  cov_xy_k = cov_sum_xy_k / (total_k - 1)
  ```

  The Pearson correlation between `x` and `y` at time `k` can then be calculated
  using the running variances, denoted `var_x_k` and `var_y_k`, and covariance:

  ```
  corr_xy_k = cov_xy_k / (std_x_k * std_y_k)
            = cov_xy_k / (sqrt(var_x_k) * sqrt(var_y_k))
  ```

  Note: When calculating the Pearson correlation on `result()`, we skip dividing
  var_sum_x, var_sum_y, and cov_sum_xy by `n = total_k - 1`, which would give us
  the true variances and covariances for each variable. Dropping the subscript
  `k`, we have:

  ```
  corr_xy = cov_xy / (std_x * std_y)
          = cov_xy / (sqrt(var_x) * sqrt(var_y))
          = (cov_sum_xy / n) / (sqrt(var_sum_x / n) * sqrt(var_sum_y / n))
          = cov_sum_xy / (sqrt(var_sum_x) * sqrt(var_sum_y))
          = cov_sum_xy / sqrt(var_sum_x * var_sum_y)
  ```

  Avoiding these extra operations gives more numerical stability.
  """

  def __init__(self, name: str = 'pearson_correlation_online', **kwargs):
    super(PearsonCorrelationOnline, self).__init__(name=name, **kwargs)
    # The running count of examples.
    self.count = self.add_weight(
        name='count', initializer='zeros', dtype=tf.float32)

    # The running means for predictions and labels.
    self.mean_x = self.add_weight(
        name='mean_x', initializer='zeros', dtype=tf.float32)
    self.mean_y = self.add_weight(
        name='mean_y', initializer='zeros', dtype=tf.float32)

    # The running sums for prediction and labels used to calculate variance.
    self.var_sum_x = self.add_weight(
        name='var_sum_x', initializer='zeros', dtype=tf.float32)
    self.var_sum_y = self.add_weight(
        name='var_sum_y', initializer='zeros', dtype=tf.float32)

    # The running sum used calculate prediction and label covariance.
    self.cov_sum_xy = self.add_weight(
        name='cov_sum_xy', initializer='zeros', dtype=tf.float32)

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> None:
    """Update running mean, variance, and covariance sums with new observations.

    Args:
      y_true: A tensor containing the ground truth labels.
      y_pred: A tensor containing predicted labels.
      sample_weight: A tensor of shape (None, 1) containing weights in {0, 1}
        used to mask observations.

    Raises:
      tf.errors.InvalidArgumentError: On `y_true` and `y_pred` shape mismatch.
      tf.errors.InvalidArgumentError: If input tensors are not shape (None, 1).
      tf.errors.InvalidArgumentError: If `sample_weight` is not `None` and is
        not shape (None, 1).
      tf.errors.InvalidArgumentError: If `sample_weight` is not `None` and
        contains elements not in {0, 1}.
    """

    tf.ensure_shape(y_pred, y_true.shape)
    tf.ensure_shape(y_pred, (None, 1))
    x = tf.cast(y_pred, tf.float32)
    y = tf.cast(y_true, tf.float32)

    if sample_weight is not None:
      # Ensure that `sample_weight` has the correct shape and only contains
      # elements in {0, 1}.
      tf.ensure_shape(sample_weight, (None, 1))
      sample_weight = tf.cast(sample_weight, 'float32')
      is_one = tf.math.equal(sample_weight, 1.0)
      is_zero = tf.math.equal(sample_weight, 0.0)
      is_one_or_zero = tf.math.logical_or(is_one, is_zero)
      tf.debugging.assert_equal(is_one_or_zero, True,
                                'Found `sample_weight` value not in {0, 1}.')

      # Only keep observations that have `sample_weight == 1`.
      x = tf.boolean_mask(x, is_one)
      y = tf.boolean_mask(y, is_one)

    # Update running count.
    count_new = tf.cast(tf.size(x), tf.float32)
    count_total = self.count.assign_add(count_new, read_value=True)

    # Update running mean and variance sum for predictions.
    delta_x_old = x - self.mean_x
    self.mean_x.assign_add(tf.reduce_sum(delta_x_old) / count_total)
    delta_x_new = x - self.mean_x
    self.var_sum_x.assign_add(tf.reduce_sum(delta_x_old * delta_x_new))

    # Update running mean and variance sum for labels.
    delta_y_old = y - self.mean_y
    self.mean_y.assign_add(tf.reduce_sum(delta_y_old) / count_total)
    delta_y_new = y - self.mean_y
    self.var_sum_y.assign_add(tf.reduce_sum(delta_y_old * delta_y_new))

    # Update running covariance sum for predictions and labels.
    self.cov_sum_xy.assign_add(tf.reduce_sum(delta_x_old * delta_y_new))

  def result(self) -> tf.Tensor:
    """Computes the Pearson correlation using running statistics.

    Returns:
      A tensor containing the running Pearson correlation.
    """
    return self.cov_sum_xy / tf.sqrt(self.var_sum_x * self.var_sum_y)


def _get_classification_metrics() -> List[tf.metrics.Metric]:
  """Returns a list of metrics for use with classification tasks."""
  metrics = [
      tf.keras.metrics.CategoricalAccuracy(name='cat_accuracy'),
  ]
  return metrics


def _get_regression_metrics() -> List[tf.metrics.Metric]:
  """Returns a list of metrics for use with regression tasks."""
  metrics = [
      tf.keras.metrics.MeanSquaredError(name='mse'),
      tf.keras.metrics.RootMeanSquaredError(name='rmse'),
      tf.keras.metrics.MeanAbsoluteError(name='mae'),
      PearsonCorrelationOnline(),
  ]
  return metrics


def get_metrics(
    outcomes: List[ml_collections.ConfigDict],
) -> Dict[str, List[tf.metrics.Metric]]:
  """Returns a dictionary mapping outcome names to metric lists.

  A list of metrics is generated for each outcome, and the dictionary is keyed
  on each outcome's name.

  Args:
    outcomes: A list of outcome ConfigDicts.

  Returns:
    A dictionary containg lists of metrics keyed on outcome name.
  """
  metrics_dict = {}

  for outcome in outcomes:
    if outcome.type == 'regression':
      metrics_dict[outcome.name] = _get_regression_metrics()
    elif outcome.type == 'classification':
      metrics_dict[outcome.name] = _get_classification_metrics()
    else:
      raise NotImplementedError

  return metrics_dict


def get_loss(config: ml_collections.ConfigDict) -> tf.losses.Loss:
  """Returns a loss for use in training and evaluation.

  Args:
    config: A ConfigDict containing a `config.loss` name.

  Returns:
    A loss function.

  Raises:
    ValueError: `config.loss` is missing or an unknown loss name.
  """
  loss_name = config.get('loss', None)

  if loss_name == 'ce':
    return tf.keras.losses.CategoricalCrossentropy(from_logits=False)

  if loss_name == 'bce':
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)

  if loss_name == 'mse':
    return tf.keras.losses.MeanSquaredError()

  raise ValueError(f'Unknown loss name: {loss_name}')
