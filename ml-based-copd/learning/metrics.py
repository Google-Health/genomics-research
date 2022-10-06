# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Custom `tf.metrics.Metric` implementations and loaders."""
from typing import Any, Dict, Optional, Type, Union

import ml_collections
import tensorflow as tf

# A registry dictionary mapping custom metric names to classes.
_METRIC_REGISTRY: Dict[str, Type[tf.metrics.Metric]] = {}

KerasIdentifier = Dict[str, Union[str, Dict[str, Any]]]


def register_metric(cls: Type[tf.metrics.Metric]) -> Type[tf.metrics.Metric]:
  """Registers a class definition` in `_METRIC_REGISTRY` as `cls.__name__`.

  Args:
    cls: The class type to register.

  Returns:
    The registered class type.
  """
  class_name = cls.__name__
  if class_name in _METRIC_REGISTRY:
    raise ValueError(f'"{class_name}" already registered: {_METRIC_REGISTRY}')
  _METRIC_REGISTRY[class_name] = cls
  return cls


@register_metric
class PearsonCorrelationOnline(tf.metrics.Metric):
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
      sample_weight: A tensor of shape (None, 1) containing weights in `{0, 1}`;
        used to mask observations from correlation computation.

    Raises:
      tf.errors.InvalidArgumentError: On `y_true` and `y_pred` shape mismatch.
      tf.errors.InvalidArgumentError: If input tensors are not shape (None, 1).
      tf.errors.InvalidArgumentError: If `sample_weight` is not `None` and
        either 1) is not shape (None, 1), 2) does not match `y_true`'s shape, or
        3) contains elements not in `{0, 1}`.
    """

    tf.ensure_shape(y_pred, y_true.shape)
    tf.ensure_shape(y_pred, (None, 1))
    x = tf.cast(y_pred, tf.float32)
    y = tf.cast(y_true, tf.float32)

    if sample_weight is not None:
      # Ensure that `sample_weight` has the correct shape and only contains
      # elements in `{0, 1}`.
      tf.ensure_shape(sample_weight, (None, 1))
      tf.ensure_shape(sample_weight, y_true.shape)
      sample_weight = tf.cast(sample_weight, 'float32')
      is_one = tf.math.equal(sample_weight, 1.0)
      is_zero = tf.math.equal(sample_weight, 0.0)
      is_one_or_zero = tf.math.logical_or(is_one, is_zero)
      tf.assert_equal(is_one_or_zero, True,
                      'Found `sample_weight` value not in {0, 1}.')

      # Only keep observations that have `sample_weight == 1`.
      # Note: `tf.boolean_mask` flattens the given tensor, so we must add the
      # last dimension back using `tf.expand_dims`. This is safe since we are
      # guarnateed inputs of shape (None, 1).
      x = tf.expand_dims(tf.boolean_mask(x, is_one), axis=1)
      y = tf.expand_dims(tf.boolean_mask(y, is_one), axis=1)

    # Only update metrics if some observations are unmasked, i.e., the current
    # batch is not empty.
    if tf.not_equal(tf.size(x), 0):
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


def _tf_metrics_get(identifier: KerasIdentifier) -> tf.metrics.Metric:
  """Returns a `tf.metrics.Metric` instance corresponding to `identifier`.

  This functions wraps both `_METRIC_REGISTRY` and `tf.metrics.get`,
  prioritizing custom metrics in the registry. If a given `identifier`'s
  `class_name` is not found in the registry, we fallback to `tf.metrics.get` and
  fetch the standard tf.keras implementation.

  Args:
    identifier: A keras identifier containing `class_name` and `config` keys.

  Returns:
    A metric instance.
  """
  # If `class_name` is a custom metric, use the registry's implementation.
  class_name = identifier['class_name']
  if class_name in _METRIC_REGISTRY:
    return _METRIC_REGISTRY[class_name](**identifier['config'])

  return tf.metrics.get(identifier)


def get_metric(metric_config: ml_collections.ConfigDict) -> tf.metrics.Metric:
  """Returns a metric for use in model training and evaluation.

  The `metric_config` `ConfigDict` must match the config schema defined in the
  README.

  Note: This function wraps `tf.metrics.get()` and the custom metric registry.
  Any `tf.metrics.Metric` registered with `@register_metric()` or any class from
  https://www.tensorflow.org/api_docs/python/tf/keras/metrics can be loaded
  with this utility. See individual class implementations for the set of
  available keyword arguments that can be configured via `metric_config.kwargs`.

  For example:

  ```
  valid_metric_config = ml_collections.ConfigDict({
      'class_name': 'RecallAtPrecision',
      'kwargs': {
          'precision': 0.8,
          'num_thresholds': 200,
      },
  })
  metric = get_metric(valid_metric_config)
  ```

  Args:
    metric_config: A ConfigDict matching the `metric_config` schema.

  Returns:
    A metric instance.
  """
  # Build an identifier dict that can be parsed by `tf.metrics.get`.
  identifier_kwargs = metric_config.get('kwargs', ml_collections.ConfigDict())
  identifier = {
      'class_name': metric_config['class_name'],
      'config': identifier_kwargs.to_dict()
  }
  return _tf_metrics_get(identifier)
