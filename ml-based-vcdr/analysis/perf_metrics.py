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
"""Utilities for calculating and displaying evaluation metrics."""

import collections
from typing import Dict, List, Text

import numpy as np
import scipy as sp
import sklearn.metrics


def compute_frequency(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      top_percentile: int = 100) -> float:
  """Computes positive class frequency at a given top percentile.

  100 indicates the frequency for all samples.

  Args:
    y_true: Ground truth (correct) target values.
    y_pred: Estimated targets as returned by a classifier.
    top_percentile: Determines the set of examples considered in the frequency
      calculation. The top percentile represents the top percentile by
      prediction risk. 100 indicates using all samples.

  Returns:
    A [0.0, 1.0] float corresponding to the positive class frequency in the top
    percentile.
  """
  pred_top_percentile = np.percentile(y_pred, 100 - top_percentile)
  mask_top_percentile = y_pred >= pred_top_percentile
  return np.mean(y_true[mask_top_percentile])


class Metric(object):
  """A callable wrapper class for a named metric function.

  The provided function is expected to accept `y_true` ground truth labels and
  `y_pred` model predictions. Additional arguments can be supplied via kwargs.
  The function is invoked when the `Metric` object is called.
  """

  def __init__(self, name: Text, func, binary_only: bool) -> None:
    self.name = name
    self.func = func
    self.binary_only = binary_only

  def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
    """Invokes the Metric's `func`.

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.
      **kwargs: Additional keyword arguments passed to the Metric's `func`.

    Returns:
      The result of the Metric's `func`.
    """
    if self.binary_only and not np.array_equal(
        np.unique(y_true), np.asarray([0.0, 1.0])):
      raise ValueError(
          '`y_true` must be in {0, 1} and have elements in each class.')

    metric_val = self.func(y_true, y_pred, **kwargs)

    # some `func`s return p-values as well, which we ignore.
    if isinstance(metric_val, tuple):
      metric_val = metric_val[0]

    return metric_val

  def __str__(self) -> Text:
    return self.name


BootstrapResults = collections.namedtuple('BootstrapResults',
                                          'mean std conf_int ci_lo ci_hi')


class TopPercentileMetric(Metric):
  """A callable wrapper class for a named metric top percentile function.

  The provided function is expected to accept `y_true` ground truth labels,
  `y_pred` model predictions, and a top percentile from which to calculate
  metrics. Additional arguments can be supplied via args. The function is
  invoked when the `TopPercentileMetric` object is called.
  """

  def __init__(self,
               name: Text,
               func,
               binary_only: bool,
               top_percentile: int = 100):
    super(TopPercentileMetric, self).__init__(
        name, func, binary_only=binary_only)
    self.top_percentile = top_percentile

  def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
    """Invokes the TopPercentileMetric's `func`, passing `top_percentile`.

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.
      **kwargs: Additional keyword arguments passed to the TopPercentileMetric's
        `func`.

    Returns:
      The result of the TopPercentileMetric's `func`.
    """
    return super(TopPercentileMetric, self).__call__(
        y_true, y_pred, top_percentile=self.top_percentile, **kwargs)


def _build_default_metrics(binary: bool) -> List[Metric]:
  """Builds and returns the default set of `Metric`s."""

  metrics = [
      Metric('num', lambda y_true, y_pred: len(y_true), binary_only=binary)
  ]

  if binary:
    metrics.extend([
        Metric('auc', sklearn.metrics.roc_auc_score, binary_only=True),
        Metric(
            'auprc', sklearn.metrics.average_precision_score, binary_only=True),
        TopPercentileMetric(
            'freq', compute_frequency, binary_only=True, top_percentile=100),
    ])
    for top_percentile in [10, 5, 1]:
      metrics.append(
          TopPercentileMetric(
              'freq @{:04.1f}'.format(top_percentile),
              compute_frequency,
              binary_only=True,
              top_percentile=top_percentile))
  else:
    metrics.extend([
        Metric('pearson', sp.stats.pearsonr, binary_only=False),
        Metric('spearman', sp.stats.spearmanr, binary_only=False),
        Metric('mse', sklearn.metrics.mean_squared_error, binary_only=False),
        Metric('mae', sklearn.metrics.mean_absolute_error, binary_only=False),
    ])
  return metrics


def _build_metric_str(metric_name: Text, results: BootstrapResults) -> Text:
  r"""Returns a formatted string for pretty printing a given metric value.

  If a std and confidence interval are supplied, the string will be of the form:
    `\t<metric_name>:\t<mean> (<std>; <conf_int>% CI:<ci_lo> - ci_hi>)`.

  If not, the std and ci are not included:
    `\t<metric_name>:\t<mean>`.

  Args:
    metric_name: A metric name string.
    results: A `BootstrapResults` namedtuple.

  Returns:
    A formatted string.
  """
  string = ['\t{}:\t'.format(metric_name)]
  if isinstance(results.mean, float):
    string.append('{:.4f}'.format(results.mean))
  else:
    string.append('{}'.format(results.mean))
  if results.std > 0:
    string.append(' ({:.4f}; {:.0f}% CI: {:.4f}-{:.4f})'.format(
        results.std, results.conf_int, results.ci_lo, results.ci_hi))
  return ''.join(string)


def _bootstrap(metric: Metric, y_true: np.ndarray, y_pred: np.ndarray,
               n_bootstrap: int, conf_interval: float, seed: int,
               **kwargs) -> BootstrapResults:
  """Performs bootstrapping on a given `Metric`.

  Args:
    metric: An instance of a `Metric`.
    y_true: Ground truth (correct) target values.
    y_pred: Estimated targets as returned by a classifier.
    n_bootstrap: An integer denoting the number of bootstrap iterations.
    conf_interval: A float denoting the width of confidence interval.
    seed: An int denoting the seed for the PRNG.
    **kwargs: Additional keyword arguments passed to each Metric's `func`.

  Returns:
    A BootstrapResults namedtuple tuple of the mean, standard deviation, and
    lower and upper bounds for conf_interval of the `Metric` over
   `n_bootstrap` bootstrapping iterations. If n_bootstrap=0, i.e., no
    bootstrapping is used, the returned standard deviation and lower and upper
    bounds are numpy.nan.
  """
  if n_bootstrap == 0:
    return BootstrapResults(
        metric(y_true, y_pred, **kwargs), np.nan, np.nan, np.nan, np.nan)

  prng = np.random.RandomState(seed)
  lo_perc = (100 - conf_interval) / 2
  hi_perc = 100 - lo_perc

  metrics = []
  num_observations = len(y_pred)
  while len(metrics) < n_bootstrap:
    idx = prng.randint(0, high=num_observations, size=num_observations)
    sample_true = y_true[idx]
    sample_preds = y_pred[idx]
    if metric.binary_only and len(np.unique(sample_true)) < 2:
      continue
    metrics.append(metric(sample_true, sample_preds, **kwargs))

  metric_mean = np.mean(metrics, axis=0)
  metric_std = np.std(metrics, axis=0)

  metric_lo, metric_hi = np.percentile(metrics, [lo_perc, hi_perc], axis=0)

  return BootstrapResults(metric_mean, metric_std, conf_interval, metric_lo,
                          metric_hi)


class PerformanceMetrics(object):
  """A named collection of invocable, bootstrapable `Metric`s.

  Initializes a class that applies the given `Metric` functions to new ground
  truth labels and predictions. `Metric`s can be evaluated with and without
  bootstrapping.

  The default metrics are number of samples, auc, auprc, and frequency
  calculations for the top 100/10/5/1 top percentiles, if `default_metrics` is
  'binary'. If `default_metrics` is 'continuous', the default metrics are
  Pearson and Spearman correlations, mean squared error (MSE) and
  mean absolute error (MAE).


  Raises:
    ValueError: if an item in `metrics` is not of type `Metric`.
  """

  def __init__(self,
               name: Text,
               default_metrics: Text = None,
               metrics: List[Metric] = None) -> None:

    if metrics is None:
      if default_metrics is None:
        raise ValueError('`default_metrics` is None and no metric is provided.')
      elif default_metrics == 'binary':
        metrics = _build_default_metrics(binary=True)
      elif default_metrics == 'continuous':
        metrics = _build_default_metrics(binary=False)
      else:
        raise ValueError(
            'unknown `default_metrics`: {}'.format(default_metrics))

    for metric in metrics:
      if not isinstance(metric, Metric):
        raise ValueError('Invalid metric value: must be of class `Metric`.')
    self.name = name
    self.metrics = metrics

  def compute(self,
              y_true: np.ndarray,
              y_pred: np.ndarray,
              mask: np.ndarray = None,
              n_bootstrap: int = 0,
              conf_interval: float = 95,
              seed: int = 42,
              **kwargs) -> Dict[Text, BootstrapResults]:
    """Evaluates all metrics using the given labels and predictions.

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.
      mask: A boolean mask; applied to `y_true` and `y_pred`.
      n_bootstrap: An integer denoting the number of bootstrap iterations for
        each evaluation metric.
      conf_interval: A float denoting the width of confidence interval.
      seed: An int denoting the seed for the PRNG.
      **kwargs: Additional keyword arguments passed to each Metric's `func`.

    Returns:
      A dictionary of bootstrapped metrics keyed on metric name with
      `BootstrapResults` values.

    Raises:
      ValueError: If the dimensions of `y_true`, `y_pred`, or `mask` do not
      match, labels are not in {0 , 1}, or predictions are not in [0, 1].
    """
    if len(y_true) != len(y_pred):
      raise ValueError('Label and prediction dimensions do not match.')

    if mask is not None and len(mask) != len(y_pred):
      raise ValueError('Label and prediction dimensions do not match mask.')

    if mask is not None:
      y_true = y_true[mask]
      y_pred = y_pred[mask]

    results = {}
    for metric in self.metrics:
      results[str(metric)] = _bootstrap(
          metric,
          y_true,
          y_pred,
          n_bootstrap=n_bootstrap,
          conf_interval=conf_interval,
          seed=seed,
          **kwargs)
    return results

  def compute_and_print(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        mask: np.ndarray = None,
                        n_bootstrap: int = 0,
                        conf_interval: float = 95,
                        seed: int = 42,
                        title: Text = '',
                        **kwargs) -> None:
    """Evaluates and pretty-prints metrics using given labels and predictions.

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.
      mask: A boolean mask; applied to `y_true` and `y_pred`.
      n_bootstrap: An integer denoting the number of bootstrap iterations for
        each evaluation metric.
      conf_interval: A float denoting the width of confidence interval.
      seed: An int denoting the seed for the PRNG.
      title: A title appended to the printed evaluation metrics.
      **kwargs: Additional keyword arguments passed to each Metric's `func`.

    Raises:
      ValueError: If any of `y_true`, `y_pred`, or `mask` are not of type
          numpy.array of if their dimensions do not match.
    """
    results = self.compute(
        y_true,
        y_pred,
        mask=mask,
        n_bootstrap=n_bootstrap,
        conf_interval=conf_interval,
        seed=seed,
        **kwargs)

    print('{}: {}'.format(self.name, title))
    for metric_name, result in sorted(results.items()):
      print(_build_metric_str(metric_name, result))


class ClassificationPerformanceMetrics(PerformanceMetrics):
  """A classification-specific PerformanceMetrics extension.

  Initializes an object that applies a default set of performance metrics for
  ground truth labels and predictions in a classification problem. This class
  adds classification-specific metrics, specifically precision, recall, and f1
  score, to the default set in PerformanceMetrics.
  """

  def __init__(self, name: Text) -> None:
    metrics = _build_default_metrics(binary=True)
    metrics.extend([
        Metric('precision', sklearn.metrics.precision_score, binary_only=True),
        Metric('recall', sklearn.metrics.recall_score, binary_only=True),
        Metric('f1', sklearn.metrics.f1_score, binary_only=True),
    ])
    super(ClassificationPerformanceMetrics, self).__init__(
        name, metrics=metrics)
