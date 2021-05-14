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
"""Library of metrics and analysis of metrics."""
from typing import Any, Callable, Dict, List, Union
import dataclasses
import tensorflow as tf
import tensorflow_probability as tfp


@dataclasses.dataclass(frozen=True)
class OptimizationMetric:
  """Container class for a metric to use as the loss for model optimization."""
  name: str
  func: Union[tf.keras.losses.Loss, Callable[..., Any]]
  # How to select the best performing checkpoint for the loss ('max', 'min').
  best_checkpoint_mode: str

  @property
  def best_checkpoint_metric(self) -> str:
    return f'val_{self.name}'


def tf_pearson(y_true: tf.Tensor, y_pred: tf.Tensor):
  """Returns an implementation of Pearson correlation."""
  return tfp.stats.correlation(y_pred, y_true)


def acceptable_model_performance(eval_metrics: List[Dict[str, float]]) -> bool:
  """Returns True if and only if performance across folds is acceptable."""
  if 'tf_pearson' in eval_metrics[0]:
    key = 'tf_pearson'
    name = 'correlation'
  else:
    key = 'auroc'
    name = 'AUROC'

  values = sorted([d[key] for d in eval_metrics])
  if values[0] <= 0:
    print(f'Some {name}s are non-positive: ', values)
    return False

  if values[-1] < 0.05:
    print(f'Weak {name}s of covariates to phenotype: ', values)
    return False

  delta = (values[-1] - values[0]) / values[0]
  if delta > 0.1:
    print('Performance gap between folds > 10%: ', values)
    return False

  return True


def get_optimization_metric(name: str) -> OptimizationMetric:
  """Returns the requested OptimizationMetric."""
  try:
    return _OPTIMIZATION_METRIC_REGISTRY[name]
  except KeyError:
    raise ValueError(f'Unsupported optimization metric "{name}", must be in: '
                     f'{sorted(_OPTIMIZATION_METRIC_REGISTRY.keys())}')


_OPTIMIZATION_METRIC_REGISTRY = {
    'mse':
        OptimizationMetric(
            name='mse',
            func=tf.keras.losses.MeanSquaredError(name='mse'),
            best_checkpoint_mode='min'),
    'crossentropy':
        OptimizationMetric(
            name='crossentropy',
            func=tf.keras.losses.BinaryCrossentropy(name='crossentropy'),
            best_checkpoint_mode='min'),
    'tf_pearson':
        OptimizationMetric(
            name='tf_pearson', func=tf_pearson, best_checkpoint_mode='max'),
}
