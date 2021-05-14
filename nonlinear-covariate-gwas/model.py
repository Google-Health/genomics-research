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
"""Library for defining DeepNull models and parameters."""
import abc
from typing import Any, Iterable, Sequence
import dataclasses
import tensorflow as tf

from deepnull import metrics


# Orchestrating the training.
@dataclasses.dataclass(frozen=True)
class ModelParameters:
  """Container class for all model parameters."""

  # The MLP units for the nonlinear path of the DeepNull model.
  mlp_units: Sequence[int] = (64, 64, 32, 16)

  # The activation function to use. See https://keras.io/api/layers/activations.
  mlp_activation: str = 'relu'

  # Learning rate for a batch size of 1024. The actual learning rate used is
  # scaled linearly as `learning_rate * batch_size / 1024`.
  learning_rate_batch_1024: float = 1e-4

  # Betas for the Adam optimizer.
  beta_1: float = 0.9
  beta_2: float = 0.99

  # Number of full passes through the training data to perform.
  num_epochs: int = 1000

  # Number of training examples per batch.
  batch_size: int = 1024

  @property
  def learning_rate(self):
    return self.learning_rate_batch_1024 * self.batch_size / 1024.


class _DeepNull(tf.keras.models.Model, abc.ABC):
  """ABC for DeepNull model with MLP layers and direct linear connection.

  Attributes:
    self.mlp: The multi-layer perceptron path through the model.
    self.linear: The linear path through the model.
  """

  def __init__(self, feature_columns: Iterable[Any], mlp_units: Sequence[int],
               mlp_activation: str,
               optimization_metric: metrics.OptimizationMetric, **kwargs):
    """Initialize.

    Args:
      feature_columns: The feature columns.
      mlp_units: A list of the number of units to use in each MLP layer.
      mlp_activation: Activation function for all MLP layers.
        See https://keras.io/api/layers/activations/ for all options.
      optimization_metric: The metric to use as loss function and which is also
        used to select the best performing checkpoint.
      **kwargs: Other arguments for tf.keras.models.Model.
    """
    super().__init__(**kwargs)
    dense_feature_layer = [tf.keras.layers.DenseFeatures(feature_columns)]
    # Non-linear path (long path) in DeepNull.
    mlp_layers = dense_feature_layer + [
        tf.keras.layers.Dense(
            unit, activation=mlp_activation, name=f'layer{i}')
        for i, unit in enumerate(mlp_units)
    ] + [tf.keras.layers.Dense(1, activation=None, name='linear_mlp')]

    # Linear path (short, ResNet-esque path) in DeepNull.
    linear_layers = dense_feature_layer + [
        tf.keras.layers.Dense(1, activation=None, name='linear')
    ]
    self.mlp = tf.keras.Sequential(mlp_layers)
    self.linear = tf.keras.Sequential(linear_layers)
    self.optimization_metric = optimization_metric

  def model(self, inputs) -> tf.keras.models.Model:
    """Defined to support use of model.summary()."""
    return tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))

  def call(self, inputs):
    """See https://keras.io/api/models/model/#model-class for details."""
    return self.final_activation(self.mlp(inputs) + self.linear(inputs))

  def best_checkpoint_metric(self):
    return self.optimization_metric.best_checkpoint_metric

  def best_checkpoint_mode(self):
    return self.optimization_metric.best_checkpoint_mode

  @abc.abstractproperty
  def final_activation(self):
    """Returns the activation to apply to the final model output."""

  @abc.abstractmethod
  def metrics_to_use(self):
    """Returns the list of metrics to use when training the model."""

  @abc.abstractmethod
  def loss_function(self):
    """The loss function to use to train the model."""


class QuantitativeDeepNull(_DeepNull):
  """Concrete subclass to train quantitative phenotypes."""

  @property
  def final_activation(self):
    # Identity function: There should be no activation applied for a
    # quantitative phenotype.
    return tf.keras.activations.linear

  def metrics_to_use(self):
    return ['mse', metrics.tf_pearson]

  def loss_function(self):
    return tf.keras.losses.MeanSquaredError(name='mse')


class BinaryDeepNull(_DeepNull):
  """Concrete subclass to train binary phenotypes."""

  @property
  def final_activation(self):
    return tf.keras.activations.sigmoid

  def metrics_to_use(self):
    return [
        'crossentropy', 'accuracy',
        tf.keras.metrics.AUC(curve='ROC', name='auroc'),
        tf.keras.metrics.AUC(curve='PR', name='auprc')
    ]

  def loss_function(self):
    return tf.keras.losses.BinaryCrossentropy(name='crossentropy')
