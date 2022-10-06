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
"""A shared library for building spirometry `ml_collections.ConfigDict`s."""
from typing import List, Optional, Sequence

import ml_collections


def binary_head_config(
    head_name: str,
    loss_weight: float = 1.0,
    additional_dims: Optional[Sequence[int]] = (64, 32, 16),
    additional_activation: str = 'relu',
    additional_kernel_l2: float = 0.0,
    output_bias: Optional[List[float]] = None,
) -> ml_collections.ConfigDict:
  """Builds a `ConfigDict` parameterizing a binary classification head.

  Args:
    head_name: The head subnetwork's name; must match the name used to register
      the head's loss, loss weight, and metrics.
    loss_weight: A scalar value used to weight the head's loss. See the docs for
      details (https://tensorflow.org/api_docs/python/tf/keras/Model#compile).
    additional_dims: An optional list integers denoting the number of units for
      each additional dense layer. If empty, only the final output layer is
      applied (i.e., there will be no nonlinearities between the backbone output
      and the head's output layer).
    additional_activation: The nonlinear activation for all additional layers.
    additional_kernel_l2: The L2 regularizer for all additional layer kernels.
    output_bias: The initial bias for the head's final output layer.

  Returns:
    A `ConfigDict` parameterizing a binary classification head.
  """
  additional_dims = [] if additional_dims is None else additional_dims
  loss_config = {
      'class_name': 'BinaryCrossentropy',
      'kwargs': {
          'from_logits': False,
      },
  }
  metric_configs = {
      'auprc': {
          'class_name': 'AUC',
          'kwargs': {
              'curve': 'PR',
              'name': 'auprc',
          },
      },
      'auroc': {
          'class_name': 'AUC',
          'kwargs': {
              'curve': 'ROC',
              'name': 'auroc',
          },
      },
      'binary_accuracy': {
          'class_name': 'BinaryAccuracy',
      },
      # Note: This is a custom metric implementation from our metrics lib:
      # `//learning/genomics/medgen/tf2:metrics`.
      'pearson_correlation': {
          'class_name': 'PearsonCorrelationOnline',
      },
  }
  network_config = {
      'class_name': 'DenseClassification',
      'kwargs': {
          'head_name': head_name,
          'num_classes': 2,
          'output_bias': output_bias,
          'additional_dims': ','.join([str(dim) for dim in additional_dims]),
          'additional_activation': additional_activation,
          'additional_kernel_l2': additional_kernel_l2,
      }
  }
  head_config = ml_collections.ConfigDict({
      'name': head_name,
      'loss_config': loss_config,
      'loss_weight': loss_weight,
      'metric_configs': metric_configs,
      'network_config': network_config,
  })
  return head_config


def regression_head_config(
    head_name: str,
    loss_weight: float = 1.0,
    additional_dims: Optional[Sequence[int]] = (64, 32, 16),
    additional_activation: str = 'relu',
    additional_kernel_l2: float = 0.0,
    output_bias: Optional[List[float]] = None,
) -> ml_collections.ConfigDict:
  """Builds a `ConfigDict` parameterizing a regression head.

  Args:
    head_name: The head subnetwork's name; must match the name used to register
      the head's loss, loss weight, and metrics.
    loss_weight: A scalar value used to weight the head's loss. See the docs for
      details (https://tensorflow.org/api_docs/python/tf/keras/Model#compile).
    additional_dims: An optional list integers denoting the number of units for
      each additional dense layer. If empty, only the final output layer is
      applied (i.e., there will be no nonlinearities between the backbone output
      and the head's output layer).
    additional_activation: The nonlinear activation for all additional layers.
    additional_kernel_l2: The L2 regularizer for all additional layer kernels.
    output_bias: The initial bias for the head's final output layer.

  Returns:
    A `ConfigDict` parameterizing a regression head.
  """
  additional_dims = [] if additional_dims is None else additional_dims
  loss_config = {
      'class_name': 'MeanSquaredError',
  }
  metric_configs = {
      'rmse': {
          'class_name': 'RootMeanSquaredError',
      },
      'mse': {
          'class_name': 'MeanSquaredError',
      },
      'mae': {
          'class_name': 'MeanAbsoluteError',
      },
      # Note: This is a custom metric implementation from our metrics lib:
      # `//learning/genomics/medgen/tf2:metrics`.
      'pearson_correlation': {
          'class_name': 'PearsonCorrelationOnline',
      }
  }
  network_config = {
      'class_name': 'DenseRegression',
      'kwargs': {
          'head_name': head_name,
          'output_bias': output_bias,
          'additional_dims': ','.join([str(dim) for dim in additional_dims]),
          'additional_activation': additional_activation,
          'additional_kernel_l2': additional_kernel_l2,
      }
  }
  head_config = ml_collections.ConfigDict({
      'name': head_name,
      'loss_config': loss_config,
      'loss_weight': loss_weight,
      'metric_configs': metric_configs,
      'network_config': network_config,
  })
  return head_config
