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
"""Utilites for loading subnetworks, losses, and metrics for model heads."""
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import ml_collections
import tensorflow as tf

import losses
import metrics
import model_utils

# Denotes a head function that converts a Keras identifier into a model.
KerasIdentifier = Dict[str, Union[str, Dict[str, Any]]]
HeadFunction = Callable[[KerasIdentifier, tf.Tensor], tf.keras.Model]

# Available head function registry keys.
DENSE_CLASSIFICATION = 'DenseClassification'
DENSE_REGRESSION = 'DenseRegression'

# A registry dictionary mapping head types to HeadFunctions.
_HEAD_REGISTRY: Dict[str, HeadFunction] = {}

# Keras identifier config properties used by model heads when building models.
_CONFIG_HEAD_NAME = 'head_name'
_CONFIG_INPUT_SHAPE = 'input_shape'
_CONFIG_NUM_CLASSES = 'num_classes'
_CONFIG_OUTPUT_BIAS_INITIALIZER = 'output_bias_initializer'
_CONFIG_ADDITIONAL_DIMS = 'additional_dims'
_CONFIG_ADDITIONAL_ACTIVATION = 'additional_activation'
_CONFIG_ADDITIONAL_KERNEL_L2 = 'additional_kernel_l2'


def register_head(head_type: str) -> Callable[[HeadFunction], HeadFunction]:
  """Registers a head function` in `_HEAD_REGISTRY` as `head_type`.

  Args:
    head_type: The head type to register.

  Returns:
    A function that registers a head function as the given `head_type`.
  """

  def _register_head(head_fn: HeadFunction) -> HeadFunction:
    if head_type in _HEAD_REGISTRY:
      raise ValueError(f'"{head_type}" already registered: {_HEAD_REGISTRY}')
    _HEAD_REGISTRY[head_type] = head_fn
    return head_fn

  return _register_head


def _parse_comma_string(comma_string: str) -> List[int]:
  """Converts a comma-separated string of integers to a list."""
  if not comma_string:
    return []
  return [int(element) for element in comma_string.split(',')]


def _dense_head_network(
    head_name: str,
    head_input: tf.Tensor,
    output_dim: int,
    output_activation: Optional[str],
    output_bias_initializer: Optional[Union[List[float], str]] = None,
    additional_dims: Optional[str] = None,
    additional_activation: Optional[str] = None,
    additional_l2: Optional[float] = None,
) -> tf.keras.Model:
  """Returns an output head model containing intermediate dense layers.

  Warning: If no `additional_dims` are provided, the network will contain a
  single `tf.keras.layers.Dense` layer parameterized by `output_*` arguments.
  Since this is a single layer, there will be no non-linearity in the output
  head subnetwork.

  Args:
    head_name: The name of the output head; corresponds to label and metric IDs.
    head_input: The head model's input tensor; this tensor is often the
      backbone's features embedding.
    output_dim: The dimension of the output tensor.
    output_activation: The activation function used for the final output layer.
    output_bias_initializer: The bias initializer in the final output layer.
      This can be a string corresponding to a
      `tf.keras.initializers.Initializer` class, e.g., 'zeros', or a float
      value.
    additional_dims: A list containing comma-separated integers representing
      dimensions of additional, intermediate Dense layers.
    additional_activation: The activation used in all additional Dense layers.
    additional_l2: The L2 penalty used in all additional Dense layers.

  Returns:
    A tf.keras.Model representing the output head network.
  """
  l2 = tf.keras.regularizers.L2(additional_l2) if additional_l2 else None
  if output_bias_initializer and isinstance(output_bias_initializer, list):
    output_bias_initializer = tf.keras.initializers.Constant(
        output_bias_initializer)

  hid = head_input
  for dim in _parse_comma_string(additional_dims):
    dense_layer = tf.keras.layers.Dense(
        dim,
        activation=additional_activation,
        kernel_regularizer=l2,
    )
    hid = dense_layer(hid)

  output_dense_layer = tf.keras.layers.Dense(
      output_dim,
      activation=output_activation,
      bias_initializer=output_bias_initializer,
  )
  output_tensor = output_dense_layer(hid)

  return tf.keras.Model(
      inputs=[head_input],
      outputs=[output_tensor],
      name=head_name,
  )


@register_head(DENSE_CLASSIFICATION)
def dense_classification_head(identifier: KerasIdentifier,
                              head_input: tf.Tensor) -> tf.keras.Model:
  """Converts a Keras identifier into a classification head model.

  These KerasIdentifier.KERAS_CONFIG config attributes are required:
    head_name: The name of the output head; corresponds to label and metric IDs.
    num_classes: The number of output classes; must be >= 2.

  These KerasIdentifier.KERAS_CONFIG config attributes are optional:
    output_bias_initializer: The bias initializer in the final output layer.
    This can be a
      string corresponding to a `tf.keras.initializers.Initializer` class, e.g.,
      'zeros', or a list of float values. The default is 'zeros'.
    additional_dims: A list containing comma-separated integers representing
      dimensions of additional, intermediate Dense layers. The default is
      `None`, i.e., no non-linearity in the head network.
    additional_activation: The activation used in all additional Dense layers.
      The default is `None`, i.e., a linear activation `a(x) = x`.
    additional_l2: The L2 penalty used in all additional Dense layers. The
      default is `None`, i.e., no L2 regularization.

  Note: Binary output heads will have output shape of `1` while multinomial
  output heads will have shape `identifier.config['num_classes']`.

  Args:
    identifier: An identifier parameterized by a 'config'.
    head_input: A tensor passed to the `head_config` model as input.

  Returns:
    A tf.keras.Model with input tensor `head_input` and output of shape
    `identifier.config['num_classes']`.
  """
  config = dict(identifier['config'])
  num_classes = config[_CONFIG_NUM_CLASSES]

  # Determine output shape and activation for binary or multinomial outputs.
  if num_classes < 2:
    raise ValueError(
        f'Invalid `num_classes` given for classification head: {num_classes}')
  output_dim = 1 if num_classes == 2 else num_classes
  output_activation = 'sigmoid' if output_dim == 1 else 'softmax'

  return _dense_head_network(
      config[_CONFIG_HEAD_NAME],
      head_input,
      output_dim,
      output_activation,
      config.get(_CONFIG_OUTPUT_BIAS_INITIALIZER, 'zeros'),
      config.get(_CONFIG_ADDITIONAL_DIMS, None),
      config.get(_CONFIG_ADDITIONAL_ACTIVATION, None),
      config.get(_CONFIG_ADDITIONAL_KERNEL_L2, None),
  )


@register_head(DENSE_REGRESSION)
def dense_regression_head(identifier: KerasIdentifier,
                          head_input: tf.Tensor) -> tf.keras.Model:
  """Converts a Keras identifier into a regression head model.

  These KerasIdentifier.KERAS_CONFIG config attributes are required:
    head_name: The name of the output head; corresponds to label and metric IDs.

  These KerasIdentifier.KERAS_CONFIG config attributes are optional:
    output_bias_initializer: The bias initializer in the final output layer.
    This can be a
      string corresponding to a `tf.keras.initializers.Initializer` class, e.g.,
      'zeros', or a list of float values. The default is 'zeros'.
    additional_dims: A list containing comma-separated integers representing
      dimensions of additional, intermediate Dense layers. The default is
      `None`, i.e., no non-linearity in the head network.
    additional_activation: The activation used in all additional Dense layers.
      The default is `None`, i.e., a linear activation `a(x) = x`.
    additional_l2: The L2 penalty used in all additional Dense layers. The
      default is `None`, i.e., no L2 regularization.

  Args:
    identifier: An identifier parameterized by a 'config'.
    head_input: A tensor passed to the `head_config` model as input.

  Returns:
    A tf.keras.Model with input tensor `head_input` and output of shape `1`.
  """
  config = dict(identifier['config'])
  return _dense_head_network(
      config[_CONFIG_HEAD_NAME],
      head_input,
      1,
      None,
      config.get(_CONFIG_OUTPUT_BIAS_INITIALIZER, 'zeros'),
      config.get(_CONFIG_ADDITIONAL_DIMS, None),
      config.get(_CONFIG_ADDITIONAL_ACTIVATION, None),
      config.get(_CONFIG_ADDITIONAL_KERNEL_L2, None),
  )


def _get_head_network(identifier: KerasIdentifier,
                      head_input: tf.Tensor) -> tf.keras.Model:
  """Converts a Keras identifier into a head model.

  Args:
    identifier: An identifier containing the `class_name`'s expected `kwargs`.
    head_input: A tensor passed to the `head_config` model as input.

  Returns:
    A tf.keras.Model generated by the specified provider.
  """
  # Ensure `class_name` is a valid custom head.
  class_name = identifier['class_name']
  if class_name not in _HEAD_REGISTRY:
    raise ValueError(f'Invalid head network `class_name`: {class_name}')

  return _HEAD_REGISTRY[class_name](identifier, head_input)


def get_head(head_config: ml_collections.ConfigDict,
             head_input: tf.Tensor) -> tf.keras.Model:
  """Converts a `ConfigDict` into a head submodel.

  These submodels can be used to convert the output of a shared backbone into
  the expected representation for the given head.

  Args:
    head_config: A ConfigDict matching the `head_config` schema.
    head_input: A tensor passed to the `head_config` model as input.

  Returns:
    A tf.keras.Model generated by the specified provider.
  """

  def _get_head_network_wrapper(identifier: KerasIdentifier) -> tf.keras.Model:
    return _get_head_network(identifier, head_input)

  return model_utils.get_network(head_config['network_config'],
                                 _get_head_network_wrapper)


def get_losses(
    head_configs: ml_collections.ConfigDict
) -> Tuple[Dict[str, tf.losses.Loss], Dict[str, float]]:
  """Returns dictionaries of loss instances and loss weights keyed on head name.

  These dictionaries can be passed directly to `tf.keras.Model`'s' `compile()`.

  Args:
    head_configs: A ConfigDict containing head configs keyed on head name.

  Returns:
    A pair of dictionaries mapping head name to loss instances and loss weights.
  """
  head_losses = {}
  head_loss_weights = {}

  for _, head_config in head_configs.items():
    head_name = head_config['name']
    head_losses[head_name] = losses.get_loss(head_config['loss_config'])
    head_loss_weights[head_name] = head_config['loss_weight']

  return head_losses, head_loss_weights


def get_metrics(
    head_configs: ml_collections.ConfigDict
) -> Dict[str, List[tf.metrics.Metric]]:
  """Returns a dictionary of metric lists keyed on head name.

  This dictionary can be passed directly to `tf.keras.Model`'s' `compile()`.

  Args:
    head_configs: A ConfigDict containing head configs keyed on head name.

  Returns:
    A dictionary of metric lists keyed on head name.
  """
  head_metrics = {}

  for _, head_config in head_configs.items():
    metric_configs = head_config['metric_configs']
    head_metrics[head_config['name']] = [
        metrics.get_metric(metric_configs[metric_name])
        for metric_name in sorted(metric_configs)
    ]

  return head_metrics
