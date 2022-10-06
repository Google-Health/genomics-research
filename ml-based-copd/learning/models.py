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
"""Implements spirometry `tf.keras.Model`s."""
from typing import Callable, List, MutableMapping, Optional, Type, TypeVar

import ml_collections
import tensorflow as tf
from tensorflow_addons import layers as tfa_layers

import head_utils

# Denotes a function that converts model and head configs into a tf.keras model.
ModelFunction = Callable[[ml_collections.ConfigDict, ml_collections.ConfigDict],
                         tf.keras.Model]

# A registry dictionary mapping model types to ModelFunctions.
_MODEL_REGISTRY: MutableMapping[str, ModelFunction] = {}

# Denotes available types when parsing comma-separated hyperparameter strings.
Parsable = TypeVar('Parsable', str, int, float)


def _parse_comma_string(comma_string: str,
                        element_class: Type[Parsable]) -> List[Parsable]:
  """Converts a comma-separated string to a list of `element_class` elements."""
  if not comma_string:
    return []
  return [element_class(element) for element in comma_string.split(',')]


def register_model(model_type: str) -> Callable[[ModelFunction], ModelFunction]:
  """Registers a model function in `_MODEL_REGISTRY` as `model_type`.

  Args:
    model_type: The model type to register.

  Returns:
    A function that registers a model function as the given `model_type`.
  """

  def _register_model(model_fn: ModelFunction) -> ModelFunction:
    if model_type in _MODEL_REGISTRY:
      raise ValueError(f'"{model_type}" already registered: {_MODEL_REGISTRY}')
    _MODEL_REGISTRY[model_type] = model_fn
    return model_fn

  return _register_model


@register_model('MLP')
def _mlp(backbone_config: ml_collections.ConfigDict,
         head_configs: ml_collections.ConfigDict) -> tf.keras.Model:
  """Returns a MLP model parameterized by backbone and head configs.

  The `backbone_config` `ConfigDict` must be a valid `network_config`. The
  following required attributes are exposed through `backbone_config.kwargs`:
    layer_dims: A comma-separated list of integers denoting dense layer units.
    input_names: A comma-separated list of strings denoting input names.
    input_shape: A tuple denoting each input's shape.
    conditional_input_names: A comma-separated list of strings denoting
      conditional input names. These input tensors are concatenated with the
      backbone's output feature embedding.
    conditional_input_shape: A tuple denoting each conditional input's shape.
    model_name: The name assigned to the tf.keras.Model that will be created.

  A note on `input_shape` and `conditional_input_shape`: The model currently
  assumes that all input tensors corresponding to `input_names` are of shape
  `input_shape`. Similarly, the model assumes that all input tensors
  corresponding to `conditional_input_names` are of shape
  `conditional_input_shape`. This allows the model to make assumptions about
  how data can be combined. For example, a model with `input_names` encoding
  spirometry blow curves can safely concatenate and combine these
  representations knowing they will all have matching shapes. The same applies
  to conditional inputs that are concatenated with the backbone's output.

  The following optional `backbone_config.kwargs` attributes are exposed:
    dense_kernel_l2: The kernel l2 applied to dense layers; default is 0.0.
    dense_dropout: The dropout rate applied after dense layers; default is 0.0.
    dense_activation: The activation function in dense layers; default is relu.

  Args:
    backbone_config: A valid `network_config` used to generate the backbone.
    head_configs: A ConfigDict mapping `head_name` to valid head ConfigDicts.

  Returns:
    A tf.keras.Model.
  """
  kwargs = backbone_config.kwargs

  # Parse config hyperparameters.
  l2 = kwargs.get('dense_kernel_l2', 0)
  dropout_rate = kwargs.get('dense_dropout', 0)
  dense_activation = kwargs.get('dense_activation', 'relu')
  layer_dims = _parse_comma_string(kwargs.layer_dims, int)

  # Generate standard input layers and conditional input layers from config.
  inputs = [
      tf.keras.Input(shape=kwargs.input_shape, name=name)
      for name in _parse_comma_string(kwargs.input_names, str)
  ]
  conditional_inputs = [
      tf.keras.Input(shape=kwargs.conditional_input_shape, name=name)
      for name in _parse_comma_string(kwargs.conditional_input_names, str)
  ]

  # Build shared layers and regularizers.
  dropout = tf.keras.layers.Dropout(dropout_rate)
  regularizer = tf.keras.regularizers.l2(l2) if l2 else None

  # Build backbone graph.
  hid = tf.keras.layers.Concatenate(axis=1)(inputs)
  for dim in layer_dims:
    dense_layer = tf.keras.layers.Dense(
        dim, activation=dense_activation, kernel_regularizer=regularizer)
    hid = dense_layer(hid)
    if dropout_rate:
      hid = dropout(hid)

  # If conditional inputs are specified, concatenate them with the backbone's
  # intermediate feature embedding.
  if conditional_inputs:
    hid = tf.keras.layers.Concatenate(axis=1)([hid] + conditional_inputs)

  # Build output graph for each head. We sort the heads by key to ensure that
  # all distributed workers initialize graph tensors in the same order.
  outputs = [
      head_utils.get_head(head_config, hid)(hid)
      for _, head_config in sorted(head_configs.items())
  ]

  return tf.keras.Model(
      inputs=inputs + conditional_inputs,
      outputs=outputs,
      name=kwargs.model_name)


def _resnet18_1d_v0_convblock(
    hid: tf.Tensor,
    filters: int,
    kernel_size: int,
    stride: int,
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
) -> tf.Tensor:
  """Passes the input tensor `hid` through a one-dimensional ResNet conv block.

  This is a standard ResNet conv block as described in [1].

  [1] https://arxiv.org/abs/1512.03385

  Args:
    hid: The input tensor passed to the block's subgraph.
    filters: The number of convolutional output filters used in the block.
    kernel_size: The length of the 1D convolution window.
    stride: The stride length of the convolution.
    kernel_regularizer: A regularizer applied to the conv layer kernel.

  Returns:
    The output tensor of a one-dimensional ResNet conv block.
  """
  hid = tf.keras.layers.Conv1D(
      filters=filters,
      kernel_size=kernel_size,
      strides=stride,
      padding='same',
      use_bias=False,
      kernel_regularizer=kernel_regularizer)(
          hid)
  hid = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(hid)
  hid = tf.keras.layers.ReLU()(hid)
  return hid


def _resnet18_1d_v0_resblock(
    hid: tf.Tensor,
    filters: int,
    kernel_size: int,
    first_stride: int = 1,
    downsample: bool = False,
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
) -> tf.Tensor:
  """Passes the input tensor `hid` through a one-dimensional ResNet-D block.

  The ResNet-D block from [2] changes the original ResNet [1] by modifying the
  downsampling block as described in Section 4.2 of [2].

  [1] https://arxiv.org/abs/1512.03385
  [2] https://arxiv.org/abs/1812.01187

  Args:
    hid: The input tensor passed to the block's subgraph.
    filters: The number of convolutional output filters used in the block.
    kernel_size: The length of the 1D convolution window.
    first_stride: The conv stride used in the residual block's first conv block.
    downsample: Whether the residual block downsamples the input.
    kernel_regularizer: A regularizer applied to all conv layer kernels.

  Returns:
    The output tensor of a one-dimensional ResNet-D block.
  """
  identity = hid
  hid = _resnet18_1d_v0_convblock(
      hid,
      filters=filters,
      kernel_size=kernel_size,
      stride=first_stride,
      kernel_regularizer=kernel_regularizer,
  )
  hid = tf.keras.layers.Conv1D(
      filters=filters,
      kernel_size=kernel_size,
      strides=1,
      padding='same',
      use_bias=False,
      kernel_regularizer=kernel_regularizer,
  )(
      hid)
  hid = tf.keras.layers.BatchNormalization(
      epsilon=1e-05, momentum=0.1, gamma_initializer='zeros')(
          hid)
  if downsample:
    identity = tf.keras.layers.AvgPool1D(
        2, strides=2, padding='valid')(
            identity)
    identity = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_regularizer=kernel_regularizer)(
            identity)
    identity = tf.keras.layers.BatchNormalization(
        epsilon=1e-05, momentum=0.1)(
            identity)
  hid += identity
  hid = tf.keras.layers.ReLU()(hid)
  return hid


@register_model('ResNet18')
def _resnet18_1d_v0(backbone_config: ml_collections.ConfigDict,
                    head_configs: ml_collections.ConfigDict) -> tf.keras.Model:
  """Returns a ResNet18-D variant parameterized by backbone and head configs.

  We modified a base ResNet [1] model using the ResNet-D stem and block tweaks
  proposed in [2]. We convert 2D operations, such as conv and pooling layers,
  to their 1D equivalents to support the spirometry modality.

  [1] https://arxiv.org/abs/1512.03385
  [2] https://arxiv.org/abs/1812.01187

  The `backbone_config` `ConfigDict` must be a valid `network_config`. The
  following required attributes are exposed through `backbone_config.kwargs`:
    input_names: A comma-separated list of strings denoting input names.
    input_shape: A tuple denoting each input's shape.
    conditional_input_names: A comma-separated list of strings denoting
      conditional input names. These input tensors are concatenated with the
      backbone's output feature embedding.
    conditional_input_shape: A tuple denoting each conditional input's shape.
    model_name: The name assigned to the tf.keras.Model that will be created.

  A note on `input_shape` and `conditional_input_shape`: The model currently
  assumes that all input tensors corresponding to `input_names` are of shape
  `input_shape`. Similarly, the model assumes that all input tensors
  corresponding to `conditional_input_names` are of shape
  `conditional_input_shape`. This allows the model to make assumptions about
  how data can be combined. For example, a model with `input_names` encoding
  spirometry blow curves can safely concatenate and combine these
  representations knowing they will all have matching shapes. The same applies
  to conditional inputs that are concatenated with the backbone's output.

  The following optional `backbone_config.kwargs` attributes are exposed:
    kernel_size: The The length of the 1D convolution window.
    kernel_l2: The regularizer used by backbone conv/dense layers; default is 0.

  Args:
    backbone_config: A valid `network_config` used to generate the backbone.
    head_configs: A ConfigDict mapping `head_name` to valid head ConfigDicts.

  Returns:
    A one-dimensional ResNet18-D tf.keras.Model.
  """
  kwargs = backbone_config.kwargs

  # Parse config hyperparameters.
  kernel_size = kwargs.get('kernel_size', 3)
  l2 = kwargs.get('kernel_l2', 0)
  regularizer = tf.keras.regularizers.l2(l2) if l2 else None

  # Generate standard input layers and conditional input layers from config.
  inputs = [
      tf.keras.Input(shape=kwargs.input_shape, name=name)
      for name in _parse_comma_string(kwargs.input_names, str)
  ]
  conditional_inputs = [
      tf.keras.Input(shape=kwargs.conditional_input_shape, name=name)
      for name in _parse_comma_string(kwargs.conditional_input_names, str)
  ]

  hid = tf.stack(inputs, axis=-1)
  hid = tf.keras.layers.ZeroPadding1D(12)(hid)

  # Input stem. We replace the standard ResNet 7x1 conv with three consecutive
  # 3x1 convs as described in Section "4.2: ResNet-D" in [2].
  hid = _resnet18_1d_v0_convblock(
      hid,
      filters=32,
      kernel_size=kernel_size,
      stride=2,
      kernel_regularizer=regularizer)
  hid = _resnet18_1d_v0_convblock(
      hid,
      filters=32,
      kernel_size=kernel_size,
      stride=1,
      kernel_regularizer=regularizer)
  hid = _resnet18_1d_v0_convblock(
      hid,
      filters=64,
      kernel_size=kernel_size,
      stride=1,
      kernel_regularizer=regularizer)
  hid = tf.keras.layers.MaxPool1D(3, strides=2, padding='same')(hid)

  # Stage 1.
  hid = _resnet18_1d_v0_resblock(
      hid, filters=64, kernel_size=kernel_size, kernel_regularizer=regularizer)
  hid = _resnet18_1d_v0_resblock(
      hid, filters=64, kernel_size=kernel_size, kernel_regularizer=regularizer)

  # Stage 2.
  hid = _resnet18_1d_v0_resblock(
      hid,
      filters=128,
      kernel_size=kernel_size,
      first_stride=2,
      downsample=True,
      kernel_regularizer=regularizer)
  hid = _resnet18_1d_v0_resblock(
      hid, filters=128, kernel_size=kernel_size, kernel_regularizer=regularizer)

  # Stage 3.
  hid = _resnet18_1d_v0_resblock(
      hid,
      filters=256,
      kernel_size=kernel_size,
      first_stride=2,
      downsample=True,
      kernel_regularizer=regularizer)
  hid = _resnet18_1d_v0_resblock(
      hid, filters=256, kernel_size=kernel_size, kernel_regularizer=regularizer)

  # Stage 4.
  hid = _resnet18_1d_v0_resblock(
      hid,
      filters=512,
      kernel_size=kernel_size,
      first_stride=2,
      downsample=True,
      kernel_regularizer=regularizer)
  hid = _resnet18_1d_v0_resblock(
      hid, filters=512, kernel_size=kernel_size, kernel_regularizer=regularizer)

  # Output embedding.
  hid = tfa_layers.AdaptiveAveragePooling1D([1], data_format='channels_last')(
      hid)
  hid = tf.keras.layers.Flatten()(hid)
  hid = tf.keras.layers.Dense(
      128, activation='relu', use_bias=False, kernel_regularizer=regularizer)(
          hid)

  # If conditional inputs are specified, concatenate them with the backbone's
  # intermediate feature embedding.
  if conditional_inputs:
    hid = tf.keras.layers.Concatenate(axis=1)([hid] + conditional_inputs)

  # Build output graph for each head. We sort the heads by key to ensure that
  # all distributed workers initialize graph tensors in the same order.
  outputs = [
      head_utils.get_head(head_config, hid)(hid)
      for _, head_config in sorted(head_configs.items())
  ]

  return tf.keras.Model(
      inputs=inputs + conditional_inputs,
      outputs=outputs,
      name=kwargs.model_name)


def get_model(backbone_config: ml_collections.ConfigDict,
              head_configs: ml_collections.ConfigDict) -> tf.keras.Model:
  """Converts backbone and head configs to a tf.keras.Model.

  Args:
    backbone_config: A valid `network_config` used to generate the backbone.
    head_configs: A ConfigDict mapping `head_name` to valid head ConfigDicts.

  Returns:
    A tf.keras.Model.
  """
  class_name = backbone_config.class_name
  if class_name not in _MODEL_REGISTRY:
    raise ValueError(f'Invalid model network `class_name`: {class_name}')

  return _MODEL_REGISTRY[class_name](backbone_config, head_configs)
