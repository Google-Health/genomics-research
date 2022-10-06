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
"""Utilities for loading `tf.keras.Models` implementations."""
from typing import Any, Callable, Dict, Union

import ml_collections
import tensorflow as tf

KerasIdentifier = Dict[str, Union[str, Dict[str, Any]]]


def get_network(
    network_config: ml_collections.ConfigDict,
    get_network_fn: Callable[[KerasIdentifier],
                             tf.keras.Model]) -> tf.keras.Model:
  """Returns a model instance loaded using the provided `get_network_fn`.

  The `network_config` `ConfigDict` must match the config schema defined in the
  README. This config is converted into a Keras identifier and then passed to
  the provided `get_network_fn` function. Since an individual training pipeline
  may implement custom models, this getter is provided by the caller.

  For example:

  ```
  valid_network_config = ml_collections.ConfigDict({
      'class_name': 'ResNet50',
      'kwargs': {
          'include_top': True,
          'weights': 'imagenet',
      },
  })
  network = get_network_fn(valid_network_config)
  ```

  Args:
    network_config: A ConfigDict matching the `network_config` schema.
    get_network_fn: A getter function that loads a `tf.keras.Model` given a
      valid Keras identifier.

  Returns:
    A network instance.
  """
  # Build an identifier dict that can be parsed by `get_network_fn`.
  identifier_kwargs = network_config.get('kwargs', ml_collections.ConfigDict())
  identifier = {
      'class_name': network_config['class_name'],
      'config': identifier_kwargs.to_dict()
  }
  return get_network_fn(identifier)
