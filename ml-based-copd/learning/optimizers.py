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
"""Custom `tf.optimizers.Optimizer` implementations and loaders."""
import ml_collections
import tensorflow as tf


def get_optimizer(
    optimizer_config: ml_collections.ConfigDict) -> tf.optimizers.Optimizer:
  """Returns an optimizer for use in model training.

  The `optimizer_config` `ConfigDict` must match the config schema defined in
  the README.

  Note: This function wraps `tf.optimizers.get()`. Any `tf.optimizers` class
  from https://www.tensorflow.org/api_docs/python/tf/keras/optimizers can be
  loaded with this utility. See individual class implementations for the set of
  available keyword arguments that can be configured via
  `optimizer_config.kwargs`.

  For example:

  ```
  valid_optimizer_config = ml_collections.ConfigDict({
      'class_name': 'Adam',
      'kwargs': {
          'learning_rate': 0.001,
          'beta_1': 0.9,
          'beta_2': 0.999,
      },
  })
  optimizer = get_optimizer(valid_optimizer_config)
  ```

  Note: Learning rate schedules are also natively supported by Keras
  identifiers, i.e., `kwargs.learning_rate` can be set either to a float value
  representing a fixed learning rate or to a Keras identifier representing a
  learning rate schedule.

  For example

  ```
  valid_optimizer_config = ml_collections.ConfigDict({
      'class_name': 'Adam',
      'kwargs': {
          'learning_rate': {
              'class_name': 'ExponentialDecay',
              'config': {
                  'initial_learning_rate': 0.1,
                  'decay_steps': 10,
                  'decay_rate': 0.99,
                  'staircase': False,
              }
          },
          'beta_1': 0.9,
          'beta_2': 0.999,
      },
  })
  optimizer = get_optimizer(valid_optimizer_config)
  ```

  Args:
    optimizer_config: A ConfigDict matching the `optimizer_config` schema.

  Returns:
    An optimizer instance.
  """
  # Build an identifier dict that can be parsed by `tf.optimizers.get`.
  identifier_kwargs = optimizer_config.get('kwargs',
                                           ml_collections.ConfigDict())
  identifier = {
      'class_name': optimizer_config['class_name'],
      'config': identifier_kwargs.to_dict()
  }

  return tf.optimizers.get(identifier)
