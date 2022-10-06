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
"""Custom `tf.losses.Loss` implementations and loaders."""
import ml_collections
import tensorflow as tf


def get_loss(loss_config: ml_collections.ConfigDict) -> tf.losses.Loss:
  """Returns a loss for use in model training and evaluation.

  The `loss_config` `ConfigDict` must match the config schema defined in the
  README.

  Note: This function wraps `tf.losses.get()`. Any `tf.losses` class from
  https://www.tensorflow.org/api_docs/python/tf/keras/losses can be loaded with
  this utility. See individual class implementations for the set of available
  keyword arguments that can be configured via `loss_config.kwargs`.

  For example:

  ```
  valid_loss_config = ml_collections.ConfigDict({
      'class_name': 'BinaryCrossentropy',
      'kwargs': {
          'from_logits': True,
          'label_smoothing': 0.2,
      },
  })
  loss = get_loss(valid_loss_config)
  ```

  Args:
    loss_config: A ConfigDict matching the `loss_config` schema.

  Returns:
    A loss instance.
  """
  # Build an identifier dict that can be parsed by `tf.losses.get`.
  identifier_kwargs = loss_config.get('kwargs', ml_collections.ConfigDict())
  identifier = {
      'class_name': loss_config['class_name'],
      'config': identifier_kwargs.to_dict()
  }
  return tf.losses.get(identifier)
