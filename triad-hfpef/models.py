# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model library for HFpEF prediction."""

from typing import Optional
import tensorflow as tf


def resnet50v2(
    image_count: int,
    target_count: int = 1,
    main_target_weight: float = 1.0,
    pretrain_weights: Optional[str] = 'imagenet',
    name: str = 'resnet50v2',
) -> tf.keras.Model:
  """ResNet50 V2 based model for multiple images for classification.

  The inputs must be of shape (batch, image_count, 224, 224, 3).

  Args:
    image_count: The number of images in a video.
    target_count: The number of targets to predict. All prediction targets must
      be either binary or "probabilities" between 0 and 1.
    main_target_weight: The weight of the main target, which is assumed to be
      the first target. Ignored when target_count is 1. For example, if
      target_count is 3 and main_target_weight is 0.5, then the loss is 0.5 *
      loss_target_0 + 0.25 * loss_target_1 + 0.25 * loss_target_2.
    pretrain_weights: Pretrain weights to use. 'imagenet' or None.
    name: The name of the model.

  Returns:
    A Keras ResNet50 V2 model for binary classification.
  """
  if pretrain_weights and pretrain_weights != 'imagenet':
    raise ValueError(f'Unsupported pretrain_weights: {pretrain_weights}')
  if main_target_weight < 0.0 or main_target_weight > 1.0:
    raise ValueError('Main target weight must be between 0 and 1.')
  elif target_count == 1 and main_target_weight != 1.0:
    raise ValueError(
        'Main target weight must be 1.0 when there is only one target.'
    )
  elif target_count > 1 and main_target_weight == 1.0:
    raise ValueError(
        'Main target weight must be <1 when there are multiple targets.'
    )
  image_shape = [224, 224, 3]
  image_inputs = tf.keras.Input(
      shape=[image_count] + image_shape, name=f'{name}_input'
  )
  label_inputs = [
      tf.keras.Input(shape=[1], name=f'{name}_label_{i}')
      for i in range(target_count)
  ]
  x = image_inputs
  shared_backbone = tf.keras.applications.ResNet50V2(
      include_top=True,
      weights=pretrain_weights,
  )
  output_lists = []
  for i in range(image_count):
    x_i = tf.gather(x, indices=i, axis=1, name=f'{name}_slice{i}')
    output_lists.append(shared_backbone(x_i))

  x = tf.keras.layers.Concatenate(name=f'{name}_concat', axis=-1)(output_lists)
  outputs = [
      tf.keras.layers.Dense(
          units=1,
          activation='sigmoid',
          name=f'{name}_pred_{i}',
      )(x)
      for i in range(target_count)
  ]
  assert len(outputs) == len(label_inputs)
  model = tf.keras.Model(
      inputs=[image_inputs] + label_inputs, outputs=outputs, name=name
  )
  losses = [
      tf.keras.backend.binary_crossentropy(
          label_inputs[i],
          outputs[i],
          from_logits=False,
      )
      for i in range(target_count)
  ]
  if target_count == 1:
    weights = [1.0]
  else:
    weights = [main_target_weight] + [
        (1 - main_target_weight) / (target_count - 1)
    ] * (target_count - 1)
  assert len(losses) == len(weights)
  combined_loss = 0
  for i in range(target_count):
    model.add_metric(
        losses[i],
        name=f'binary_crossentropy_{i}',
    )
    combined_loss += weights[i] * losses[i]
  model.add_loss(combined_loss)
  return model
