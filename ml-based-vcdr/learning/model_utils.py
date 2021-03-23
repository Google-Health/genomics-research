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
"""Utility functions for building models and optimizers."""
import copy
import os
import tempfile
from typing import List, Tuple
import warnings

import metrics
import ml_collections
import tensorflow as tf
import tensorflow_addons as tfa

DEFAULT_IMAGE_SHAPE = (587, 587, 3)

# Silence `tfa.optimizers.MovingAverage` BatchNormalization warnings.
warnings.filterwarnings('ignore')


def build_outcome_head(config: ml_collections.ConfigDict,
                       inputs: tf.Tensor,
                       l2: float = 0.0) -> tf.Tensor:
  """Returns an output head tensor configured for the given outcome.

  Supports regression, binary classification, and multinomial classification
  outcomes.

  Note: binary classification labels are assumed to be of shape (2,). Binary
  heads consist of a `tf.keras.layers.Dense(2)` with a softmax activation.

  Args:
    config: An outcome ConfigDict.
    inputs: The backbone output tensor; used as the input to the head.
    l2: The l2 regularization factor used in `tf.keras.layers.Dense` layers.

  Returns:
    A tensor representing the output of the given head.

  Raises:
    ValueError: `config` missing a valid `type`.
    ValueError: A binary classification config uses num_classes=1 rather than
    num_classes=2.
  """
  outcome_type = config.get('type', None)
  if outcome_type is None:
    raise ValueError(f'Provided `config` missing `type`: {config}')

  l2_regularizer = tf.keras.regularizers.L2(l2) if l2 else None

  if outcome_type == 'regression':
    head = tf.keras.layers.Dense(
        1,
        dtype=tf.float32,
        name=config.name,
        kernel_regularizer=l2_regularizer)
    return head(inputs)

  if outcome_type == 'classification':
    if config.num_classes < 2:
      raise ValueError('Binary heads should specify `config.num_classes=2`.'
                       'Binary labels are assumed to be one-hot vectors.')

    head = tf.keras.layers.Dense(
        config.num_classes,
        activation='softmax',
        dtype=tf.float32,
        name=config.name,
        kernel_regularizer=l2_regularizer)
    return head(inputs)

  raise ValueError(f'Unknown outcome type: {outcome_type}')


def inceptionv3(model_config: ml_collections.ConfigDict,
                outcomes: List[ml_collections.ConfigDict]) -> tf.keras.Model:
  """Returns an InceptionV3 architecture as defined by the configuration.

  See https://tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3.

  Args:
    model_config: A ConfigDict containing model hyperparamters.
    outcomes: A list of outcome ConfigDict instances.

  Returns:
    An InceptionV3-based model.
  """
  input_shape = model_config.get('input_shape', DEFAULT_IMAGE_SHAPE)

  backbone = tf.keras.applications.InceptionV3(
      include_top=False,
      weights=model_config.get('weights', 'imagenet'),
      input_shape=input_shape,
      pooling=model_config.get('pooling', 'avg'))
  weight_decay = model_config.get('weight_decay', 0.0)
  if weight_decay:
    backbone = add_l2_regularizers(
        backbone, tf.keras.layers.Conv2D, l2=weight_decay)
  backbone_drop_rate = model_config.get('backbone_drop_rate', 0.2)

  inputs_image = tf.keras.Input(shape=input_shape, name='image')
  hid = backbone(inputs_image)
  hid = tf.keras.layers.Dropout(backbone_drop_rate)(hid)

  outputs = []
  for outcome in outcomes:
    outputs.append(build_outcome_head(outcome, hid, l2=weight_decay))

  model = tf.keras.Model(
      inputs=[inputs_image], outputs=outputs, name=model_config.backbone)
  model.summary()
  print(f'Number of l2 regularizers: {len(model.losses)}.')
  return model


def build_model_graph(
    config: ml_collections.ConfigDict,
    outcomes: List[ml_collections.ConfigDict]) -> tf.keras.Model:
  """Returns a tf.keras.Model configured with the given ConfigDict.

  Args:
    config: A ConfigDict containing a `config.model` sub-config.
    outcomes: A list of outcome ConfigDict instances.

  Returns:
    A tf.keras.Model.

  Raises:
    ValueError: `config` missing the `model` sub-config.
    ValueError: `config.model` contains an invalid model backbone.
  """
  model_config = config.get('model', None)
  if model_config is None:
    raise ValueError(f'Provided `config` missing `model` sub-config: {config}')

  model_backbone = model_config.get('backbone', None)

  # Config specifies model choice.
  if model_backbone == 'inceptionv3':
    return inceptionv3(model_config, outcomes)

  raise ValueError(f'Unknown model backbone: {model_backbone}')


def get_optimizer(
    config: ml_collections.ConfigDict) -> tf.keras.optimizers.Optimizer:
  """Returns an optimizer based on the given configuration.

  Supports the Adam optimizer. Default values for optional optimizer parameters
  come from TensorFlow Core v2.4.1:
  https://www.tensorflow.org/api_docs/python/tf/keras/optimizers.

  Args:
    config: A ConfigDict containing a `config.opt` sub-config.

  Returns:
    A tf.keras optimizer.

  Raises:
    ValueError: `config` missing the `opt` sub-config.
    ValueError: `config.opt` contains an invalid optimizer class.
    ValueError: `config.schedule` contains an invalid schedule class.
  """
  opt_config = config.get('opt', None)
  if opt_config is None:
    raise ValueError(f'Provided `config` missing `opt` sub-config: {config}')

  initial_learning_rate = opt_config.get('learning_rate', 0.001)
  steps_per_epoch = config.dataset.num_train_examples / config.dataset.batch_size

  schedule_config = config.get('schedule', {})
  schedule_type = schedule_config.get('schedule', None)
  if schedule_type == 'exponential':
    decay_steps = int(schedule_config.epochs_per_decay * steps_per_epoch)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=schedule_config.decay_rate,
        staircase=schedule_config.staircase)
  elif schedule_type is None:
    learning_rate = initial_learning_rate
    print(f'No LR schedule provided. Using a fixed LR of "{learning_rate}".')
  else:
    raise ValueError(f'Unknown scheduler name: {schedule_type}')

  opt_type = opt_config.get('optimizer', None)
  if opt_type == 'adam':
    opt = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=opt_config.get('beta_1', 0.9),
        beta_2=opt_config.get('beta_2', 0.999),
        epsilon=opt_config.get('epsilon', 1e-07),
        amsgrad=opt_config.get('amsgrad', False))
    start_step = int(steps_per_epoch * config.train.initial_epoch)
    if opt_config.get('use_model_averaging', True):
      opt = tfa.optimizers.MovingAverage(
          opt, average_decay=0.9999, start_step=start_step)
    return opt

  raise ValueError(f'Unknown optimizer name: {opt_type}')


def add_l2_regularizers(model: tf.keras.Model,
                        layer_class: tf.keras.layers.Layer,
                        l2: float = 0.00004,
                        regularizer_attr: str = 'kernel_regularizer'):
  """Adds L2 regularizers to all `layer_class` layers in `model`.

  Models from `tf.keras.applications` do not support specifying kernel or bias
  regularizers. However, adding regularization is important when fine tuning
  'imagenet' pretrained weights. In order to do this, this function updates the
  current model's configuration to include regularizers and reloads the model so
  that the newly created losses are registered.

  Note: this will not overwrite existing `kernel_regularizer` regularizers on
  the given layer.

  Args:
    model: The base model.
    layer_class: We add regularizers to all layers of type `layer_class`.
    l2: The l2 regularization factor.
    regularizer_attr: The layer's regularizer attribute.

  Returns:
    A model with l2 regularization added to each `layer_class` layer.
  """
  # Save the original model weights.
  tmp_weights_dir = tempfile.gettempdir()
  tmp_weights_path = os.path.join(tmp_weights_dir, 'tmp_weights.h5')
  model.save_weights(tmp_weights_path)

  # Clone the original model.
  reg_model = tf.keras.models.clone_model(model)

  # Set the L2 `regularizer_attr` on all layers of type `layer_class`. This
  # change is only reflected in the model's config file.
  num_regularizers_added = 0
  for layer in reg_model.layers:
    if not isinstance(layer, layer_class):
      continue
    if not hasattr(layer, regularizer_attr):
      continue
    if getattr(layer, regularizer_attr) is not None:
      continue
    setattr(layer, regularizer_attr, tf.keras.regularizers.l2(l2=l2))
    num_regularizers_added += 1

  # Save the updated model configuration.
  reg_model_json = reg_model.to_json()

  # Create a "new" model from the updated configuration and load the original
  # model's weights.
  reg_model = tf.keras.models.model_from_json(reg_model_json)
  reg_model.load_weights(tmp_weights_path, by_name=True)

  # Ensure model weights have not changed after adding regularization layers.
  for layer, reg_layer in zip(model.layers, reg_model.layers):
    weights = layer.weights
    reg_weights = reg_layer.weights
    if not weights:
      assert not reg_weights
    else:
      for i, weight in enumerate(weights):
        tf.debugging.assert_near(weight, reg_weights[i])

  # Ensure the newly added regularizers are registered as losses.
  assert len(reg_model.losses) == (len(model.losses) + num_regularizers_added)

  return reg_model


def compile_model(config: ml_collections.ConfigDict) -> tf.keras.Model:
  """Builds a graph and compiles a tf.keras.Model based on the configuration."""

  model = build_model_graph(config, config.outcomes)

  losses = {}
  loss_weights = {}
  for outcome in config.outcomes:
    losses[outcome.name] = metrics.get_loss(outcome)
    loss_weights[outcome.name] = outcome.get('loss_weight', 1.0)

  model.compile(
      optimizer=get_optimizer(config),
      loss=losses,
      loss_weights=loss_weights,
      weighted_metrics=metrics.get_metrics(config.outcomes))

  return model


def get_model(
    config: ml_collections.ConfigDict
) -> Tuple[ml_collections.ConfigDict, tf.keras.Model]:
  """Returns a compiled model.

  If `config.train.use_distributed_training` is set to `True`, the model and its
  metrics are compiled under a `tf.distribute.MirroredStrategy` scope, allowing
  for distributed training.

  Note: If using distributed training, a subset of hyperparameters are scaled
  based on the number of model replicas. We linearly increase global batch size
  and learning rate, and decrease the maximum number of steps and logging
  frequency.

  Args:
    config: The experiment configuration.

  Returns:
    A compiled tf.keras.Model and a (potentially) modified ConfigDict.
  """
  config = copy.deepcopy(config)

  if config.train.get('use_distributed_training', False):

    mirrored_strategy = tf.distribute.MirroredStrategy()
    num_gpus = mirrored_strategy.num_replicas_in_sync

    # Update hparams that are dependent on the number of GPUs.
    config.dataset.batch_size *= num_gpus
    config.opt.learning_rate *= num_gpus  #  https://arxiv.org/abs/1706.02677
    config.train.max_num_steps = int(config.train.max_num_steps / num_gpus)
    config.train.log_step_freq = int(config.train.log_step_freq / num_gpus)

    with mirrored_strategy.scope():
      model = compile_model(config)

  else:

    model = compile_model(config)

  return config, model
