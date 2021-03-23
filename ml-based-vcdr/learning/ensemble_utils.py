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
r"""Utilities for constructing and evaluating ensembles."""
import collections
import pathlib
import re
from typing import List, Tuple

import metrics
import ml_collections
import model_utils
import tensorflow as tf


def _build_ensemble_graph(models: List[tf.keras.Model],
                          config: ml_collections.ConfigDict) -> tf.keras.Model:
  """Builds an ensemble graph that outputs the mean of member predictions."""
  inputs = tf.keras.Input(shape=config.model.input_shape, name='image')

  # Group model outputs by outcome layer name.
  grouped_outputs = collections.defaultdict(list)
  for model in models:
    for output_name, output_tensor in zip(model.output_names, model(inputs)):
      grouped_outputs[output_name].append(output_tensor)

  # Average model outputs to create ensemble outputs.
  averaged_outputs = []
  for output_name, grouped_output in grouped_outputs.items():
    average_layer = tf.keras.layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=0), name=output_name)
    averaged_outputs.append(average_layer(grouped_output))

  # Build a new model from the common input image and ensembled outputs.
  model = tf.keras.Model(
      inputs=[inputs],
      outputs=averaged_outputs,
      name=f'{config.model.backbone}_ensemble_{len(models)}_members')
  model.summary()

  return model


def build_ensemble(models: List[tf.keras.Model],
                   config: ml_collections.ConfigDict) -> tf.keras.Model:
  """Compiles and returns an averaged ensemble model."""

  model = _build_ensemble_graph(models, config)

  losses = {}
  loss_weights = {}
  for outcome in config.outcomes:
    losses[outcome.name] = metrics.get_loss(outcome)
    loss_weights[outcome.name] = outcome.get('loss_weight', 1.0)

  model.compile(
      optimizer=model_utils.get_optimizer(config),
      loss=losses,
      loss_weights=loss_weights,
      metrics=metrics.get_metrics(config.outcomes))

  return model


def rename_member_layers(checkpoint_path: pathlib.Path,
                         model: tf.keras.Model) -> tf.keras.Model:
  """Renames each model layer to avoid conflicts when building the ensemble."""
  model_dir_regex = re.compile(r'.+/(.+)/checkpoints.*')
  match = model_dir_regex.match(str(checkpoint_path))
  model_dir = match.group(1)
  model._name = f'{model_dir}_{model._name}'
  for layer in model.layers:
    layer._name = f'{model_dir}_{layer._name}'
  return model


def load_model(
    checkpoint_dir: pathlib.Path, config: ml_collections.ConfigDict
) -> Tuple[ml_collections.ConfigDict, tf.keras.Model]:
  """Loads a model using the best checkpoint at the given path."""
  _, model = model_utils.get_model(config)

  best_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  print(f'Loading model from {best_checkpoint}...')
  model.load_weights(best_checkpoint).expect_partial()

  return model


def load_models(checkpoint_paths: List[pathlib.Path],
                config: ml_collections.ConfigDict) -> List[tf.keras.Model]:
  """Loads all models from the given list of checkpoint paths."""
  return [load_model(path, config) for path in checkpoint_paths]


def get_checkpoint_dirs(base_workdir: pathlib.Path) -> List[pathlib.Path]:
  """Returns a list of paths containing each ensemble checkpoint directory."""
  return [dir_path / 'checkpoints' for dir_path in base_workdir.iterdir()]
